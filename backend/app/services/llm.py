"""LLM gateway — grounded explanation generation with validation, fallback,
and Redis caching (spec §7.14, §13).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import redis
from openai import OpenAI

from app.core.config import settings
from app.schemas.api import CaseExplanation, ExplainResponse

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is retrieved historical evidence for clinical decision support. "
    "It is not a diagnosis or treatment recommendation. "
    "All medical decisions remain the responsibility of the treating physician."
)

BANNED_PATTERNS = [
    re.compile(r"\bthe (query )?patient (has|is suffering from|should)\b", re.IGNORECASE),
    re.compile(r"\b(you|the clinician) should (prescribe|administer|order|start)\b", re.IGNORECASE),
    re.compile(r"\bdiagnosis (for the query patient )?is\b", re.IGNORECASE),
    re.compile(r"\bwe (diagnose|recommend treating)\b", re.IGNORECASE),
]

PROMPT_TEMPLATE = """You are a clinical evidence assistant inside a Clinical Decision Support System.
Your ONLY job: explain why each retrieved historical case is similar to the query, using ONLY the provided case texts.

STRICT RULES:
1. Use ONLY facts present in the provided case texts. Never use outside medical knowledge to assert facts.
2. Every factual statement MUST cite its source case using its exact ID, e.g. [acn-12345].
3. NEVER diagnose the query patient. NEVER recommend treatments. You describe historical cases only.
4. If the cases contain insufficient information for a field, write "insufficient evidence in retrieved cases".
5. Output VALID JSON only, exactly matching the schema below. No markdown, no prose outside JSON.

QUERY (description of the current patient/situation):
{query}

RETRIEVED CASES:
{cases_block}

OUTPUT JSON SCHEMA:
{{
  "explanations": [
    {{
      "case_id": "<id>",
      "similarity_factors": [
        {{"factor": "<shared clinical feature>", "detail": "<specifics>", "citations": ["<case_id>"]}}
      ],
      "differences": [
        {{"detail": "<how this case differs from the query>", "citations": ["<case_id>"]}}
      ],
      "treatments_observed": [
        {{"treatment": "<treatment given in this historical case>", "outcome_note": "<documented response>", "citations": ["<case_id>"]}}
      ],
      "confidence": "high|moderate|weak"
    }}
  ],
  "cohort_observation": "<1-3 sentences on patterns across ALL retrieved cases, fully cited>"
}}"""

PROMPT_VERSION = "similarity_explanation_v1"


class LLMGateway:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)
        self._redis: redis.Redis | None = None

    def connect_cache(self) -> None:
        try:
            self._redis = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
            self._redis.ping()
            logger.info("Redis cache connected")
        except Exception:
            self._redis = None
            logger.warning("Redis unavailable — running without explanation cache")

    def cache_healthy(self) -> bool:
        try:
            return bool(self._redis and self._redis.ping())
        except Exception:
            return False

    # ------------------------------------------------------------ context

    @staticmethod
    def build_context(documents: dict[str, str], metadata: dict[str, dict]) -> str:
        blocks = []
        for cid, doc in documents.items():
            meta = metadata.get(cid, {})
            blocks.append(
                f"=== CASE {cid} (sex: {meta.get('sex')}, age: {meta.get('age')}, "
                f"outcome: {meta.get('outcome_class')}) ===\n{doc[:settings.max_case_context_chars]}"
            )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------ generation

    def explain(self, query: str, documents: dict[str, str], metadata: dict[str, dict]) -> ExplainResponse:
        cache_key = self._cache_key(query, documents)
        cached = self._cache_get(cache_key)
        if cached:
            cached["cached"] = True
            return ExplainResponse(**cached)

        prompt = PROMPT_TEMPLATE.format(query=query, cases_block=self.build_context(documents, metadata))
        valid_ids = set(documents.keys())

        attempts = [settings.llm_model_primary, settings.llm_model_primary, settings.llm_model_fallback]
        for model_name in attempts:
            try:
                raw = self._call(prompt, model_name)
            except Exception as e:
                logger.warning("LLM provider error on %s: %s", model_name, e)
                continue
            parsed = self._extract_json(raw)
            if parsed is None:
                logger.warning("Invalid JSON from %s", model_name)
                continue
            problems = self._validate(parsed, valid_ids)
            if problems:
                logger.warning("Validation failed (%s): %s", model_name, problems[:3])
                continue

            response = ExplainResponse(
                query=query,
                explanations=[CaseExplanation(**e) for e in parsed["explanations"]],
                cohort_observation=parsed.get("cohort_observation", ""),
                disclaimer=DISCLAIMER,
                model_used=model_name,
                degraded=False,
                cached=False,
                took_ms=0,  # set by the route
            )
            self._cache_set(cache_key, response)
            return response

        # graceful degradation (spec §13.5)
        logger.error("All LLM attempts failed — returning degraded response")
        return ExplainResponse(
            query=query,
            explanations=[CaseExplanation(case_id=cid, confidence="weak") for cid in documents],
            cohort_observation="LLM explanation unavailable — structured retrieval results only.",
            disclaimer=DISCLAIMER,
            model_used="none",
            degraded=True,
            cached=False,
            took_ms=0,
        )

    # ------------------------------------------------------------ internals

    def _call(self, prompt: str, model_name: str) -> str:
        resp = self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            top_p=0.7,
            max_tokens=settings.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _validate(output: dict, valid_ids: set[str]) -> list[str]:
        problems = []
        if not isinstance(output.get("explanations"), list) or not output["explanations"]:
            return ["missing explanations array"]
        for exp in output["explanations"]:
            cid = exp.get("case_id", "?")
            if cid not in valid_ids:
                problems.append(f"unknown case_id {cid}")
            for section in ("similarity_factors", "differences", "treatments_observed"):
                for item in exp.get(section) or []:
                    cites = item.get("citations", [])
                    if not cites:
                        problems.append(f"{cid}/{section}: uncited claim")
                    problems.extend(
                        f"{cid}/{section}: bad citation {c}" for c in cites if c not in valid_ids
                    )
        blob = json.dumps(output)
        problems.extend(
            f"banned phrasing: {p.pattern}" for p in BANNED_PATTERNS if p.search(blob)
        )
        return problems

    # ------------------------------------------------------------ cache

    @staticmethod
    def _cache_key(query: str, documents: dict[str, str]) -> str:
        content = json.dumps({"q": query, "ids": sorted(documents), "v": PROMPT_VERSION}, sort_keys=True)
        return "explain:" + hashlib.sha256(content.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        if not self._redis:
            return None
        try:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _cache_set(self, key: str, response: ExplainResponse) -> None:
        if not self._redis:
            return
        try:
            self._redis.setex(key, settings.cache_ttl_seconds, response.model_dump_json())
        except Exception:
            pass


llm_gateway = LLMGateway()
