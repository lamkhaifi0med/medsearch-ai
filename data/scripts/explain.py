"""MedSearch AI — grounded similarity explanation engine (spec §13).

Pipeline: hybrid search (Qdrant) -> fetch full case texts -> build labeled,
token-budgeted context -> grounded prompt -> NVIDIA NIM (Llama 3.3 70B) ->
JSON schema validation + citation checks + banned-phrase screen -> render.

Usage:
    python data/scripts/explain.py "elderly man with heart failure and leg swelling" -k 3
    python data/scripts/explain.py "seizures in a child" -k 3 --outcome improved --age-max 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

PROCESSED = ROOT / "data" / "processed"
COLLECTION = "cases_v1"
QDRANT_URL = "http://localhost:6333"

PRIMARY_MODEL = os.environ.get("LLM_MODEL_PRIMARY", "meta/llama-3.3-70b-instruct")
FALLBACK_MODEL = os.environ.get("LLM_MODEL_FALLBACK", "nvidia/llama-3.3-nemotron-super-49b-v1.5")

MAX_CASE_CHARS = 4500          # per-case context budget (~1100 tokens)
DISCLAIMER = ("This is retrieved historical evidence for clinical decision support. "
              "It is not a diagnosis or treatment recommendation. "
              "All medical decisions remain the responsibility of the treating physician.")

# Diagnostic/prescriptive phrasings about the QUERY patient are forbidden (spec §13.5)
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


# ---------------------------------------------------------------- retrieval

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading BGE-M3 for query embedding...", file=sys.stderr)
        from FlagEmbedding import BGEM3FlagModel
        _embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _embed_model


def hybrid_search(query: str, k: int, sex: str | None, outcome: str | None,
                  age_min: int | None, age_max: int | None):
    model = get_embed_model()
    enc = model.encode([query], return_dense=True, return_sparse=True, max_length=512)
    dense = enc["dense_vecs"][0].tolist()
    sp = enc["lexical_weights"][0]
    sparse = models.SparseVector(indices=[int(i) for i in sp.keys()],
                                 values=[float(v) for v in sp.values()])

    must = [models.FieldCondition(key="embedding_version", match=models.MatchValue(value="bgem3-v1"))]
    if sex:
        must.append(models.FieldCondition(key="sex", match=models.MatchValue(value=sex)))
    if outcome:
        must.append(models.FieldCondition(key="outcome_class", match=models.MatchValue(value=outcome)))
    if age_min is not None or age_max is not None:
        must.append(models.FieldCondition(key="age", range=models.Range(gte=age_min, lte=age_max)))
    qfilter = models.Filter(must=must)

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    result = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", filter=qfilter, limit=50),
            models.Prefetch(query=sparse, using="sparse", filter=qfilter, limit=50),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return result.points


def load_case_texts(case_ids: set[str]) -> dict[str, str]:
    """Stream the corpus and pull full documents for the retrieved IDs."""
    texts = {}
    with (PROCESSED / "cases_clean.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["case_id"] in case_ids:
                texts[rec["case_id"]] = rec["document"]
                if len(texts) == len(case_ids):
                    break
    return texts


# ---------------------------------------------------------------- context + LLM

def build_context(points, texts: dict[str, str]) -> str:
    blocks = []
    for point in points:
        p = point.payload
        cid = p["case_id"]
        doc = texts.get(cid, "")[:MAX_CASE_CHARS]
        blocks.append(
            f"=== CASE {cid} (sex: {p.get('sex')}, age: {p.get('age')}, "
            f"outcome: {p.get('outcome_class')}, retrieval score: {point.score:.3f}) ===\n{doc}"
        )
    return "\n\n".join(blocks)


def call_llm(prompt: str, model_name: str) -> str:
    client = OpenAI(base_url=os.environ["NVIDIA_BASE_URL"], api_key=os.environ["NVIDIA_API_KEY"])
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,       # low temp for factual, grounded output (spec §13.5)
        top_p=0.7,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def extract_json(raw: str) -> dict | None:
    """Parse model output; tolerate accidental markdown fences."""
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


def validate(output: dict, valid_ids: set[str]) -> list[str]:
    """Return list of validation problems (empty = valid). Spec §13.8."""
    problems = []
    if "explanations" not in output or not isinstance(output["explanations"], list):
        return ["missing 'explanations' array"]
    for exp in output["explanations"]:
        cid = exp.get("case_id", "?")
        if cid not in valid_ids:
            problems.append(f"unknown case_id {cid}")
        for section in ("similarity_factors", "differences", "treatments_observed"):
            for item in exp.get(section, []) or []:
                cites = item.get("citations", [])
                if not cites:
                    problems.append(f"{cid}/{section}: uncited claim")
                for c in cites:
                    if c not in valid_ids:
                        problems.append(f"{cid}/{section}: citation to unknown case {c}")
    text_blob = json.dumps(output)
    for pattern in BANNED_PATTERNS:
        if pattern.search(text_blob):
            problems.append(f"banned diagnostic/prescriptive phrasing: {pattern.pattern}")
    return problems


# ---------------------------------------------------------------- rendering

def render(output: dict, points) -> None:
    score_by_id = {p.payload["case_id"]: p.score for p in points}
    print("\n" + "=" * 80)
    for exp in output.get("explanations", []):
        cid = exp.get("case_id", "?")
        print(f"\nCASE {cid}  (retrieval score {score_by_id.get(cid, 0):.3f}, "
              f"LLM confidence: {exp.get('confidence', '?')})")
        print("-" * 80)
        print("  WHY SIMILAR:")
        for f in exp.get("similarity_factors", []) or []:
            print(f"   + {f.get('factor')}: {f.get('detail')}  {f.get('citations')}")
        if exp.get("differences"):
            print("  DIFFERENCES:")
            for d in exp["differences"]:
                print(f"   - {d.get('detail')}  {d.get('citations')}")
        if exp.get("treatments_observed"):
            print("  TREATMENTS OBSERVED (historical):")
            for t in exp["treatments_observed"]:
                print(f"   * {t.get('treatment')} -> {t.get('outcome_note')}  {t.get('citations')}")
    if output.get("cohort_observation"):
        print("\nCOHORT OBSERVATION:")
        print(f"  {output['cohort_observation']}")
    print("\n" + "=" * 80)
    print(f"DISCLAIMER: {DISCLAIMER}")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Search + grounded LLM explanation")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=3, help="cases to explain (keep small: context budget)")
    parser.add_argument("--sex", choices=["male", "female"])
    parser.add_argument("--outcome", choices=["improved", "deteriorated", "deceased", "unknown"])
    parser.add_argument("--age-min", type=int)
    parser.add_argument("--age-max", type=int)
    args = parser.parse_args()

    print(f"Searching: {args.query!r}", file=sys.stderr)
    points = hybrid_search(args.query, args.k, args.sex, args.outcome, args.age_min, args.age_max)
    if not points:
        sys.exit("No cases retrieved — relax the filters.")
    ids = {p.payload["case_id"] for p in points}
    print(f"Retrieved: {', '.join(sorted(ids))}", file=sys.stderr)

    texts = load_case_texts(ids)
    prompt = PROMPT_TEMPLATE.format(query=args.query, cases_block=build_context(points, texts))

    for attempt, model_name in enumerate([PRIMARY_MODEL, PRIMARY_MODEL, FALLBACK_MODEL], 1):
        print(f"LLM call {attempt} ({model_name})...", file=sys.stderr)
        t0 = time.time()
        try:
            raw = call_llm(prompt, model_name)
        except Exception as e:
            print(f"  provider error: {e}", file=sys.stderr)
            continue
        output = extract_json(raw)
        if output is None:
            print("  invalid JSON, retrying...", file=sys.stderr)
            continue
        problems = validate(output, ids)
        if problems:
            print(f"  validation failed: {problems[:3]}", file=sys.stderr)
            continue
        print(f"  ok ({time.time()-t0:.1f}s)", file=sys.stderr)
        render(output, points)
        return

    # graceful degradation (spec §13.5): structured results without LLM prose
    print("\nLLM validation failed after retries — structured results only:", file=sys.stderr)
    for p in points:
        pl = p.payload
        print(f"  {pl['case_id']} score={p.score:.3f} [{pl.get('sex')}, {pl.get('age')}, {pl.get('outcome_class')}]")
    print(f"\nDISCLAIMER: {DISCLAIMER}")


if __name__ == "__main__":
    main()
