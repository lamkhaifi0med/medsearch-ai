"""MedSearch AI — gold query set builder (spec §22, Phase 4 exit criteria).

Builds a synthetic-but-realistic gold retrieval set:
  1. Stratified-sample N cases from the cleaned corpus (by outcome_class, seed 42).
  2. For each case, an LLM writes a short clinician-style search query describing
     the patient's presentation — paraphrased, never copied verbatim, so the
     sparse (lexical) retriever gets no free exact-match advantage.
  3. The source case is the known-relevant document (self-retrieval protocol).

Output: evaluation/gold_queries.jsonl
  {"query_id", "query", "relevant_case_ids", "sex", "age", "outcome_class"}

Usage:
    python evaluation/build_gold_set.py            # 100 queries
    python evaluation/build_gold_set.py -n 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CASES_FILE = ROOT / "data" / "processed" / "cases_clean.jsonl"
OUT_FILE = ROOT / "evaluation" / "gold_queries.jsonl"

PRIMARY_MODEL = os.environ.get("LLM_MODEL_PRIMARY", "meta/llama-3.3-70b-instruct")
FALLBACK_MODEL = os.environ.get("LLM_MODEL_FALLBACK", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
SEED = 42

PROMPT = """You are helping build a search-quality benchmark for a clinical case retrieval system.

Below is a clinical case report. Write ONE short search query (15-30 words) that a physician
seeing a NEW patient with a similar presentation would type into the system.

RULES:
- Describe the presentation: demographics, chief complaints, key findings.
- PARAPHRASE everything. Do NOT reuse distinctive multi-word phrases verbatim from the text.
- Use common clinical vocabulary a physician would naturally type (synonyms are good).
- No patient identifiers, no hospital names, no rare proper nouns.
- Do NOT mention the final diagnosis by name if one is stated; describe the picture instead.
- Output ONLY the query text. No quotes, no preamble.

CASE:
{case_text}"""


def stratified_sample(cases: list[dict], n: int) -> list[dict]:
    """Sample proportionally to outcome_class distribution, seed fixed."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_class[c["outcome_class"]].append(c)
    rng = random.Random(SEED)
    total = len(cases)
    picked: list[dict] = []
    for cls, pool in sorted(by_class.items()):
        quota = max(1, round(n * len(pool) / total))
        picked.extend(rng.sample(pool, min(quota, len(pool))))
    rng.shuffle(picked)
    return picked[:n]


def make_query(client: OpenAI, case: dict) -> str | None:
    text = case["document"][:3500]
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(case_text=text)}],
                    temperature=0.7,
                    max_tokens=100,
                    timeout=120,  # free tier queues; never hang forever
                )
                q = (resp.choices[0].message.content or "").strip().strip('"').strip()
                q = re.sub(r"\s+", " ", q)
                # sanity: reasonable length, not a refusal / preamble
                if 8 <= len(q.split()) <= 45 and not q.lower().startswith(("i ", "sorry", "here")):
                    return q
            except Exception as e:  # noqa: BLE001 — retry then fall through to fallback model
                msg = str(e)
                if "429" in msg:  # rate limited: exponential backoff
                    wait = 20 * (attempt + 1)
                    print(f"    [429] backing off {wait}s", file=sys.stderr, flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [{model} attempt {attempt+1}] {msg[:150]}", file=sys.stderr, flush=True)
                    time.sleep(2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="number of gold queries")
    parser.add_argument("--workers", type=int, default=3, help="concurrent LLM requests")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("NVIDIA_API_KEY missing from .env")
    client = OpenAI(base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=api_key)

    print(f"Loading corpus from {CASES_FILE.name} ...")
    cases = [json.loads(line) for line in open(CASES_FILE, encoding="utf-8")]
    sample = stratified_sample(cases, args.n)
    print(f"Sampled {len(sample)} cases (stratified by outcome_class, seed {SEED})")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    written = 0

    # Resume support: skip cases already in the output file
    existing: set[str] = set()
    if OUT_FILE.exists():
        for line in open(OUT_FILE, encoding="utf-8"):
            try:
                existing.add(json.loads(line)["relevant_case_ids"][0])
            except Exception:  # noqa: BLE001
                pass
        written = len(existing)
        if existing:
            print(f"Resuming: {written} queries already generated", flush=True)
    todo = [c for c in sample if c["case_id"] not in existing]

    def worker(case: dict) -> tuple[dict, str | None]:
        return case, make_query(client, case)

    with open(OUT_FILE, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case, q in pool.map(worker, todo):
            done += 1
            if q is None:
                print(f"  [{done}/{len(todo)}] SKIP {case['case_id']}", flush=True)
                continue
            written += 1
            rec = {
                "query_id": f"gq-{written:03d}",
                "query": q,
                "relevant_case_ids": [case["case_id"]],
                "sex": case.get("sex"),
                "age": case.get("age"),
                "outcome_class": case.get("outcome_class"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # incremental — survives interruption, enables resume
            print(f"  [{done}/{len(todo)}] ok={written} | {time.time()-t0:.0f}s | {q[:60]}", flush=True)

    print(f"\nDone: {written} gold queries in {time.time()-t0:.0f}s -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
