"""End-to-end gold-set evaluation through the LIVE API (production path).

Unlike run_eval.py (which talks to Qdrant directly), this measures the full
production stack: FastAPI -> embedding -> hybrid fusion -> [negation layer]
-> [optional rerank]. Use it to verify that service-layer changes don't
regress retrieval quality.

Usage:
    python evaluation/eval_api.py            # fast mode
    python evaluation/eval_api.py --rerank   # thorough mode
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

API = "http://localhost:8000/api/v1/search"
GOLD = Path(__file__).parent / "gold_queries.jsonl"
K = 10


def search(query: str, rerank: bool) -> dict:
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query, "k": K, "rerank": rerank, "filters": {}}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    gold = [json.loads(l) for l in GOLD.open(encoding="utf-8")]
    print(f"{len(gold)} gold queries | mode: {'thorough' if args.rerank else 'fast'}")

    r1 = r5 = r10 = mrr = ndcg = 0.0
    reranked_count = 0
    t_all = time.time()
    for i, g in enumerate(gold, 1):
        resp = search(g["query"], args.rerank)
        if resp.get("reranked"):
            reranked_count += 1
        ids = [c["case_id"] for c in resp["results"]]
        relevant = set(g["relevant_case_ids"])
        rank = next((j + 1 for j, cid in enumerate(ids) if cid in relevant), None)
        if rank:
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
            mrr += 1 / rank
            ndcg += 1 / math.log2(rank + 1)
        if i % 20 == 0:
            print(f"  {i}/{len(gold)} done ({time.time() - t_all:.0f}s)")

    n = len(gold)
    print(f"\nmode={'thorough' if args.rerank else 'fast'}  (reranked responses: {reranked_count}/{n})")
    print(f"recall@1  = {r1 / n:.4f}")
    print(f"recall@5  = {r5 / n:.4f}")
    print(f"recall@10 = {r10 / n:.4f}")
    print(f"mrr@10    = {mrr / n:.4f}")
    print(f"ndcg@10   = {ndcg / n:.4f}")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
