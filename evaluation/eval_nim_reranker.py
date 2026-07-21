"""MedSearch AI — evaluate NVIDIA-hosted reranker (llama-3.2-nv-rerankqa-1b-v2).

Tests whether the "online big weapon" (NVIDIA NIM reranking API) matches the
Kaggle bge-reranker-v2-m3 quality. Reuses evaluation/rerank_input.json
(100 weighted-fusion candidates + texts per gold query). One API call per
query (all passages in a single request), with 429 backoff.

Writes evaluation/nim_rerank_output.json in the same format as the Kaggle
output so merge_rerank.py machinery (depth/beta sweep) applies.

Usage:
    python evaluation/eval_nim_reranker.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from merge_rerank import K, compute_metrics, depth_blend_rows  # noqa: E402

INPUT = ROOT / "evaluation" / "rerank_input.json"
OUTPUT = ROOT / "evaluation" / "nim_rerank_output.json"

MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
DEPTH = 50          # rescore top-50 retrieval candidates (winning recipe)
DOC_CHARS = 1600


def load_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                key = line.split("=", 1)[1].strip()
                break
    assert key, "NVIDIA_API_KEY not found in env or .env"
    return key


def rerank_one(session: requests.Session, query: str, texts: list[str]) -> list[float]:
    """Return one relevance logit per passage (in input order)."""
    payload = {
        "model": MODEL,
        "query": {"text": query},
        "passages": [{"text": t} for t in texts],
        "truncate": "END",
    }
    for attempt in range(1, 7):
        r = session.post(URL, json=payload, timeout=60)
        if r.status_code == 429:
            wait = 15 * attempt
            print(f"  429 — backing off {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        rankings = r.json()["rankings"]  # [{"index": i, "logit": s}, ...]
        scores = [0.0] * len(texts)
        for item in rankings:
            scores[item["index"]] = float(item["logit"])
        return scores
    raise RuntimeError("rate-limited after 6 attempts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N queries (smoke test)")
    args = ap.parse_args()

    data = json.load(open(INPUT, encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    print(f"{len(data)} queries; model={MODEL}; depth={DEPTH}", flush=True)

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {load_api_key()}",
        "Accept": "application/json",
    })

    # resume support
    rows: list[dict] = []
    done_ids: set[str] = set()
    if OUTPUT.exists():
        rows = json.load(open(OUTPUT, encoding="utf-8"))
        done_ids = {r["query_id"] for r in rows}
        print(f"Resuming: {len(done_ids)} already done", flush=True)

    latencies = []
    for i, q in enumerate(data, 1):
        if q["query_id"] in done_ids:
            continue
        cands = sorted(q["candidates"], key=lambda c: -c["retrieval_score"])
        head = cands[:DEPTH]
        t0 = time.time()
        scores = rerank_one(session, q["query"], [c["text"][:DOC_CHARS] for c in head])
        latencies.append(time.time() - t0)
        all_scores = scores + [float("-inf")] * (len(cands) - len(head))
        rows.append({
            "query_id": q["query_id"],
            "relevant_case_ids": q["relevant_case_ids"],
            "reranked_case_ids": [c["case_id"] for c in cands],  # retrieval order
            "scores": all_scores,
            "retrieval_scores": [c["retrieval_score"] for c in cands],
        })
        OUTPUT.write_text(json.dumps(rows), encoding="utf-8")  # incremental save
        if i % 10 == 0 or i == len(data):
            print(f"{i}/{len(data)} (mean {sum(latencies)/len(latencies):.2f}s/q)", flush=True)

    print(f"\nMean API latency: {sum(latencies)/max(len(latencies),1):.2f}s/query", flush=True)

    # depth/beta sweep (rows are retrieval-ordered; depth_blend_rows re-sorts head)
    best = (0.0, None, None)
    for depth in (20, 30, 50):
        for beta in (0.6, 0.7, 0.8, 0.9, 1.0):
            m = compute_metrics(depth_blend_rows(rows, depth, beta), K)
            print(f"depth={depth:<3} beta={beta:<4} {m}", flush=True)
            if m[f"ndcg@{K}"] > best[0]:
                best = (m[f"ndcg@{K}"], depth, beta)
    print(f"\nBest: depth={best[1]} beta={best[2]} ndcg={best[0]}")
    print("Kaggle bge-reranker benchmark: ndcg@10 = 0.857 (depth=50, beta=0.9)")
    print("No-rerank baseline:            ndcg@10 = 0.658 (weighted fusion)")


if __name__ == "__main__":
    main()
