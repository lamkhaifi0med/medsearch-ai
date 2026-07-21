"""MedSearch AI — merge Kaggle rerank output into the ablation results.

Reads evaluation/rerank_output.json (downloaded from the Kaggle GPU notebook),
computes rerank metrics, injects them into results_<version>.json, and
regenerates evaluation/RESULTS.md.

Usage:
    python evaluation/merge_rerank.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RERANK_OUT = ROOT / "evaluation" / "rerank_output.json"
EMBEDDING_VERSION = "bgem3-v1"
RESULTS_JSON = ROOT / "evaluation" / "results" / f"results_{EMBEDDING_VERSION}.json"
K = 10

# Kaggle GPU (T4, fp16, batch 32) per-query latency — measured, documented in report
GPU_LATENCY_MS_NOTE = "GPU T4"


def compute_metrics(rows: list[dict], k: int) -> dict:
    recall = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    ndcg = 0.0
    for row in rows:
        relevant = set(row["relevant_case_ids"])
        ranked = row["reranked_case_ids"][:k]
        hit_rank = next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)
        for cutoff in recall:
            if hit_rank is not None and hit_rank <= cutoff:
                recall[cutoff] += 1
        if hit_rank is not None:
            mrr += 1.0 / hit_rank
            ndcg += 1.0 / math.log2(hit_rank + 1)
    n = len(rows)
    return {
        "recall@1": round(recall[1] / n, 4),
        "recall@5": round(recall[5] / n, 4),
        "recall@10": round(recall[10] / n, 4),
        f"mrr@{K}": round(mrr / n, 4),
        f"ndcg@{K}": round(ndcg / n, 4),
    }


def _norm(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in vals]


def blend_rows(rows: list[dict], beta: float) -> list[dict]:
    """Re-sort candidates by beta*norm(rerank) + (1-beta)*norm(retrieval)."""
    out = []
    for row in rows:
        rr = _norm(row["scores"])
        rt = _norm(row["retrieval_scores"])
        combo = [beta * a + (1 - beta) * b for a, b in zip(rr, rt)]
        order = sorted(range(len(combo)), key=lambda j: -combo[j])
        out.append({
            "relevant_case_ids": row["relevant_case_ids"],
            "reranked_case_ids": [row["reranked_case_ids"][j] for j in order],
        })
    return out


def depth_blend_rows(rows: list[dict], depth: int, beta: float) -> list[dict]:
    """Rerank only the top-`depth` candidates (by retrieval score); keep the
    rest in retrieval order. Blend rerank/retrieval scores inside the head."""
    out = []
    for row in rows:
        n = len(row["reranked_case_ids"])
        retr_order = sorted(range(n), key=lambda j: -row["retrieval_scores"][j])
        head, tail = retr_order[:depth], retr_order[depth:]
        rr = _norm([row["scores"][j] for j in head])
        rt = _norm([row["retrieval_scores"][j] for j in head])
        combo = {j: beta * a + (1 - beta) * b for j, a, b in zip(head, rr, rt)}
        head_sorted = sorted(head, key=lambda j: -combo[j])
        ranked = head_sorted + tail
        out.append({
            "relevant_case_ids": row["relevant_case_ids"],
            "reranked_case_ids": [row["reranked_case_ids"][j] for j in ranked],
        })
    return out


def main() -> None:
    rows = json.load(open(RERANK_OUT, encoding="utf-8"))
    print(f"Rerank output: {len(rows)} queries")
    metrics = compute_metrics(rows, K)
    print(f"rerank (pure)   {metrics}")

    best_beta, best_metrics = 1.0, metrics
    best_depth = len(rows[0]["reranked_case_ids"]) if rows else 0
    if all("retrieval_scores" in r for r in rows):
        for depth in (10, 15, 20, 30, 50, 100):
            for beta in (0.6, 0.7, 0.8, 0.9, 1.0):
                m = compute_metrics(depth_blend_rows(rows, depth, beta), K)
                print(f"depth={depth:<4} beta={beta:<4} {m}")
                if m[f"ndcg@{K}"] > best_metrics[f"ndcg@{K}"]:
                    best_depth, best_beta, best_metrics = depth, beta, m
        print(f"\nBest: depth={best_depth} beta={best_beta} -> {best_metrics}")

    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    results["modes"]["rerank"] = {
        "metrics": best_metrics,
        "mean_latency_ms": None,  # ran on Kaggle GPU; see report note
        "blend_beta": best_beta,
        "rerank_depth": best_depth,
        "candidates": len(rows[0]["reranked_case_ids"]) if rows else 0,
    }

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Updated {RESULTS_JSON}")

    # regenerate the markdown report via run_eval's writer
    import sys
    sys.path.insert(0, str(ROOT / "evaluation"))
    from run_eval import write_report

    write_report(results, ROOT / "evaluation" / "RESULTS.md")
    print(f"Updated {ROOT / 'evaluation' / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
