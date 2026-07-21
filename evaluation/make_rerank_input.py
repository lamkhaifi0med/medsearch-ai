"""MedSearch AI — export rerank input for the Kaggle GPU notebook (v2).

Embeds the gold queries (BGE-M3, CPU, ~90s), retrieves WEIGHTED-FUSION
(alpha=0.4) top-100 candidates per query from Qdrant, attaches candidate
document texts + retrieval scores, and writes ONE self-contained file to
upload to Kaggle:

    evaluation/rerank_input.json
      [{"query_id", "query", "relevant_case_ids",
        "candidates": [{"case_id", "text", "retrieval_score"}, ...]}, ...]

Usage:
    python evaluation/make_rerank_input.py
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "data" / "processed" / "cases_clean.jsonl"
OUT_FILE = ROOT / "evaluation" / "rerank_input.json"
GOLD_FILE = ROOT / "evaluation" / "gold_queries.jsonl"

COLLECTION = "cases_v1"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_VERSION = "bgem3-v1"
RERANK_DEPTH = 100       # candidates handed to the GPU reranker (was 30)
RERANK_DOC_CHARS = 4000  # case text per candidate (was 1600)
ALPHA = 0.4              # weighted fusion: alpha*dense + (1-alpha)*sparse
PREFETCH = 150


def main() -> None:
    gold = [json.loads(line) for line in open(GOLD_FILE, encoding="utf-8")]
    print(f"Gold set: {len(gold)} queries", flush=True)

    print("Embedding queries with BGE-M3 (CPU) ...", flush=True)
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    t0 = time.time()
    enc = model.encode([g["query"] for g in gold], return_dense=True,
                       return_sparse=True, max_length=512, batch_size=8)
    print(f"  embedded in {time.time()-t0:.0f}s", flush=True)
    dense_vecs = [v.tolist() for v in enc["dense_vecs"]]
    sparse_vecs = [
        models.SparseVector(indices=[int(k) for k in w.keys()],
                            values=[float(v) for v in w.values()])
        for w in enc["lexical_weights"]
    ]
    del model, enc
    gc.collect()

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    f = models.Filter(must=[models.FieldCondition(
        key="embedding_version", match=models.MatchValue(value=EMBEDDING_VERSION))])

    print(f"Retrieving weighted-fusion (alpha={ALPHA}) top-{RERANK_DEPTH} per query ...", flush=True)
    candidate_lists: list[list[tuple[str, float]]] = []  # (case_id, fused score)
    for d, s in zip(dense_vecs, sparse_vecs):
        dres = client.query_points(COLLECTION, query=d, using="dense",
                                   query_filter=f, limit=PREFETCH,
                                   with_payload=["case_id"]).points
        sres = client.query_points(COLLECTION, query=s, using="sparse",
                                   query_filter=f, limit=PREFETCH,
                                   with_payload=["case_id"]).points

        def norm(pts):
            if not pts:
                return {}
            sc = [p.score for p in pts]
            lo, hi = min(sc), max(sc)
            rng = (hi - lo) or 1.0
            return {p.payload["case_id"]: (p.score - lo) / rng for p in pts}

        dn, sn = norm(dres), norm(sres)
        fused: dict[str, float] = {}
        for cid, sc in dn.items():
            fused[cid] = ALPHA * sc
        for cid, sc in sn.items():
            fused[cid] = fused.get(cid, 0.0) + (1 - ALPHA) * sc
        top = sorted(fused.items(), key=lambda x: -x[1])[:RERANK_DEPTH]
        candidate_lists.append(top)

    # ceiling: recall@RERANK_DEPTH of the candidate pool (rerank can't beat this)
    hits = sum(
        1 for g, cands in zip(gold, candidate_lists)
        if set(g["relevant_case_ids"]) & {cid for cid, _ in cands}
    )
    print(f"Candidate-pool ceiling: recall@{RERANK_DEPTH} = {hits/len(gold):.4f}", flush=True)

    needed = {cid for lst in candidate_lists for cid, _ in lst}
    print(f"Loading {len(needed)} candidate documents ...", flush=True)
    docs: dict[str, str] = {}
    for line in open(CASES_FILE, encoding="utf-8"):
        r = json.loads(line)
        if r["case_id"] in needed:
            docs[r["case_id"]] = r["document"][:RERANK_DOC_CHARS]
            if len(docs) == len(needed):
                break

    out = [
        {
            "query_id": g["query_id"],
            "query": g["query"],
            "relevant_case_ids": g["relevant_case_ids"],
            "candidates": [
                {"case_id": cid, "text": docs.get(cid, ""),
                 "retrieval_score": round(score, 4)}
                for cid, score in cands
            ],
        }
        for g, cands in zip(gold, candidate_lists)
    ]
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({OUT_FILE.stat().st_size/1e6:.1f} MB)", flush=True)
    print("\nNext: upload rerank_input.json to Kaggle as a dataset and run "
          "evaluation/kaggle_rerank_notebook.py in a GPU notebook.", flush=True)


if __name__ == "__main__":
    main()
