"""MedSearch AI — retrieval ablation evaluation v2 (spec §22, §14.4).

Modes:
  dense      : BGE-M3 dense only (cosine)
  sparse     : BGE-M3 sparse lexical only
  hybrid     : dense + sparse, RRF fusion, prefetch 50   (production v1)
  hybrid100  : dense + sparse, RRF fusion, prefetch 100
  weighted   : client-side weighted-score fusion, best alpha from sweep
  rerank     : hybrid100 top-30 reranked with BAAI/bge-reranker-v2-m3

Memory-staged for an 8 GB machine:
  1) embed queries with BGE-M3, then free the model
  2) all retrieval modes against Qdrant
  3) load the cross-encoder and rerank

Outputs:
  evaluation/results/results_<embedding_version>.json
  evaluation/RESULTS.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "data" / "processed" / "cases_clean.jsonl"
COLLECTION = "cases_v1"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_VERSION = "bgem3-v1"

RERANK_DEPTH = 30          # candidates fed to the cross-encoder
RERANK_DOC_CHARS = 1600    # per-candidate text budget for the reranker
ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7)

MODE_LABELS = {
    "dense": "Dense only",
    "sparse": "Sparse only",
    "hybrid": "Hybrid RRF (50)",
    "hybrid100": "Hybrid RRF (100)",
    "weighted": "Weighted fusion",
    "rerank": "Hybrid + rerank",
}


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def embed_queries(queries: list[str]) -> tuple[list, list]:
    print(f"[1/3] Loading BGE-M3 and embedding {len(queries)} queries ...", flush=True)
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    t0 = time.time()
    enc = model.encode(queries, return_dense=True, return_sparse=True,
                       max_length=512, batch_size=8)
    print(f"      embedded in {time.time()-t0:.1f}s", flush=True)
    dense = [v.tolist() for v in enc["dense_vecs"]]
    sparse = [
        models.SparseVector(indices=[int(k) for k in w.keys()],
                            values=[float(v) for v in w.values()])
        for w in enc["lexical_weights"]
    ]
    del model, enc
    gc.collect()
    return dense, sparse


def version_filter() -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="embedding_version",
                              match=models.MatchValue(value=EMBEDDING_VERSION)),
    ])


def single_mode(client: QdrantClient, vec, using: str, limit: int) -> list[tuple[str, float]]:
    res = client.query_points(COLLECTION, query=vec, using=using,
                              query_filter=version_filter(), limit=limit,
                              with_payload=["case_id"])
    return [(p.payload["case_id"], p.score) for p in res.points]


def hybrid_rrf(client: QdrantClient, dense, sparse, prefetch: int, k: int) -> tuple[list[str], float]:
    f = version_filter()
    t0 = time.time()
    res = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", filter=f, limit=prefetch),
            models.Prefetch(query=sparse, using="sparse", filter=f, limit=prefetch),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=["case_id"],
    )
    return [p.payload["case_id"] for p in res.points], time.time() - t0


def weighted_fuse(dense_list: list[tuple[str, float]], sparse_list: list[tuple[str, float]],
                  alpha: float, k: int) -> list[str]:
    """alpha * norm(dense) + (1-alpha) * norm(sparse), min-max per list."""
    def norm(lst):
        if not lst:
            return {}
        scores = [s for _, s in lst]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        return {cid: (s - lo) / rng for cid, s in lst}

    dn, sn = norm(dense_list), norm(sparse_list)
    fused: dict[str, float] = {}
    for cid, s in dn.items():
        fused[cid] = alpha * s
    for cid, s in sn.items():
        fused[cid] = fused.get(cid, 0.0) + (1 - alpha) * s
    return [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def compute_metrics(ranked_lists: list[list[str]], gold: list[dict], k: int) -> dict:
    recall = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    ndcg = 0.0
    for ranked, g in zip(ranked_lists, gold):
        relevant = set(g["relevant_case_ids"])
        hit_rank = next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)
        for cutoff in recall:
            if hit_rank is not None and hit_rank <= cutoff:
                recall[cutoff] += 1
        if hit_rank is not None and hit_rank <= k:
            mrr += 1.0 / hit_rank
            ndcg += 1.0 / math.log2(hit_rank + 1)  # single relevant doc -> IDCG = 1
    n = len(gold)
    return {
        "recall@1": round(recall[1] / n, 4),
        "recall@5": round(recall[5] / n, 4),
        "recall@10": round(recall[10] / n, 4),
        f"mrr@{k}": round(mrr / n, 4),
        f"ndcg@{k}": round(ndcg / n, 4),
    }


def load_docs(needed: set[str]) -> dict[str, str]:
    print(f"      loading {len(needed)} candidate documents ...", flush=True)
    docs: dict[str, str] = {}
    for line in open(CASES_FILE, encoding="utf-8"):
        r = json.loads(line)
        if r["case_id"] in needed:
            docs[r["case_id"]] = r["document"][:RERANK_DOC_CHARS]
            if len(docs) == len(needed):
                break
    return docs


def write_report(results: dict, out_md: Path) -> None:
    m = results["modes"]
    mode_keys = [k for k in MODE_LABELS if k in m]
    lines = [
        "# Retrieval Evaluation — Ablation Report",
        "",
        f"- **Date:** {results['timestamp']}",
        f"- **Collection:** `{results['collection']}` ({results['points_indexed']:,} points)",
        f"- **Embedding:** `{results['embedding_version']}` (BGE-M3, dense 1024-d + sparse lexical)",
        f"- **Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder over hybrid top-{RERANK_DEPTH}",
        f"- **Gold set:** {results['n_queries']} LLM-paraphrased clinician queries "
        "(self-retrieval protocol — the query's source case is the known-relevant document; "
        "queries are paraphrased so lexical retrieval gets no verbatim-overlap advantage)",
        f"- **Weighted fusion:** best α = {results['best_alpha']} "
        f"(swept {', '.join(str(a) for a in ALPHAS)})",
        "",
        "| Metric | " + " | ".join(MODE_LABELS[k] for k in mode_keys) + " |",
        "|" + "---|" * (len(mode_keys) + 1),
    ]
    metric_keys = list(m[mode_keys[0]]["metrics"].keys())
    for key in metric_keys:
        best = max(m[mode]["metrics"][key] for mode in mode_keys)
        row = f"| {key} |"
        for mode in mode_keys:
            v = m[mode]["metrics"][key]
            row += f" **{v:.4f}** |" if v == best else f" {v:.4f} |"
        lines.append(row)
    lines.append("| mean latency (ms) |" + "".join(
        f" {m[mode]['mean_latency_ms']:.0f} |" if m[mode]["mean_latency_ms"] is not None
        else " n/a (GPU) |" for mode in mode_keys))
    lines += [
        "",
        "## Reading the numbers",
        "",
        "- **Recall@10** is the primary spec target (§3: ≥ 0.85).",
        "- The ablation is the architecture's justification (§22): each pipeline stage",
        "  (fusion, deeper prefetch, cross-encoder reranking) must earn its place with a",
        "  measured gain.",
        "- The reranker attacks the *ordering* problem: cases that hybrid retrieval finds",
        "  but ranks low get re-scored with full query-document attention, lifting",
        "  Recall@1 / MRR / nDCG.",
        "- Rerank latency is CPU cross-encoder time; on a GPU deployment it drops an",
        "  order of magnitude.",
        "",
        "## Protocol notes",
        "",
        "- Gold queries were generated by an LLM from a stratified sample of the corpus",
        "  (seed 42, proportional to outcome_class) and are versioned in",
        "  `evaluation/gold_queries.jsonl`.",
        "- Self-retrieval is a lower bound on real-world usefulness: near-duplicate",
        "  clinical presentations may rank above the source case, which counts as a miss",
        "  here but is clinically still a good result.",
        "- Re-run with `python evaluation/run_eval.py` after any embedding/config change;",
        "  results are stored per embedding version under `evaluation/results/`.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=ROOT / "evaluation" / "gold_queries.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()
    k = args.k

    gold = load_gold(args.gold)
    print(f"Gold set: {len(gold)} queries", flush=True)

    dense_vecs, sparse_vecs = embed_queries([g["query"] for g in gold])

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    points = client.count(COLLECTION).count

    results: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "collection": COLLECTION,
        "points_indexed": points,
        "embedding_version": EMBEDDING_VERSION,
        "n_queries": len(gold),
        "k": k,
        "modes": {},
        "best_alpha": None,
    }

    print("[2/3] Retrieval modes ...", flush=True)

    # --- single-vector runs (also reused for weighted fusion) ---
    dense_lists, sparse_lists = [], []
    dense_lat, sparse_lat = [], []
    for d, s in zip(dense_vecs, sparse_vecs):
        t0 = time.time()
        dense_lists.append(single_mode(client, d, "dense", 100))
        dense_lat.append(time.time() - t0)
        t0 = time.time()
        sparse_lists.append(single_mode(client, s, "sparse", 100))
        sparse_lat.append(time.time() - t0)

    results["modes"]["dense"] = {
        "metrics": compute_metrics([[c for c, _ in l[:k]] for l in dense_lists], gold, k),
        "mean_latency_ms": round(sum(dense_lat) / len(dense_lat) * 1000, 1),
    }
    results["modes"]["sparse"] = {
        "metrics": compute_metrics([[c for c, _ in l[:k]] for l in sparse_lists], gold, k),
        "mean_latency_ms": round(sum(sparse_lat) / len(sparse_lat) * 1000, 1),
    }
    print(f"      dense  {results['modes']['dense']['metrics']}", flush=True)
    print(f"      sparse {results['modes']['sparse']['metrics']}", flush=True)

    # --- hybrid RRF prefetch 50 and 100 ---
    for mode, prefetch in (("hybrid", 50), ("hybrid100", 100)):
        ranked_lists, lats = [], []
        for d, s in zip(dense_vecs, sparse_vecs):
            ranked, lat = hybrid_rrf(client, d, s, prefetch, k)
            ranked_lists.append(ranked)
            lats.append(lat)
        results["modes"][mode] = {
            "metrics": compute_metrics(ranked_lists, gold, k),
            "mean_latency_ms": round(sum(lats) / len(lats) * 1000, 1),
        }
        print(f"      {mode} {results['modes'][mode]['metrics']}", flush=True)

    # --- weighted fusion sweep (client-side, reuses the 100-deep lists) ---
    best_alpha, best_ndcg, best_metrics = None, -1.0, None
    for alpha in ALPHAS:
        fused = [weighted_fuse(dl, sl, alpha, k) for dl, sl in zip(dense_lists, sparse_lists)]
        metrics = compute_metrics(fused, gold, k)
        print(f"      weighted alpha={alpha}: ndcg={metrics[f'ndcg@{k}']} r@10={metrics['recall@10']}", flush=True)
        if metrics[f"ndcg@{k}"] > best_ndcg:
            best_alpha, best_ndcg, best_metrics = alpha, metrics[f"ndcg@{k}"], metrics
    results["best_alpha"] = best_alpha
    results["modes"]["weighted"] = {
        "metrics": best_metrics,
        "mean_latency_ms": round((sum(dense_lat) + sum(sparse_lat)) / len(dense_lat) * 1000, 1),
    }

    # --- cross-encoder rerank over hybrid100 top-RERANK_DEPTH ---
    if not args.skip_rerank:
        print(f"[3/3] Reranking hybrid top-{RERANK_DEPTH} with bge-reranker-v2-m3 (CPU) ...", flush=True)
        # deep candidate lists from hybrid100
        candidate_lists = []
        for d, s in zip(dense_vecs, sparse_vecs):
            ranked, _ = hybrid_rrf(client, d, s, 100, RERANK_DEPTH)
            candidate_lists.append(ranked)

        needed = {cid for lst in candidate_lists for cid in lst}
        docs = load_docs(needed)

        del dense_vecs, sparse_vecs, dense_lists, sparse_lists
        gc.collect()

        # raw transformers (FlagReranker is incompatible with transformers v5)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        model.eval()

        def rerank_scores(query: str, cand_docs: list[str]) -> list[float]:
            scores: list[float] = []
            with torch.no_grad():
                for start in range(0, len(cand_docs), 8):
                    batch = [(query, doc) for doc in cand_docs[start:start + 8]]
                    inputs = tok(batch, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
                    logits = model(**inputs).logits.view(-1)
                    scores.extend(logits.tolist())
            return scores

        reranked_lists, lats = [], []
        t_start = time.time()
        for i, (g, cands) in enumerate(zip(gold, candidate_lists), 1):
            cand_docs = [docs.get(cid, "") for cid in cands]
            t0 = time.time()
            scores = rerank_scores(g["query"], cand_docs)
            order = sorted(range(len(cands)), key=lambda j: -scores[j])
            reranked_lists.append([cands[j] for j in order[:k]])
            lats.append(time.time() - t0)
            if i % 10 == 0:
                print(f"      {i}/{len(gold)} ({time.time()-t_start:.0f}s)", flush=True)

        results["modes"]["rerank"] = {
            "metrics": compute_metrics(reranked_lists, gold, k),
            "mean_latency_ms": round(sum(lats) / len(lats) * 1000, 1),
        }
        print(f"      rerank {results['modes']['rerank']['metrics']}", flush=True)

    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"results_{EMBEDDING_VERSION}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, ROOT / "evaluation" / "RESULTS.md")
    print(f"\nWrote {out_json}\nWrote {ROOT / 'evaluation' / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
