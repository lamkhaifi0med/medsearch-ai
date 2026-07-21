"""MedSearch AI — first hybrid search demo.

Embeds a free-text clinical query with BGE-M3 on CPU (single query ≈ 1 s),
runs dense + sparse hybrid search with RRF fusion inside Qdrant (spec §14.3),
applies optional metadata filters, and prints ranked similar cases.

Usage:
    python data/scripts/search.py "elderly man with chest pain and dyspnea"
    python data/scripts/search.py "diabetic foot ulcer" --sex male --outcome improved
    python data/scripts/search.py "seizures in a child" --age-min 0 --age-max 12 -k 5
"""

from __future__ import annotations

import argparse
import time

from qdrant_client import QdrantClient, models

COLLECTION = "cases_v1"
QDRANT_URL = "http://localhost:6333"

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading BGE-M3 (first run downloads ~2.3 GB)...")
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)  # CPU
    return _model


def build_filter(args) -> models.Filter | None:
    must = [
        models.FieldCondition(key="embedding_version", match=models.MatchValue(value="bgem3-v1")),
    ]
    if args.sex:
        must.append(models.FieldCondition(key="sex", match=models.MatchValue(value=args.sex)))
    if args.outcome:
        must.append(models.FieldCondition(key="outcome_class", match=models.MatchValue(value=args.outcome)))
    if args.age_min is not None or args.age_max is not None:
        must.append(models.FieldCondition(
            key="age",
            range=models.Range(gte=args.age_min, lte=args.age_max),
        ))
    return models.Filter(must=must)


def search(query: str, args) -> None:
    model = get_model()

    t0 = time.time()
    enc = model.encode([query], return_dense=True, return_sparse=True, max_length=512)
    dense = enc["dense_vecs"][0].tolist()
    sp = enc["lexical_weights"][0]
    sparse = models.SparseVector(
        indices=[int(k) for k in sp.keys()],
        values=[float(v) for v in sp.values()],
    )
    embed_ms = (time.time() - t0) * 1000

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    qfilter = build_filter(args)

    t1 = time.time()
    # Hybrid: dense + sparse prefetch, fused with RRF inside Qdrant (spec §14.3)
    result = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", filter=qfilter, limit=50),
            models.Prefetch(query=sparse, using="sparse", filter=qfilter, limit=50),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=args.k,
        with_payload=True,
    )
    search_ms = (time.time() - t1) * 1000

    print(f"\nQuery: {query!r}")
    active = [f"sex={args.sex}" if args.sex else None,
              f"outcome={args.outcome}" if args.outcome else None,
              f"age {args.age_min}-{args.age_max}" if args.age_min is not None or args.age_max is not None else None]
    active = [a for a in active if a]
    if active:
        print(f"Filters: {', '.join(active)}")
    print(f"(embed {embed_ms:.0f} ms | search {search_ms:.0f} ms)\n" + "=" * 80)

    for rank, point in enumerate(result.points, 1):
        p = point.payload
        demo = f"{p.get('sex','?')}, age {p.get('age','?')}"
        print(f"#{rank}  {p['case_id']}  score={point.score:.4f}  [{demo} | outcome: {p.get('outcome_class','?')}]")
        print(f"    {p.get('snippet','')[:220]}".replace("\n", " "))
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid clinical case search")
    parser.add_argument("query", help="free-text clinical query")
    parser.add_argument("-k", type=int, default=10, help="number of results")
    parser.add_argument("--sex", choices=["male", "female"])
    parser.add_argument("--outcome", choices=["improved", "deteriorated", "deceased", "unknown"])
    parser.add_argument("--age-min", type=int, default=None)
    parser.add_argument("--age-max", type=int, default=None)
    args = parser.parse_args()
    search(args.query, args)


if __name__ == "__main__":
    main()
