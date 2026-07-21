"""Load precomputed BGE-M3 embeddings from parquet into Qdrant.

Creates the `cases_v1` collection with named dense + sparse vectors and
payload indexes (spec §11), then bulk-upserts all points idempotently
(deterministic UUIDs derived from case_id — safe to re-run).

Prereq: Qdrant running on localhost:6333
    docker run -d --name medsearch-qdrant -p 6333:6333 \
        -v medsearch_qdrant_storage:/qdrant/storage qdrant/qdrant

Usage:
    python data/scripts/load_qdrant.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models

PROCESSED = Path(__file__).resolve().parents[1] / "processed"
PARQUET = PROCESSED / "embeddings.parquet"
MANIFEST = PROCESSED / "manifest.json"
CASES_FILE = PROCESSED / "cases_clean.jsonl"

COLLECTION = "cases_v1"
BATCH = 512
QDRANT_URL = "http://localhost:6333"

# Deterministic point IDs: same case_id always maps to the same UUID (idempotent upserts)
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def point_id(case_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, case_id))


def load_snippets() -> dict[str, str]:
    """First ~300 chars of each document, stored in payload for result display."""
    snippets = {}
    with CASES_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            snippets[rec["case_id"]] = rec["document"][:300]
    return snippets


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    print(f"Manifest: {manifest['model']} | {manifest['embedding_version']} | "
          f"{manifest['num_cases']} cases | dim {manifest['dense_dim']}")

    df = pd.read_parquet(PARQUET)
    assert len(df) == manifest["num_cases"], "parquet/manifest count mismatch"

    snippets = load_snippets()
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    # ---------------------------------------------------------------- collection
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        print(f"Collection {COLLECTION} exists ({info.points_count} points) — re-upserting idempotently.")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=manifest["dense_dim"],
                    distance=models.Distance.COSINE,
                    on_disk=True,  # memory-map vectors: keeps RAM low on 8 GB machines
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True),
                ),
            },
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128, on_disk=True),
        )
        # Payload indexes for every filterable field (spec §11.3/§11.6)
        for field, ftype in [
            ("sex", models.PayloadSchemaType.KEYWORD),
            ("age_band", models.PayloadSchemaType.KEYWORD),
            ("outcome_class", models.PayloadSchemaType.KEYWORD),
            ("embedding_version", models.PayloadSchemaType.KEYWORD),
            ("age", models.PayloadSchemaType.INTEGER),
        ]:
            client.create_payload_index(COLLECTION, field_name=field, field_schema=ftype)
        print(f"Created collection {COLLECTION} with payload indexes.")

    # ---------------------------------------------------------------- upsert
    total = len(df)
    for start in range(0, total, BATCH):
        chunk = df.iloc[start : start + BATCH]
        points = []
        for row in chunk.itertuples(index=False):
            age = None if pd.isna(row.age) else int(row.age)
            points.append(
                models.PointStruct(
                    id=point_id(row.case_id),
                    vector={
                        "dense": list(row.dense),
                        "sparse": models.SparseVector(
                            indices=list(row.sparse_indices),
                            values=list(row.sparse_values),
                        ),
                    },
                    payload={
                        "case_id": row.case_id,
                        "sex": row.sex,
                        "age": age,
                        "age_band": row.age_band,
                        "outcome_class": row.outcome_class,
                        "quality_flags": list(row.quality_flags),
                        "embedding_version": row.embedding_version,
                        "snippet": snippets.get(row.case_id, ""),
                    },
                )
            )
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        done = min(start + BATCH, total)
        print(f"\rUpserted {done}/{total}", end="", flush=True)

    print()
    info = client.get_collection(COLLECTION)
    print(f"Done. Collection {COLLECTION}: {info.points_count} points, status={info.status}")


if __name__ == "__main__":
    main()
