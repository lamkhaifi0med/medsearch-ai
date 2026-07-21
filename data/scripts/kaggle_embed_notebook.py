# %% [markdown]
# # MedSearch AI — BGE-M3 Embedding Generation
#
# Embeds the cleaned clinical cases corpus with **BGE-M3** (dense 1024-d + sparse
# lexical weights) on Kaggle GPU.
#
# **Setup checklist (right panel):**
# - Input: add the `medsearch-cases-clean` dataset (containing `cases_clean.jsonl`)
# - Accelerator: **GPU T4 x2** or **P100**
# - Internet: **ON** (needed to download the model)
#
# **Output:** `embeddings.parquet` + `manifest.json` in `/kaggle/working/`
# → after the run finishes, download from the notebook's Output tab.

# %%
!pip install -q -U FlagEmbedding pyarrow

# %%
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------- config
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_VERSION = "bgem3-v1"      # stamp every vector with this (spec §12)
MAX_LENGTH = 4096                   # tokens per document (BGE-M3 supports 8192; 4096 is faster and covers p95 note length)
BATCH_SIZE = 16                     # safe for T4 16GB at max_length 4096; raise to 32 on P100/A100
OUT_DIR = Path("/kaggle/working")

# Locate the input file regardless of the exact dataset folder name
candidates = glob.glob("/kaggle/input/**/cases_clean.jsonl", recursive=True) \
           + glob.glob("/kaggle/input/**/cases_dev_5k.jsonl", recursive=True)
assert candidates, "cases_clean.jsonl not found — did you attach the dataset as Input?"
DATA_FILE = Path(candidates[0])
print(f"Input:  {DATA_FILE}")
print(f"GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — enable GPU!'}")
assert torch.cuda.is_available(), "Enable a GPU accelerator in Session options."

# %%
# ---------------------------------------------------------------- load corpus
cases = []
with DATA_FILE.open(encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        cases.append({
            "case_id": rec["case_id"],
            "text": rec["document"],
            "text_hash": hashlib.sha256(rec["document"].encode()).hexdigest(),
            # payload fields carried through so local indexing needs no join
            "sex": rec.get("sex", "unknown"),
            "age": rec.get("age"),
            "age_band": rec.get("age_band", "unknown"),
            "outcome_class": rec.get("outcome_class", "unknown"),
            "quality_flags": rec.get("quality_flags", []),
        })
print(f"Loaded {len(cases)} cases")

# %%
# ---------------------------------------------------------------- load model
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)  # fp16 halves memory, fine for retrieval
print("Model loaded.")

# %%
# ---------------------------------------------------------------- embed
texts = [c["text"] for c in cases]

t0 = time.time()
output = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    return_dense=True,
    return_sparse=True,          # lexical weights for the hybrid path
    return_colbert_vecs=False,   # multi-vector off: huge storage, not needed for v1
)
elapsed = time.time() - t0
print(f"Embedded {len(texts)} docs in {elapsed/60:.1f} min "
      f"({len(texts)/elapsed:.1f} docs/sec)")

dense = np.asarray(output["dense_vecs"], dtype=np.float32)      # (N, 1024), L2-normalized by the model
sparse = output["lexical_weights"]                              # list of {token_id: weight}
print(f"Dense shape: {dense.shape}")

# %%
# ---------------------------------------------------------------- save parquet
# Sparse vectors stored as parallel arrays (indices, values) — maps directly
# to Qdrant's SparseVector format at load time.
rows = []
for i, case in enumerate(cases):
    sp = sparse[i]
    rows.append({
        "case_id": case["case_id"],
        "text_hash": case["text_hash"],
        "dense": dense[i].tolist(),
        "sparse_indices": [int(k) for k in sp.keys()],
        "sparse_values": [float(v) for v in sp.values()],
        "sex": case["sex"],
        "age": case["age"],
        "age_band": case["age_band"],
        "outcome_class": case["outcome_class"],
        "quality_flags": case["quality_flags"],
        "embedding_version": EMBEDDING_VERSION,
    })

df = pd.DataFrame(rows)
out_file = OUT_DIR / "embeddings.parquet"
df.to_parquet(out_file, index=False)
print(f"Saved {out_file}  ({out_file.stat().st_size / 1e6:.0f} MB)")

# %%
# ---------------------------------------------------------------- manifest
manifest = {
    "embedding_version": EMBEDDING_VERSION,
    "model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "dense_dim": int(dense.shape[1]),
    "distance": "cosine (vectors pre-normalized -> dot)",
    "num_cases": len(cases),
    "source_file": DATA_FILE.name,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "encode_minutes": round(elapsed / 60, 1),
}
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

# %%
# ---------------------------------------------------------------- sanity check
# Nearest-neighbor smoke test: pick a case, find its top-5 by dense cosine.
q = 0
sims = dense @ dense[q]
top = np.argsort(-sims)[:6]
print(f"Query case: {cases[q]['case_id']}")
print(cases[q]["text"][:200], "\n---")
for rank, idx in enumerate(top[1:6], 1):
    print(f"#{rank}  {cases[idx]['case_id']}  sim={sims[idx]:.3f}")
    print("   ", cases[idx]["text"][:150].replace("\n", " "))
