# MedSearch AI — GPU reranking on Kaggle (paste into a Kaggle notebook cell).
#
# Setup (same pattern as the embedding notebook):
#   1. Create a new Kaggle notebook, Settings -> Accelerator: GPU T4 x2 (or P100).
#   2. Add your uploaded dataset containing rerank_input.json
#      (Datasets -> New Dataset -> upload evaluation/rerank_input.json).
#   3. Paste this whole file into one cell and run.
#   4. Download /kaggle/working/rerank_output.json
#      -> save it as evaluation/rerank_output.json locally
#   5. Run locally: python evaluation/merge_rerank.py
#
# Uses plain transformers (no FlagEmbedding — its reranker is broken on
# transformers v5: XLMRobertaTokenizer lost `prepare_for_model`).

import glob
import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# locate the uploaded input regardless of dataset folder name
matches = glob.glob("/kaggle/input/**/rerank_input.json", recursive=True)
assert matches, "rerank_input.json not found — attach your dataset to the notebook"
INPUT = matches[0]
print("Input:", INPUT)

data = json.load(open(INPUT, encoding="utf-8"))
print(f"{len(data)} queries, {sum(len(q['candidates']) for q in data)} pairs")

MODEL = "BAAI/bge-reranker-v2-m3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
).to(device).eval()

BATCH = 32
MAX_LEN = 512   # short inputs score better (measured: nDCG 0.857 @512 vs 0.798 @1024)
DOC_CHARS = 1600
t0 = time.time()
out = []
with torch.no_grad():
    for i, q in enumerate(data, 1):
        cands = q["candidates"]
        scores: list[float] = []
        for b in range(0, len(cands), BATCH):
            batch = cands[b:b + BATCH]
            inputs = tokenizer(
                [q["query"]] * len(batch),
                [c["text"][:DOC_CHARS] for c in batch],
                padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
        order = sorted(range(len(cands)), key=lambda j: -scores[j])
        out.append({
            "query_id": q["query_id"],
            "relevant_case_ids": q["relevant_case_ids"],
            "reranked_case_ids": [cands[j]["case_id"] for j in order],
            "scores": [round(scores[j], 4) for j in order],
            "retrieval_scores": [cands[j].get("retrieval_score", 0.0) for j in order],
        })
        if i % 10 == 0:
            print(f"{i}/{len(data)} ({time.time()-t0:.0f}s)")

with open("/kaggle/working/rerank_output.json", "w", encoding="utf-8") as f:
    json.dump(out, f)

print(f"Done in {time.time()-t0:.0f}s -> /kaggle/working/rerank_output.json")
