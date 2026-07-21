"""Create a stratified 5k dev subset from the cleaned corpus.

Stratifies on (outcome_class, age_band, sex) so the dev subset mirrors the
full corpus distribution. Deterministic (seeded) so the subset is reproducible.

Usage:
    python data/scripts/make_dev_subset.py [--size 5000]
Output:
    data/processed/cases_dev_5k.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CLEAN_FILE = Path(__file__).resolve().parents[1] / "processed" / "cases_clean.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "processed"

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5000)
    args = parser.parse_args()

    if not CLEAN_FILE.exists():
        raise SystemExit(f"Clean corpus not found: {CLEAN_FILE} — run clean_dataset.py first")

    # Pass 1: bucket line offsets by stratum (memory: offsets only, not documents)
    strata: dict[tuple, list[int]] = defaultdict(list)
    offsets = []
    with CLEAN_FILE.open("r", encoding="utf-8") as fh:
        pos = fh.tell()
        line = fh.readline()
        while line:
            rec = json.loads(line)
            key = (rec["outcome_class"], rec["age_band"], rec["sex"])
            strata[key].append(pos)
            offsets.append(pos)
            pos = fh.tell()
            line = fh.readline()

    total = len(offsets)
    rng = random.Random(SEED)

    # Proportional allocation per stratum (at least 1 where the stratum is big enough)
    selected: set[int] = set()
    for key, bucket in sorted(strata.items()):
        quota = round(args.size * len(bucket) / total)
        quota = min(quota, len(bucket))
        selected.update(rng.sample(bucket, quota))

    # Top up / trim to exact size deterministically
    remaining = [o for o in offsets if o not in selected]
    rng.shuffle(remaining)
    while len(selected) < args.size and remaining:
        selected.add(remaining.pop())
    selected_list = sorted(selected)[: args.size]
    selected = set(selected_list)

    out_path = OUT_DIR / f"cases_dev_{args.size // 1000}k.jsonl"
    written = 0
    with CLEAN_FILE.open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as fout:
        for off in selected_list:
            fh.seek(off)
            fout.write(fh.readline())
            written += 1

    print(f"Wrote {written} cases -> {out_path}")
    print(f"Strata used: {len(strata)} (seed={SEED})")


if __name__ == "__main__":
    main()
