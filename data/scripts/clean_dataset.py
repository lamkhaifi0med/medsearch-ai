"""Clean and prepare the augmented-clinical-notes dataset.

Pipeline (streaming, constant memory — safe on 8 GB machines):
  1. Parse each JSONL record.
  2. Deduplicate by content hash of the note text.
  3. Repair malformed summary JSON (control chars, unescaped quotes, trailing commas).
  4. Normalize demographics: sex -> {male, female, other/unknown}, age -> int + age band.
  5. Clean text: unicode NFKC, control-char stripping, whitespace healing,
     de-identification placeholder normalization.
  6. Derive outcome_class from the discharge section (improved/deceased/... /unknown).
  7. Emit one clean case per line to data/processed/cases_clean.jsonl
     and a cleaning report to data/reports/cleaning_report.md.

Usage:
    python data/scripts/clean_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from json_repair import repair_json

RAW_FILE = Path(__file__).resolve().parents[1] / "raw" / "augmented_notes_30K.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

EMPTY_VALUES = {"", "none", "n/a", "na", "unknown", "not applicable", "not specified", "-", "null"}

# ---------------------------------------------------------------- demographics

SEX_MAP = {
    "male": "male", "man": "male", "boy": "male", "gentleman": "male", "m": "male",
    "female": "female", "woman": "female", "girl": "female", "lady": "female", "f": "female",
    "trans man": "male", "trans woman": "female",
}

AGE_RE = re.compile(r"(\d{1,3})\s*(?:-|–|\s)?\s*(year|yr|y/o|yo|month|week|day)?", re.IGNORECASE)


def normalize_sex(raw) -> str:
    if not isinstance(raw, str):
        return "unknown"
    return SEX_MAP.get(raw.strip().lower(), "unknown")


def normalize_age(raw) -> int | None:
    """Return age in years (int) or None. Handles '16-year-old', '3 months', '45'."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in EMPTY_VALUES:
        return None
    m = AGE_RE.search(text)
    if not m:
        return None
    value = int(m.group(1))
    unit = (m.group(2) or "year").lower()
    if unit.startswith(("month",)):
        value = 0  # under 1 year
    elif unit.startswith(("week", "day")):
        value = 0
    return value if 0 <= value <= 120 else None


def age_band(age: int | None) -> str:
    if age is None:
        return "unknown"
    lo = (age // 10) * 10
    return f"{lo}-{lo + 9}"


# ---------------------------------------------------------------- text cleaning

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# de-id placeholder styles occasionally present in case reports
PLACEHOLDER_RES = [
    (re.compile(r"\[\*\*.*?\*\*\]"), "[REDACTED]"),
    (re.compile(r"\bXXXX+\b"), "[REDACTED]"),
]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    for pattern, repl in PLACEHOLDER_RES:
        text = pattern.sub(repl, text)
    # heal words broken across line-wraps: "hyper-\ntension" -> "hypertension"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- summary repair

def parse_summary(raw) -> tuple[dict | None, str]:
    """Return (summary_dict, status) where status in {ok, repaired, failed}."""
    if isinstance(raw, dict):
        return raw, "ok"
    if not isinstance(raw, str) or not raw.strip():
        return None, "failed"
    try:
        return json.loads(raw), "ok"
    except json.JSONDecodeError:
        pass
    try:
        repaired = repair_json(raw, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired, "repaired"
    except Exception:
        pass
    return None, "failed"


def prune_empty(value):
    """Recursively drop empty/'None'/'Unknown' values from the summary."""
    if isinstance(value, dict):
        pruned = {k: prune_empty(v) for k, v in value.items()}
        pruned = {k: v for k, v in pruned.items() if v not in (None, {}, [])}
        return pruned or None
    if isinstance(value, list):
        pruned = [prune_empty(v) for v in value]
        pruned = [v for v in pruned if v not in (None, {}, [])]
        return pruned or None
    if isinstance(value, str):
        return None if value.strip().lower() in EMPTY_VALUES else value.strip()
    return value


# ---------------------------------------------------------------- outcome class

DECEASED_RE = re.compile(r"\b(died|death|expired|deceased|passed away|fatal)\b", re.IGNORECASE)
IMPROVED_RE = re.compile(
    r"\b(improv|recover|resolv|discharged (home|in (a )?(stable|good))|stable condition|"
    r"symptom[- ]free|uneventful|good condition|well at follow)\w*", re.IGNORECASE)
DETERIORATED_RE = re.compile(r"\b(deteriorat|worsen|progress(ed|ion) of disease|relapse|recurren)\w*", re.IGNORECASE)


def derive_outcome_class(summary: dict | None, note_text: str) -> str:
    """Best-effort outcome classification from discharge section, else note tail."""
    texts = []
    if summary:
        discharge = summary.get("discharge")
        if isinstance(discharge, dict):
            texts.extend(str(v) for v in discharge.values() if v)
    # last ~15% of the note usually holds the outcome sentence
    tail = note_text[int(len(note_text) * 0.85):]
    texts.append(tail)
    joined = " ".join(texts)
    if DECEASED_RE.search(joined):
        return "deceased"
    if DETERIORATED_RE.search(joined):
        return "deteriorated"
    if IMPROVED_RE.search(joined):
        return "improved"
    return "unknown"


# ---------------------------------------------------------------- main

def main() -> None:
    if not RAW_FILE.exists():
        raise SystemExit(f"Raw dataset not found: {RAW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cases_clean.jsonl"

    stats: Counter = Counter()
    seen_hashes: set[str] = set()
    outcome_dist: Counter = Counter()
    sex_dist: Counter = Counter()
    summary_status_dist: Counter = Counter()

    with RAW_FILE.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["record_parse_error"] += 1
                continue

            note = rec.get("note") or ""
            # dedup on the ORIGINAL note text (duplicates share source note)
            note_hash = hashlib.sha256(note.encode()).hexdigest()
            if note_hash in seen_hashes:
                stats["duplicates_dropped"] += 1
                continue
            seen_hashes.add(note_hash)

            # full_note is the primary document (note is truncated at source)
            document = clean_text(rec.get("full_note") or note)
            if len(document.split()) < 60:
                stats["too_short_dropped"] += 1
                continue

            summary, status = parse_summary(rec.get("summary"))
            summary_status_dist[status] += 1
            if summary is not None:
                summary = prune_empty(summary) or {}

            pi = (summary or {}).get("patient information", {}) or {}
            sex = normalize_sex(pi.get("sex") if isinstance(pi, dict) else None)
            age = normalize_age(pi.get("age") if isinstance(pi, dict) else None)
            outcome = derive_outcome_class(summary, document)

            sex_dist[sex] += 1
            outcome_dist[outcome] += 1

            case = {
                "case_id": f"acn-{rec.get('idx', stats['read'])}",
                "source": "augmented-clinical-notes",
                "document": document,
                "reference_note": clean_text(note),
                "summary": summary,          # None if unrepairable
                "summary_status": status,    # ok | repaired | failed
                "sex": sex,
                "age": age,
                "age_band": age_band(age),
                "outcome_class": outcome,
                "quality_flags": (
                    (["no_structured_summary"] if summary is None else [])
                ),
            }
            fout.write(json.dumps(case, ensure_ascii=False) + "\n")
            stats["written"] += 1

    report = {
        "input": RAW_FILE.name,
        "output": str(out_path.name),
        "stats": dict(stats),
        "summary_status": dict(summary_status_dist),
        "sex_distribution": dict(sex_dist),
        "outcome_distribution": dict(outcome_dist),
    }
    (REPORTS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Cleaning Report — cases_clean.jsonl",
        "",
        f"- Records read: **{stats['read']}**",
        f"- Written: **{stats['written']}**",
        f"- Duplicates dropped: **{stats['duplicates_dropped']}**",
        f"- Too short dropped: **{stats['too_short_dropped']}**",
        f"- Record parse errors: **{stats['record_parse_error']}**",
        "",
        "## Summary JSON status",
        *[f"- {k}: {v}" for k, v in summary_status_dist.most_common()],
        "",
        "## Sex (normalized)",
        *[f"- {k}: {v}" for k, v in sex_dist.most_common()],
        "",
        "## Outcome class (derived)",
        *[f"- {k}: {v}" for k, v in outcome_dist.most_common()],
        "",
    ]
    (REPORTS_DIR / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {stats['written']} clean cases -> {out_path}")
    print(f"Report -> {REPORTS_DIR / 'cleaning_report.md'}")


if __name__ == "__main__":
    main()
