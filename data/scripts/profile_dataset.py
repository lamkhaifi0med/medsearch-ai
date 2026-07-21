"""Profile the augmented-clinical-notes dataset.

Streams the JSONL file (never loads 372 MB into memory) and produces a
profiling report: record counts, field presence, note length distributions,
summary-field coverage, duplicates, and data quality flags.

Usage:
    python data/scripts/profile_dataset.py
Output:
    data/reports/profile_report.md
    data/reports/profile_report.json
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "raw" / "augmented_notes_30K.jsonl"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Values in `summary` that mean "not documented"
EMPTY_VALUES = {"", "none", "n/a", "na", "unknown", "not applicable", "not specified", "-"}


def is_empty(value) -> bool:
    """True if a summary field value carries no information."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in EMPTY_VALUES
    if isinstance(value, (list, dict)):
        return all(is_empty(v) for v in (value.values() if isinstance(value, dict) else value)) or len(value) == 0
    return False


def flatten_summary_coverage(summary: dict, prefix: str = "") -> dict[str, bool]:
    """Map each top-level summary section -> whether it has any real content."""
    coverage = {}
    for key, value in summary.items():
        coverage[f"{prefix}{key}"] = not is_empty(value)
    return coverage


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"Dataset not found: {DATA_FILE}")

    n_records = 0
    parse_errors = 0
    field_presence: Counter = Counter()
    note_words: list[int] = []
    full_note_words: list[int] = []
    note_hashes: Counter = Counter()
    summary_coverage: Counter = Counter()
    summary_parse_errors = 0
    ages: Counter = Counter()
    sexes: Counter = Counter()
    non_ascii_notes = 0
    suspicious_short_notes = 0

    age_re = re.compile(r"(\d{1,3})")

    with DATA_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            n_records += 1

            for field in rec:
                field_presence[field] += 1

            note = rec.get("note", "") or ""
            wc = word_count(note)
            note_words.append(wc)
            if wc < 50:
                suspicious_short_notes += 1
            if any(ord(c) > 0x2FFF for c in note[:2000]):
                non_ascii_notes += 1
            note_hashes[hashlib.sha256(note.encode()).hexdigest()] += 1

            full_note = rec.get("full_note", "") or ""
            if full_note:
                full_note_words.append(word_count(full_note))

            summary_raw = rec.get("summary")
            summary = None
            if isinstance(summary_raw, dict):
                summary = summary_raw
            elif isinstance(summary_raw, str) and summary_raw.strip():
                try:
                    summary = json.loads(summary_raw)
                except json.JSONDecodeError:
                    summary_parse_errors += 1
            if isinstance(summary, dict):
                for section, filled in flatten_summary_coverage(summary).items():
                    if filled:
                        summary_coverage[section] += 1
                pi = summary.get("patient information", {})
                if isinstance(pi, dict):
                    sex = str(pi.get("sex", "")).strip().lower()
                    if sex and sex not in EMPTY_VALUES:
                        sexes[sex] += 1
                    age_raw = str(pi.get("age", ""))
                    m = age_re.search(age_raw)
                    if m:
                        age = int(m.group(1))
                        if 0 <= age <= 120:
                            band = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                            ages[band] += 1

    duplicates = sum(c - 1 for c in note_hashes.values() if c > 1)

    def dist(values: list[int]) -> dict:
        if not values:
            return {}
        values_sorted = sorted(values)
        return {
            "min": values_sorted[0],
            "p25": values_sorted[len(values_sorted) // 4],
            "median": int(statistics.median(values_sorted)),
            "p75": values_sorted[3 * len(values_sorted) // 4],
            "p95": values_sorted[int(0.95 * len(values_sorted))],
            "max": values_sorted[-1],
            "mean": round(statistics.mean(values_sorted), 1),
        }

    report = {
        "dataset": DATA_FILE.name,
        "records": n_records,
        "parse_errors": parse_errors,
        "field_presence": dict(field_presence),
        "note_word_distribution": dist(note_words),
        "full_note_word_distribution": dist(full_note_words),
        "duplicate_notes": duplicates,
        "short_notes_under_50_words": suspicious_short_notes,
        "notes_with_cjk_or_unusual_unicode": non_ascii_notes,
        "summary_parse_errors": summary_parse_errors,
        "summary_section_coverage_pct": {
            k: round(100 * v / n_records, 1) for k, v in sorted(summary_coverage.items(), key=lambda x: -x[1])
        },
        "sex_distribution": dict(sexes.most_common(10)),
        "age_band_distribution": {k: ages[k] for k in sorted(ages)},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "profile_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Dataset Profile — augmented_notes_30K.jsonl",
        "",
        f"- Records: **{n_records}** (parse errors: {parse_errors})",
        f"- Duplicate notes (by content hash): **{duplicates}**",
        f"- Notes under 50 words: **{suspicious_short_notes}**",
        f"- Notes with unusual unicode: **{non_ascii_notes}**",
        f"- Summary JSON parse errors: **{summary_parse_errors}**",
        "",
        "## Field presence",
        *[f"- `{k}`: {v}" for k, v in field_presence.items()],
        "",
        "## Note length (words)",
        f"```\n{json.dumps(report['note_word_distribution'], indent=2)}\n```",
        "",
        "## Full note length (words)",
        f"```\n{json.dumps(report['full_note_word_distribution'], indent=2)}\n```",
        "",
        "## Summary section coverage (% of records with content)",
        *[f"- {k}: {v}%" for k, v in report["summary_section_coverage_pct"].items()],
        "",
        "## Sex distribution",
        *[f"- {k}: {v}" for k, v in report["sex_distribution"].items()],
        "",
        "## Age bands",
        *[f"- {k}: {v}" for k, v in report["age_band_distribution"].items()],
        "",
    ]
    (REPORTS_DIR / "profile_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Profiled {n_records} records -> {REPORTS_DIR / 'profile_report.md'}")


if __name__ == "__main__":
    main()
