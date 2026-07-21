"""Negation-aware scoring — NegEx-lite (spec §15.2).

Clinical notes are full of "pertinent negatives": "denies chest pain",
"no fever", "ruled out pulmonary embolism". Plain vector search treats those
mentions as positive evidence, which is clinically wrong. This layer detects
negated findings in BOTH the query and the candidate documents and adjusts
retrieval scores:

  1. Query asks for X, document mentions X ONLY under negation  -> penalty
     ("denies chest pain" must not satisfy a "chest pain" query)
  2. Query excludes X ("no fever"), document asserts X          -> penalty
  3. Query excludes X, document also negates X                  -> small boost
     (a shared pertinent negative is clinically meaningful)

Implementation: NegEx-style trigger lexicon + scope window. Pure regex —
no model, deterministic, <10 ms for a 50-candidate pool. Conservative by
design: it only acts when a term is present and its assertion status is
unambiguous; anything else is left untouched.
"""

from __future__ import annotations

import logging
import re

from app.core.config import settings
from app.schemas.api import CaseResult

logger = logging.getLogger(__name__)

# Longest-first so "no evidence of" wins over "no".
NEGATION_TRIGGERS = [
    "no evidence of",
    "no signs of",
    "no sign of",
    "no history of",
    "no previous history of",
    "no known history of",
    "negative for",
    "absence of",
    "free of",
    "denies any",
    "denied any",
    "denies",
    "denied",
    "ruled out",
    "rules out",
    "never had",
    "without",
    "no",
    "not",
]

_TRIGGER_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in NEGATION_TRIGGERS) + r")\b",
    re.IGNORECASE,
)

# A negation scope ends at punctuation, a contrast conjunction, or ~80 chars.
_TERMINATOR_RE = re.compile(
    r"[.;:,()\n]|\b(but|however|except|although|aside from|other than|apart from|besides)\b",
    re.IGNORECASE,
)

_SCOPE_CHARS = 80

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]+")

# Words that never form a useful clinical term on their own.
STOPWORDS = frozenset(
    "a an the and or of with in on at for to from by is are was were be been "
    "has have had his her their its he she they it this that these those who "
    "which what when where while during after before over under also very "
    "patient patients man woman male female boy girl year years old aged "
    "presented presenting presents presentation complaining complains "
    "reported reporting associated including onset recent sudden severe mild "
    "moderate acute chronic left right bilateral upper lower".split()
)

_TRIGGER_WORDS = frozenset(w for t in NEGATION_TRIGGERS for w in t.split())


# ---------------------------------------------------------------- primitives

def negated_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges under negation scope."""
    spans: list[tuple[int, int]] = []
    for m in _TRIGGER_RE.finditer(text):
        start = m.end()
        window = text[start : start + _SCOPE_CHARS]
        tm = _TERMINATOR_RE.search(window)
        end = start + (tm.start() if tm else len(window))
        if end > start:
            spans.append((start, end))
    return spans


def term_status(text: str, spans: list[tuple[int, int]], term: str) -> str:
    """'absent' | 'negated' (every occurrence is in a negation scope) | 'asserted'."""
    hits = [m.start() for m in re.finditer(re.escape(term), text, re.IGNORECASE)]
    if not hits:
        return "absent"
    if all(any(a <= h < b for a, b in spans) for h in hits):
        return "negated"
    return "asserted"


def parse_query(query: str) -> tuple[list[str], list[str]]:
    """Split the query into (positive_terms, negated_terms).

    Terms = content-word bigrams (e.g. "chest pain") + unigrams of length >= 5
    (e.g. "fever"), classified by whether they fall inside a negation scope.
    """
    spans = negated_spans(query)
    tokens = [(m.group().lower(), m.start()) for m in _WORD_RE.finditer(query)]

    def in_scope(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    def is_content(w: str) -> bool:
        return w not in STOPWORDS and w not in _TRIGGER_WORDS and len(w) > 2

    pos_terms: set[str] = set()
    neg_terms: set[str] = set()
    for i, (w, p) in enumerate(tokens):
        bucket = neg_terms if in_scope(p) else pos_terms
        if is_content(w) and len(w) >= 5:
            bucket.add(w)
        if i + 1 < len(tokens):
            w2, p2 = tokens[i + 1]
            if is_content(w) and is_content(w2):
                (neg_terms if in_scope(p2) else bucket).add(f"{w} {w2}")

    # a bigram subsumes its unigrams — drop unigrams contained in a bigram
    def dedupe(terms: set[str]) -> list[str]:
        bigrams = {t for t in terms if " " in t}
        return sorted(t for t in terms if " " in t or not any(t in b for b in bigrams))

    return dedupe(pos_terms), dedupe(neg_terms)


# ---------------------------------------------------------------- service

class NegationService:
    def adjust(
        self,
        query: str,
        results: list[CaseResult],
        docs: dict[str, str] | None,
    ) -> list[CaseResult]:
        """Re-score `results` for negation conflicts/concordances and re-sort.

        `docs` is the trimmed full-text cache (may be None while warming —
        falls back to the 300-char snippet, weaker but never blocking).
        """
        if not results:
            return results
        pos_terms, neg_terms = parse_query(query)
        if not pos_terms and not neg_terms:
            return results

        penalized = 0
        for r in results:
            text = (docs or {}).get(r.case_id) or r.snippet
            spans = negated_spans(text)
            conflicts = 0
            concordances = 0
            for t in pos_terms:
                if term_status(text, spans, t) == "negated":
                    conflicts += 1
            for t in neg_terms:
                st = term_status(text, spans, t)
                if st == "asserted":
                    conflicts += 1
                elif st == "negated":
                    concordances += 1
            if conflicts:
                r.score = round(r.score * (1 - settings.negation_penalty) ** conflicts, 4)
                if "negation_conflict" not in r.quality_flags:
                    r.quality_flags.append("negation_conflict")
                penalized += 1
            elif concordances:
                r.score = round(min(1.0, r.score * (1 + settings.negation_bonus)), 4)

        if penalized:
            logger.info(
                "negation: penalized %d/%d candidates (pos=%s neg=%s)",
                penalized, len(results), pos_terms[:4], neg_terms[:4],
            )
        results.sort(key=lambda r: -r.score)
        return results


negation_service = NegationService()
