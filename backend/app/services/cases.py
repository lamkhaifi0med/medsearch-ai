"""Case data service — full documents and structured summaries.

Loads a case_id -> file-offset index at startup, then serves any case by
seeking into cases_clean.jsonl (constant memory: nothing but offsets in RAM).
PostgreSQL replaces this in a later phase; the interface stays the same.
"""

from __future__ import annotations

import json
import logging
import time

from app.core.config import settings
from app.schemas.api import CaseDetail

logger = logging.getLogger(__name__)


class CaseStore:
    def __init__(self) -> None:
        self._offsets: dict[str, int] = {}

    def build_index(self) -> None:
        """Called once at startup."""
        t0 = time.time()
        with settings.cases_file.open("rb") as fh:
            pos = 0
            for line in fh:
                # cheap case_id extraction without full JSON parse
                marker = line.find(b'"case_id"')
                if marker != -1:
                    start = line.find(b'"', marker + 9 + 1) + 1
                    end = line.find(b'"', start)
                    self._offsets[line[start:end].decode()] = pos
                pos += len(line)
        logger.info("Case index: %d cases in %.1fs", len(self._offsets), time.time() - t0)

    def get(self, case_id: str) -> CaseDetail | None:
        off = self._offsets.get(case_id)
        if off is None:
            return None
        with settings.cases_file.open("rb") as fh:
            fh.seek(off)
            rec = json.loads(fh.readline())
        return CaseDetail(
            case_id=rec["case_id"],
            document=rec["document"],
            summary=rec.get("summary"),
            sex=rec.get("sex", "unknown"),
            age=rec.get("age"),
            age_band=rec.get("age_band", "unknown"),
            outcome_class=rec.get("outcome_class", "unknown"),
            quality_flags=rec.get("quality_flags", []),
        )

    def get_documents(self, case_ids: list[str]) -> dict[str, str]:
        out = {}
        for cid in case_ids:
            detail = self.get(cid)
            if detail:
                out[cid] = detail.document
        return out


case_store = CaseStore()
