"""Evidence manifest and the citation registry.

The manifest locates files; it says nothing about which case a file concerns.
The citation registry knows every id that may appear in evidence_id
(evidence files, activity rows, audit rows) and the time at which each became
available, so time eligibility is decided in one place.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import DATASET_DIR
from ..telemetry import get_logger

log = get_logger(__name__)

FORMATS = {".json": "json", ".txt": "txt", ".log": "log", ".sql": "sql", ".png": "png", ".pdf": "pdf"}


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


@dataclass(frozen=True)
class EvidenceFile:
    file_id: str
    rel_path: str
    path: Path
    origin: str
    captured_at: datetime
    format: str

    def eligible(self, audit_at: datetime) -> bool:
        # Inclusive on the full timestamp: a file captured exactly at audit_at
        # was available for the audit (sample case_11 cites one).
        return self.captured_at <= audit_at


class Manifest:
    def __init__(self, files: list[EvidenceFile]):
        # Sorted by file_id so manifest row order never affects results.
        self.files = sorted(files, key=lambda f: f.file_id)
        self.by_id = {f.file_id: f for f in self.files}

    @classmethod
    def load(cls, dataset_dir: Path = DATASET_DIR) -> Manifest:
        files = []
        with open(dataset_dir / "evidence_files.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = dataset_dir / row["file_path"]
                fmt = FORMATS.get(path.suffix.lower())
                if fmt is None:
                    # Fallback: unknown format has no parser; skip it loudly.
                    log.warning("manifest_unknown_format", file_id=row["file_id"], path=row["file_path"])
                    continue
                if not path.is_file():
                    log.warning("manifest_missing_file", file_id=row["file_id"], path=row["file_path"])
                    continue
                files.append(EvidenceFile(
                    file_id=row["file_id"],
                    rel_path=row["file_path"],
                    path=path,
                    origin=row["origin"],
                    captured_at=parse_ts(row["captured_at"]),
                    format=fmt,
                ))
        return cls(files)

    def eligible(self, audit_at: datetime) -> list[EvidenceFile]:
        return [f for f in self.files if f.eligible(audit_at)]


class CitationRegistry:
    """id -> time it became citable, across all three citable sources."""

    def __init__(self, times: dict[str, datetime]):
        self.times = times

    @classmethod
    def load(cls, dataset_dir: Path = DATASET_DIR) -> CitationRegistry:
        times: dict[str, datetime] = {}
        for name, id_col, time_col in (
            ("evidence_files.csv", "file_id", "captured_at"),
            ("system_activity.csv", "activity_id", "event_time"),
            ("user_audit_log.csv", "audit_id", "event_time"),
        ):
            with open(dataset_dir / name, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    times[row[id_col]] = parse_ts(row[time_col])
        return cls(times)

    def known(self, cited_id: str) -> bool:
        return cited_id in self.times

    def citable_at(self, cited_id: str, audit_at: datetime) -> bool:
        t = self.times.get(cited_id)
        return t is not None and t <= audit_at
