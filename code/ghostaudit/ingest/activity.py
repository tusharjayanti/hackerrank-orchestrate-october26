"""system_activity.csv: a global event stream of claims.

reported_result is never used as resource state. Three uses only:
  - explicit identity links (LINK_RESOURCE_IDENTITY from data_catalog)
  - intake classifications (CLASSIFY_REQUEST), parked as an intent cross-check
  - deletion-worker events, as candidates for proximity linking
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import DATASET_DIR
from ..obs import get_logger
from .manifest import parse_ts

log = get_logger(__name__)

_LINKED = re.compile(r"linked (?P<rid>\S+) \((?P<label>[^)]*)\) to (?P<ident>\S+) and (?P<uid>user_\d+)")
_CONFIRMED = re.compile(
    r"confirmed (?P<rid>\S+) \((?P<label>[^)]*)\) belongs only to (?P<ident>\S+) and (?P<uid>user_\d+)")
_CLASSIFIED = re.compile(r"classified request for (?P<rid>\S+) as (?P<label>[\w-]+)")


@dataclass(frozen=True)
class ActivityEvent:
    activity_id: str
    event_time: datetime
    event_type: str
    source_system: str
    target_system: str
    resource_id: str
    reported_result: str
    details: str


@dataclass(frozen=True)
class LinkFact:
    activity_id: str
    event_time: datetime
    system_id: str
    resource_id: str
    label: str
    identifier: str
    user_id: str


@dataclass(frozen=True)
class IntakeClassification:
    activity_id: str
    event_time: datetime
    resource_id: str
    label: str


class Activity:
    def __init__(self, events: list[ActivityEvent]):
        self.events = sorted(events, key=lambda e: (e.event_time, e.activity_id))
        self.links = self._links()
        self.classifications = self._classifications()

    @classmethod
    def load(cls, dataset_dir: Path = DATASET_DIR) -> Activity:
        with open(dataset_dir / "system_activity.csv", newline="", encoding="utf-8") as f:
            events = [ActivityEvent(r["activity_id"], parse_ts(r["event_time"]), r["event_type"], r["source_system"],
                                    r["target_system"], r["resource_id"], r["reported_result"], r["details"])
                      for r in csv.DictReader(f)]
        return cls(events)

    def _links(self) -> list[LinkFact]:
        out = []
        for e in self.events:
            if e.event_type != "LINK_RESOURCE_IDENTITY":
                continue
            m = _LINKED.search(e.details) or _CONFIRMED.search(e.details)
            if not m or m["rid"] != e.resource_id:
                # Fallback: a link we cannot read exactly is not a link.
                log.warning("activity_link_unparsed", activity_id=e.activity_id)
                continue
            out.append(LinkFact(e.activity_id, e.event_time, e.target_system, e.resource_id, m["label"],
                                m["ident"], m["uid"]))
        return out

    def _classifications(self) -> list[IntakeClassification]:
        out = []
        for e in self.events:
            if e.event_type != "CLASSIFY_REQUEST":
                continue
            m = _CLASSIFIED.search(e.details)
            if not m:
                log.warning("activity_classification_unparsed", activity_id=e.activity_id)
                continue
            out.append(IntakeClassification(e.activity_id, e.event_time, m["rid"], m["label"]))
        return out

    def primary_profile_deletes(self) -> list[ActivityEvent]:
        return [e for e in self.events if e.event_type == "DELETE_RECORD" and e.source_system == "deletion_worker"
                and e.target_system == "account_service"]
