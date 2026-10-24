"""JSON and plain-text records.

JSON files are dispatched on their top-level keys, never on file name:
  service_exports           per-service lookup results (service_resource_export)
  snapshots                 full collection snapshots (customer_data_snapshot_bundle)
  collection + records      a single collection snapshot (analytics report, export jobs)
  origin + records          result ledger rows: system/resource/record_type/result
Plain text is the shift-handover format: "- ts | system | resource | type | result | subject".
"""

from __future__ import annotations

import json
import re

from ..obs import get_logger
from ..schemas import Coverage
from .manifest import EvidenceFile, parse_ts
from .observations import (
    ParseResult,
    flatten,
    identifiers_in,
    observe_ledger,
    observe_record,
    present,
    system_for,
)

log = get_logger(__name__)

_HANDOVER = re.compile(
    r"^-\s*(?P<ts>\S+)\s*\|\s*(?P<system>[\w-]+)\s*\|\s*(?P<rid>[\w-]+)\s*\|\s*(?P<type>\w+)\s*\|"
    r"\s*(?P<result>\w+)\s*\|\s*(?P<subject>.+?)\s*$"
)


def parse(f: EvidenceFile) -> ParseResult:
    if f.format == "txt":
        return _parse_handover(f)
    data = json.loads(f.path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("top-level JSON is not an object")
    if "service_exports" in data:
        return _parse_service_exports(f, data)
    if "snapshots" in data:
        return _parse_snapshots(f, data)
    if "records" in data and "collection" in data:
        return _parse_collection(f, data)
    if "records" in data and all(isinstance(r, dict) and "result" in r for r in data["records"]):
        return _parse_ledger(f, data["records"], "ledger")
    # Fallback: an unknown shape is skipped whole and logged; guessing field
    # meanings would produce confident wrong observations.
    log.warning("json_shape_unknown", file_id=f.file_id, keys=sorted(data))
    return ParseResult(notes=[f"unknown shape {sorted(data)}"])


def _parse_service_exports(f: EvidenceFile, data: dict) -> ParseResult:
    res = ParseResult()
    for system, sections in sorted(data["service_exports"].items()):
        for section, entries in sorted(sections.items()):
            for e in entries:
                res.rows += 1
                at = parse_ts(e.get("observed_at") or data["exported_at"])
                detail = f"service_exports.{system}.{section}"
                if section == "index_queries":
                    _search_query(f, res, at, e, detail)
                    continue
                record = dict(e)
                absent = False
                if "status_code" in e:
                    # profile lookups: 404 means the profile is gone; the
                    # nested profile carries whatever PII remains.
                    absent = int(e["status_code"]) == 404 or e.get("profile") is None
                    record = {"resource_id": e["resource_id"], **(e.get("profile") or {})}
                o = observe_record(f, at, record, system_hint=system, source_detail=detail, absent=absent)
                _add(res, o)
    return res


def _search_query(f: EvidenceFile, res: ParseResult, at, e: dict, detail: str) -> None:
    hits = e.get("hits") or {}
    docs = hits.get("documents") or []
    query = e.get("query")
    extra = [query] if present(query) else []
    if not docs:
        # A query that names a resource and returns nothing is evidence the
        # index entry is gone.
        _add(res, observe_record(f, at, {"resource_id": e["resource_id"]}, system_hint="search_service",
                                 source_detail=detail, absent=True, extra_ids=extra))
        return
    for d in docs:
        # The query links a hit only when the hit carries no identifier of its
        # own: a query for jonas.berg can return jonas.berglund's document.
        own, _ = identifiers_in(flatten(d))
        _add(res, observe_record(f, at, d, system_hint="search_service", source_detail=detail,
                                 extra_ids=[] if own else extra))


def _parse_snapshots(f: EvidenceFile, data: dict) -> ParseResult:
    res = ParseResult()
    for snap in data["snapshots"]:
        at = parse_ts(snap.get("captured_at") or data["exported_at"])
        store, coll = snap["data_store"], snap["collection"]
        detail = f"snapshot.{store}.{coll}"
        records = snap.get("records") or []
        if coll == "operational_record_history":
            sub = _parse_ledger(f, records, detail)
            res.rows += sub.rows
            res.observations.extend(sub.observations)
            res.skipped += sub.skipped
            continue
        rids = set()
        for r in records:
            res.rows += 1
            o = observe_record(f, at, r, system_hint=store, collection=coll, source_detail=detail)
            if o:
                rids.add(o.resource_id)
            _add(res, o)
        res.coverage.append(Coverage(file_id=f.file_id, captured_at=f.captured_at, observed_at=at,
                                     system_id=store, source_detail=detail, resource_ids=frozenset(rids)))
    return res


def _parse_collection(f: EvidenceFile, data: dict) -> ParseResult:
    res = ParseResult()
    at = parse_ts(data["captured_at"])
    coll = data["collection"]
    detail = f"collection.{coll}"
    rids = set()
    systems = set()
    for r in data["records"]:
        res.rows += 1
        o = observe_record(f, at, r, system_hint=data.get("source"), collection=coll, source_detail=detail)
        if o:
            rids.add(o.resource_id)
            systems.add(o.system_id)
        _add(res, o)
    for system in sorted(systems):
        res.coverage.append(Coverage(file_id=f.file_id, captured_at=f.captured_at, observed_at=at,
                                     system_id=system, source_detail=detail, resource_ids=frozenset(rids)))
    return res


def _parse_ledger(f: EvidenceFile, records: list[dict], detail: str) -> ParseResult:
    res = ParseResult()
    for r in records:
        res.rows += 1
        o = observe_ledger(
            f, parse_ts(r.get("recorded_at") or f.captured_at.isoformat()),
            system=r.get("system") or system_for(r["resource_id"], "") or "",
            resource_id=r["resource_id"], record_type=r.get("record_type", ""),
            result=r.get("result", ""), subject=str(r.get("subject_reference") or r.get("lookup_value") or ""),
            source_detail=detail,
        )
        _add(res, o)
    return res


def _parse_handover(f: EvidenceFile) -> ParseResult:
    res = ParseResult()
    for n, line in enumerate(f.path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.lstrip().startswith("-"):
            continue
        m = _HANDOVER.match(line.strip())
        if not m:
            res.skipped += 1
            log.warning("handover_line_unparsed", file_id=f.file_id, line=n)
            continue
        res.rows += 1
        _add(res, observe_ledger(
            f, parse_ts(m["ts"]), system=m["system"], resource_id=m["rid"], record_type=m["type"],
            result=m["result"], subject=m["subject"], source_detail=f"handover:L{n}",
        ))
    return res


def _add(res: ParseResult, o) -> None:
    if o is None:
        res.skipped += 1
    else:
        res.observations.append(o)
