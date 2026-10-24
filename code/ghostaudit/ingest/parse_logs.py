"""Log files in seven line syntaxes, all reduced to (timestamp, key/value dict).

  ts [LEVEL] [service] [trace] event: k="v" ...         bracketed
  ts | LEVEL | service | trace | event | k="v" ...       pipe-delimited
  ts <pri>1 host service trace - - k="v" ...             syslog
  ts {"severity": ..., "event": ...}                     JSON payload
  ts requestId=.. function=.. message="k=\\"v\\" ..."     cloud runtime (nested message)
  ts level=.. service=.. trace_id=.. event=.. k=v        logfmt
  ts INFO service=.. event=record_lookup system=.. result=..   result ledger

A lookup is usually split across a trace: the *_started line carries the
identifier that was looked up, the *_completed line carries the resource and
its result. Lines are joined by trace id within a file. A trace with no line
naming a resource yields nothing.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime

from ..obs import get_logger
from ..schemas import ObservedState
from .manifest import EvidenceFile, parse_ts
from .observations import (
    LOOKUP_KEYS,
    ParseResult,
    observe_ledger,
    observe_record,
    present,
    system_for,
    truthy,
)

log = get_logger(__name__)

_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s+(.*)$")
_KV = re.compile(r'([A-Za-z_][\w.]*)=("(?:[^"\\]|\\.)*"|\S+)')
_BRACKET = re.compile(r"^\[(\w+)\] \[([\w-]+)\] \[([\w-]+)\] (\w+):\s*(.*)$")
_SYSLOG = re.compile(r"^<\d+>\d+ (\S+) (\S+) (\S+) - - (.*)$")


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1].replace('\\"', '"')
    return v


def _kv(text: str) -> dict[str, str]:
    return {k: _unquote(v) for k, v in _KV.findall(text)}


def tokenize(line: str) -> tuple[datetime, dict[str, str]] | None:
    m = _LINE.match(line.strip())
    if not m:
        return None
    ts, rest = m.groups()
    d: dict[str, str]
    if rest.startswith("{"):
        d = {k: ("" if v is None else str(v)) for k, v in json.loads(rest).items()}
    elif rest.startswith("["):
        b = _BRACKET.match(rest)
        if not b:
            return None
        d = {"level": b[1], "service": b[2], "trace": b[3], "event": b[4], **_kv(b[5])}
    elif " | " in rest:
        p = [x.strip() for x in rest.split(" | ", 5)]
        if len(p) < 5:
            return None
        d = {"level": p[0], "service": p[1], "trace": p[2], "event": p[3], **_kv(p[4])}
    elif rest.startswith("<"):
        s = _SYSLOG.match(rest)
        if not s:
            return None
        d = {"host": s[1], "service": s[2], "trace": s[3], **_kv(s[4])}
    else:
        d = _kv(rest)
        if "message" in d and "=" in d["message"]:
            d.update(_kv(d.pop("message")))
        if "function" in d:
            d["service"] = d.pop("function")
        if "requestId" in d:
            d["trace"] = d.pop("requestId")
        if not d:
            # Plain "INFO service=..." prefix: a bare level word precedes kv.
            return None
    if "trace_id" in d:
        d["trace"] = d.pop("trace_id")
    return parse_ts(ts), d


# event -> how the completed line states the resource's condition
def _state_hint(event: str, d: dict[str, str]) -> tuple[bool, dict] | None:
    """Returns (absent, record_overrides) or None if the line is not a
    resource observation."""
    if event in ("profile_lookup_completed", "audit_record_read_completed", "contact_query_completed",
                 "contact_lookup_completed"):
        rows = d.get("rows")
        status = d.get("http_status")
        absent = status == "404" or rows == "0"
        return absent, {}
    if event == "index_query_completed":
        return d.get("result", "").lower() == "miss" or d.get("total_hits") == "0", {}
    if event in ("export_metadata_read_completed", "export_download_check_completed"):
        return truthy(d.get("available")) is False, {}
    if event == "restore_policy_read_completed":
        return truthy(d.get("archive_present")) is False, {}
    if event == "restore_authorization_completed":
        return False, {"restore_allowed": d.get("decision")}
    if event == "restore_candidate_evaluated":
        # Restorable when the archive is eligible and no deletion block is set.
        allowed = truthy(d.get("eligible")) is True and truthy(d.get("deletion_block")) is not True
        return False, {"restore_allowed": allowed}
    if event in ("connector_heartbeat", "connector_config_loaded"):
        return False, {}
    if event == "retained_order_loaded":
        return False, {}
    if event in ("connector_config_read_completed", "analytics_profile_read_completed",
                 "analytics_profile_read", "support_ticket_read_completed", "message_read",
                 "message_pii_scan", "attachment_pii_scan", "analytics_identity_scan"):
        return False, {}
    if event == "deletion_marker_check":
        return truthy(d.get("marker_found")) is True, {}
    return None


_NOISE = {"level", "severity", "logger", "service", "trace", "event", "instance", "pod", "region", "host",
          "attempt", "duration_ms", "took_ms", "shard", "wait_ms", "seq", "latency_ms", "rows",
          "http_status", "result", "index", "total_hits", "integrity_check", "detector", "file",
          "available", "archive_present", "decision", "policy", "source", "cache", "pool"}


def parse(f: EvidenceFile) -> ParseResult:
    res = ParseResult()
    lines: list[tuple[datetime, dict[str, str], int]] = []
    for n, raw in enumerate(f.path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        t = tokenize(raw)
        if t is None:
            res.skipped += 1
            log.warning("log_line_unparsed", file_id=f.file_id, line=n)
            continue
        lines.append((t[0], t[1], n))

    trace_ids: dict[str, set[str]] = defaultdict(set)
    trace_has_resource: dict[str, bool] = defaultdict(bool)
    for _, d, _ in lines:
        tr = d.get("trace")
        if not tr:
            continue
        for k in LOOKUP_KEYS:
            if present(d.get(k)) and not _is_resource_id(d[k]):
                trace_ids[tr].add(d[k])
        if present(d.get("resource_id")):
            trace_has_resource[tr] = True

    for ts, d, n in lines:
        event = d.get("event", "")
        rid = d.get("resource_id")
        if event == "record_lookup" and present(rid):
            res.rows += 1
            o = observe_ledger(f, ts, system=d.get("system") or system_for(rid, "") or "", resource_id=rid,
                               record_type=d.get("record_type", ""), result=d.get("result", ""),
                               subject=d.get("subject_reference", ""), source_detail=f"log:L{n}:{event}")
            _add(res, o)
            continue
        if not present(rid):
            continue
        hint = _state_hint(event, d)
        if hint is None:
            if not event.endswith("_started"):
                # Fallback: an event we cannot read a state from is not turned
                # into an observation (e.g. thumbnail_failed says nothing
                # about whether the attachment still exists).
                log.warning("log_event_unmapped", file_id=f.file_id, line=n, log_event=event, resource_id=rid)
                res.notes.append(f"unmapped event {event} L{n}")
            continue
        res.rows += 1
        absent, overrides = hint
        record = {k: v for k, v in d.items() if k not in _NOISE}
        record.update(overrides)
        if event == "deletion_marker_check":
            # A deletion marker means the record was deleted, not that it holds PII.
            record = {"resource_id": rid}
        extra = sorted(trace_ids.get(d.get("trace", ""), set()))
        o = observe_record(f, ts, record, system_hint=d.get("service"), source_detail=f"log:L{n}:{event}",
                           absent=absent, extra_ids=extra)
        if o is not None and event == "attachment_pii_scan":
            o = o.model_copy(update={"state": ObservedState.PRESENT})
        _add(res, o)

    orphans = sorted(tr for tr in trace_ids if not trace_has_resource[tr])
    if orphans:
        res.notes.append(f"orphan_traces={len(orphans)}")
        log.info("log_orphan_traces", file_id=f.file_id, count=len(orphans))
    return res


def _is_resource_id(value: str) -> bool:
    return system_for(value) is not None and "@" not in value


def _add(res: ParseResult, o) -> None:
    if o is None:
        res.skipped += 1
    else:
        res.observations.append(o)
