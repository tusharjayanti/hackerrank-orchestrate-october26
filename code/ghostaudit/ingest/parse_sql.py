"""SQLite-compatible snapshot exports.

Each file is executed into an in-memory database and read back through
sqlite_master, so tables are mapped by name and columns by column name, never
by position. Three table families occur:
  customer_data_<role>       generic rows: origin_system, collection_name, record_json
  <typed table>[_<role>]     one table per system (customer_profiles, search_documents, ...)
  snapshot_scope             which systems the snapshot covers in full
Primary, replica, regional and failover tables become separate observations
with the same logical system_id; the table name goes into source_detail so a
copy that survives only in a replica is still visible.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime

from ..schemas import Coverage
from ..telemetry import get_logger
from .manifest import EvidenceFile, parse_ts
from .observations import ParseResult, observe_record

log = get_logger(__name__)

_CAPTURED = re.compile(r"^--\s*Captured at:\s*(\S+)", re.MULTILINE)
_COMPLETED = re.compile(r"^--\s*snapshot_completed_at=(\S+)", re.MULTILINE)
_ROLE_SUFFIX = re.compile(r"_(primary|failover_replica|read_replica|reporting_replica|replica_[a-z_]+|replica)$")

TYPED_TABLES: dict[str, tuple[str, str | None]] = {
    "customer_profiles": ("account_service", None),
    "customer_search_documents": ("search_service", None),
    "search_documents": ("search_service", None),
    "customer_analytics_profiles": ("analytics_platform", None),
    "support_records": ("support_platform", None),
    "customer_exports": ("export_service", None),
    "connector_subscriptions": ("integration_service", None),
    "backup_restore_entries": ("backup_service", None),
    "security_audit_records": ("security_service", None),
}
GENERIC_COLUMNS = {"resource_id", "origin_system", "collection_name", "record_json"}


def parse(f: EvidenceFile) -> ParseResult:
    script = f.path.read_text(encoding="utf-8")
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(script)
        return _read(f, db, script)
    finally:
        db.close()


def _snapshot_time(f: EvidenceFile, script: str, scope: dict[str, datetime]) -> datetime:
    m = _CAPTURED.search(script) or _COMPLETED.search(script)
    if m:
        return parse_ts(m.group(1))
    if scope:
        return max(scope.values())
    return f.captured_at


def _read(f: EvidenceFile, db: sqlite3.Connection, script: str) -> ParseResult:
    res = ParseResult()
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    scope: dict[str, datetime] = {}
    if "snapshot_scope" in tables:
        scope = {s: parse_ts(t) for s, t in db.execute("SELECT origin_system, captured_at FROM snapshot_scope")}
    at = _snapshot_time(f, script, scope)

    generic_systems: set[str] = set()
    generic_rids: dict[str, dict[str, set[str]]] = {}
    typed_rids: dict[str, tuple[str, set[str]]] = {}

    for table in tables:
        if table == "snapshot_scope":
            continue
        cols = [r[1] for r in db.execute(f'PRAGMA table_info("{table}")')]
        # Sorted on full content, not only resource_id: a generic table can
        # hold the same resource twice and SQLite does not promise their order.
        rows = sorted((dict(zip(cols, r, strict=True)) for r in db.execute(f'SELECT * FROM "{table}"')),
                      key=lambda row: (str(row.get("resource_id")), json.dumps(row, sort_keys=True, default=str)))
        if GENERIC_COLUMNS <= set(cols):
            per_system = generic_rids.setdefault(table, {})
            for row in rows:
                res.rows += 1
                try:
                    record = json.loads(row["record_json"])
                except json.JSONDecodeError:
                    res.skipped += 1
                    log.warning("sql_record_json_invalid", file_id=f.file_id, table=table, rid=row["resource_id"])
                    continue
                record.setdefault("resource_id", row["resource_id"])
                o = observe_record(f, at, record, system_hint=row["origin_system"],
                                   collection=row["collection_name"], source_detail=f"sql.{table}")
                if o is None:
                    res.skipped += 1
                    continue
                res.observations.append(o)
                generic_systems.add(o.system_id)
                per_system.setdefault(o.system_id, set()).add(o.resource_id)
            continue

        base = _ROLE_SUFFIX.sub("", table)
        mapped = TYPED_TABLES.get(base)
        if mapped is None or "resource_id" not in cols:
            # Fallback: a table we cannot map is skipped and named, never
            # interpreted by guesswork.
            log.warning("sql_table_unmapped", file_id=f.file_id, table=table, columns=cols)
            res.notes.append(f"unmapped table {table}")
            continue
        system, collection = mapped
        rids = set()
        for row in rows:
            res.rows += 1
            o = observe_record(f, at, row, system_hint=system, collection=collection, source_detail=f"sql.{table}")
            if o is None:
                res.skipped += 1
                continue
            res.observations.append(o)
            rids.add(o.resource_id)
        typed_rids[table] = (system, rids)

    # Coverage: an inventory covers exactly the systems in snapshot_scope;
    # other snapshots cover each table's own system. For generic tables the
    # systems are those present anywhere in the file, so a row present in a
    # replica but missing from the primary reads as absent from the primary.
    for table, (system, rids) in sorted(typed_rids.items()):
        if scope and system not in scope:
            continue
        when = scope.get(system, at)
        res.coverage.append(Coverage(file_id=f.file_id, captured_at=f.captured_at, observed_at=when,
                                     system_id=system, source_detail=f"sql.{table}", resource_ids=frozenset(rids)))
    for table, per_system in sorted(generic_rids.items()):
        for system in sorted(generic_systems):
            res.coverage.append(Coverage(file_id=f.file_id, captured_at=f.captured_at, observed_at=at,
                                         system_id=system, source_detail=f"sql.{table}",
                                         resource_ids=frozenset(per_system.get(system, set()))))
    for system in sorted(set(scope) - {s for s, _ in typed_rids.values()}):
        log.warning("sql_scope_without_table", file_id=f.file_id, system=system)
    return res
