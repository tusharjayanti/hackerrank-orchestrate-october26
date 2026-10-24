"""Format-independent record classification and the observation table.

Every parser reduces its format to plain record dicts and hands them to
`observe_record` / `observe_ledger`, so the rules for "which system, which
category, which state, which identifiers" live in exactly one place.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..identity import IdentityIndex, normalize_identifier
from ..obs import get_logger
from ..schemas import Coverage, DataCategory, Observation, ObservedState
from .manifest import EvidenceFile, Manifest

log = get_logger(__name__)

# Resource-id prefix is the most reliable system signal: log service names
# vary (profile_service, support_replica) but ids do not.
PREFIX_SYSTEM: tuple[tuple[str, str], ...] = (
    ("audit_record_", "security_service"),
    ("data_flow_", "integration_service"),
    ("legal_hold_", "legal_service"),
    ("email_delivery_", "email_delivery_service"),
    ("tombstone_", "identity_service"),
    ("analytics_", "analytics_platform"),
    ("telemetry_", "telemetry_service"),
    ("marketing_", "marketing_platform"),
    ("account_", "account_service"),
    ("search_", "search_service"),
    ("support_", "support_platform"),
    ("export_", "export_service"),
    ("backup_", "backup_service"),
    ("cache_", "cache_service"),
)

SYSTEM_CATEGORY: dict[str, DataCategory] = {
    "account_service": DataCategory.USER_PROFILE,
    "search_service": DataCategory.SEARCH_INDEX_ENTRY,
    "analytics_platform": DataCategory.ANALYTICS_PROFILE,
    "support_platform": DataCategory.SUPPORT_MESSAGE,
    "export_service": DataCategory.USER_EXPORT,
    "integration_service": DataCategory.DATA_FLOW,
    "backup_service": DataCategory.BACKUP_ARCHIVE,
    "security_service": DataCategory.SECURITY_AUDIT_LOG,
    "legal_service": DataCategory.LEGAL_HOLD_RECORD,
    "identity_service": DataCategory.ACCOUNT_TOMBSTONE,
    "marketing_platform": DataCategory.MARKETING_PROFILE,
    "cache_service": DataCategory.CACHE_ENTRY,
    "email_delivery_service": DataCategory.EMAIL_DELIVERY_LOG,
    "telemetry_service": DataCategory.PRODUCT_TELEMETRY,
}

COLLECTION_CATEGORY: dict[str, DataCategory] = {
    "ticket_attachments": DataCategory.SUPPORT_ATTACHMENT,
    # A contact-index row is a copy of the subject's contact identifiers, not
    # a conversation with shared context (sample case_10 deletes support_002).
    "legacy_contact_index": DataCategory.SECONDARY_IDENTIFIER,
    # A contact row carrying a marketing opt-in is a marketing profile,
    # whatever service serves it (sample case_12 deletes analytics_9914).
    "marketing_contact": DataCategory.MARKETING_PROFILE,
    # Orders kept for payment disputes and tax: rule_13 (RETAIN while the
    # hold is active), verified PRESENT like audit records.
    "retained_order_records": DataCategory.LEGAL_HOLD_RECORD,
}

# Fields that carry a direct identifier (prohibited after REDACT/ANONYMIZE).
DIRECT_PII_KEYS = frozenset({
    "email", "phone", "email_hash", "full_name", "name", "display_name",
    "requester_email", "requester_phone", "requester_name",
    "owner_email", "owner_phone", "customer_email", "uploaded_by", "matched_value",
})
# Stable internal references: link a record to a user but are permitted on a
# tombstone or a redacted support record.
REFERENCE_KEYS = frozenset({"customer_reference", "user_id", "subject_reference", "subject_selector"})
# Lookup keys in logs and ledgers: say who the record was looked up for, not
# what the record contains.
LOOKUP_KEYS = frozenset({
    "lookup_key", "match_key", "correlation_key", "subject_key", "scan_key", "subject_ref",
    "owner_key", "lookup_value", "query",
})
FREE_TEXT_KEYS = frozenset({"message_text", "message_excerpt"})

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+\d[\d\s().-]{6,}\d")
# Retention dates: a field of its own, or "retain until <date>" in free text.
RETENTION_KEYS = ("retention_until", "retained_until", "retain_until")
_RETAIN_TEXT = re.compile(r"retain(?:ed)?\s+until\s*:?\s*(\d{4}-\d\d-\d\d(?:[T ]\d\d:\d\d(?::\d\d)?Z?)?)",
                          re.IGNORECASE)
_NULLS = {None, "", "null", "none", "None", "NULL", "[REDACTED]"}

DEIDENTIFIED_STATE: dict[DataCategory, ObservedState] = {
    DataCategory.ANALYTICS_PROFILE: ObservedState.ANONYMIZED,
    DataCategory.PRODUCT_TELEMETRY: ObservedState.ANONYMIZED,
    DataCategory.SUPPORT_MESSAGE: ObservedState.REDACTED,
    DataCategory.EMAIL_DELIVERY_LOG: ObservedState.REDACTED,
    DataCategory.ACCOUNT_TOMBSTONE: ObservedState.REDACTED,
}

LEDGER_RESULT: dict[str, ObservedState] = {
    "FOUND": ObservedState.PRESENT,
    "NOT_FOUND": ObservedState.ABSENT,
    "IDENTIFIABLE": ObservedState.IDENTIFIABLE,
    "ANONYMIZED": ObservedState.ANONYMIZED,
    "REDACTED": ObservedState.REDACTED,
    "ACTIVE": ObservedState.ACTIVE,
    "DISABLED": ObservedState.DISABLED,
    "BLOCKED": ObservedState.BLOCKED,
}


def present(v: Any) -> bool:
    return v not in _NULLS and not (isinstance(v, str) and v.strip() in _NULLS)


def truthy(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "allow", "allowed", "enabled"}:
        return True
    if s in {"false", "0", "no", "deny", "denied", "disabled"}:
        return False
    return None


def system_for(resource_id: str, fallback: str | None = None) -> str | None:
    for prefix, system in PREFIX_SYSTEM:
        if resource_id.startswith(prefix):
            return system
    return fallback


def category_for(system_id: str, collection: str | None = None) -> DataCategory | None:
    if collection and collection in COLLECTION_CATEGORY:
        return COLLECTION_CATEGORY[collection]
    return SYSTEM_CATEGORY.get(system_id)


def text_identifiers(text: str) -> list[str]:
    return _EMAIL.findall(text) + [m.strip() for m in _PHONE.findall(text)]


def flatten(record: dict) -> dict:
    """One level of nesting is enough for every shape seen (search
    `document`, account `profile`); deeper values are ignored."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if not isinstance(v2, (dict, list)):
                    out.setdefault(k2, v2)
        elif not isinstance(v, list):
            out[k] = v
    return out


def identifiers_in(record: dict) -> tuple[list[str], bool]:
    """(identifiers that link the record to someone, has direct PII)."""
    ids: list[str] = []
    direct = False
    for k, v in record.items():
        if not present(v):
            continue
        if k in DIRECT_PII_KEYS:
            direct = True
            ids.extend(text_identifiers(str(v)) if k == "matched_value" else [str(v)])
        elif k in REFERENCE_KEYS or k in LOOKUP_KEYS:
            ids.append(str(v))
        elif k in FREE_TEXT_KEYS:
            found = text_identifiers(str(v))
            if found:
                direct = True
                ids.extend(found)
    # Names are PII but never link records (decoys share name fragments).
    ids = [i for i in ids if "@" in i or i.lower().startswith("user_") or _looks_linkable(i)]
    return sorted(set(ids)), direct


def _parse_date(value: str) -> datetime | None:
    v = value.strip().replace(" ", "T")
    if len(v) == 10:
        v += "T00:00:00"
    if not v.endswith("Z") and "+" not in v[10:]:
        v += "Z"
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def retain_until_in(record: dict) -> datetime | None:
    """The date until which a retained record may be kept, when the record
    states one. Several statements in one record: the earliest binds."""
    found = []
    for k in RETENTION_KEYS:
        if present(record.get(k)):
            d = _parse_date(str(record[k]))
            if d is None:
                log.warning("retention_date_unparsed", key=k)
            else:
                found.append(d)
    for k, v in sorted(record.items()):
        if isinstance(v, str) and k not in RETENTION_KEYS:
            found.extend(d for d in (_parse_date(m) for m in _RETAIN_TEXT.findall(v)) if d is not None)
    return min(found) if found else None


def _looks_linkable(value: str) -> bool:
    n = normalize_identifier(value)
    return n.isdigit() or (len(n) == 64 and all(c in "0123456789abcdef" for c in n))


def _flow_or_restore_flag(category: DataCategory | None, record: dict) -> bool | None:
    if category is DataCategory.DATA_FLOW:
        return truthy(record.get("enabled"))
    if category is DataCategory.BACKUP_ARCHIVE:
        return truthy(record.get("restore_allowed", record.get("decision")))
    return None


def field_defaulted(category: DataCategory | None, record: dict) -> bool:
    """A data flow or backup whose on/off field is missing or unreadable."""
    return category in (DataCategory.DATA_FLOW, DataCategory.BACKUP_ARCHIVE) and \
        _flow_or_restore_flag(category, record) is None


def state_for(category: DataCategory | None, record: dict, direct_pii: bool) -> ObservedState:
    if category in (DataCategory.DATA_FLOW, DataCategory.BACKUP_ARCHIVE):
        # Fail closed: an unknown flag is the exposed reading (flow running,
        # restore possible), never the compliant one (spec §6.5: when the
        # evidence cannot be resolved, the safer reading wins).
        flag = _flow_or_restore_flag(category, record)
        if flag is False:
            return ObservedState.DISABLED if category is DataCategory.DATA_FLOW else ObservedState.BLOCKED
        return ObservedState.ACTIVE
    if category in DEIDENTIFIED_STATE:
        linked = any(present(record.get(k)) for k in ("customer_reference", "user_id"))
        if direct_pii or (category in (DataCategory.ANALYTICS_PROFILE, DataCategory.PRODUCT_TELEMETRY) and linked):
            return ObservedState.IDENTIFIABLE
        return DEIDENTIFIED_STATE[category]
    return ObservedState.PRESENT


@dataclass
class ParseResult:
    observations: list[Observation] = field(default_factory=list)
    coverage: list[Coverage] = field(default_factory=list)
    rows: int = 0
    skipped: int = 0
    notes: list[str] = field(default_factory=list)


def observe_record(f: EvidenceFile, observed_at: datetime, record: dict, *, system_hint: str | None = None,
                   collection: str | None = None, source_detail: str, absent: bool = False,
                   extra_ids: list[str] | None = None) -> Observation | None:
    flat = flatten(record)
    rid = flat.get("resource_id")
    if not present(rid):
        return None
    rid = str(rid)
    system = system_for(rid, system_hint)
    if system is None:
        log.warning("observation_unmapped_system", file_id=f.file_id, resource_id=rid)
        return None
    category = category_for(system, collection)
    ids, direct = identifiers_in(flat)
    ids = sorted(set(ids) | set(extra_ids or []))
    state = ObservedState.ABSENT if absent else state_for(category, flat, direct)
    return Observation(
        file_id=f.file_id, captured_at=f.captured_at, observed_at=observed_at,
        system_id=system, resource_id=rid, data_category=category, state=state,
        identifiers=tuple(ids), source_detail=source_detail, vision=False,
        retain_until=retain_until_in(flat),
        field_defaulted=not absent and field_defaulted(category, flat),
    )


def observe_ledger(f: EvidenceFile, observed_at: datetime, *, system: str, resource_id: str,
                   record_type: str, result: str, subject: str, source_detail: str) -> Observation | None:
    """Rows that already state a result (handover, ledgers, record_lookup)."""
    state = LEDGER_RESULT.get(result.strip().upper())
    if state is None:
        log.warning("ledger_result_unmapped", file_id=f.file_id, resource_id=resource_id, result=result)
        return None
    try:
        category = DataCategory(record_type.strip().upper())
    except ValueError:
        category = category_for(system)
        log.warning("ledger_category_unmapped", file_id=f.file_id, record_type=record_type)
    # FOUND on a redact/anonymize category says nothing about PII; it stays
    # PRESENT and policy decides, rather than guessing IDENTIFIABLE here.
    ids =[subject] if present(subject) and subject.strip().lower() != "no direct identifier" else []
    return Observation(
        file_id=f.file_id, captured_at=f.captured_at, observed_at=observed_at,
        system_id=system_for(resource_id, system) or system, resource_id=resource_id,
        data_category=category, state=state, identifiers=tuple(ids),
        source_detail=source_detail, vision=False,
    )


@dataclass
class FileStats:
    file_id: str
    format: str
    status: str  # parsed | failed | deferred
    rows: int = 0
    observations: int = 0
    coverage: int = 0
    skipped: int = 0
    notes: list[str] = field(default_factory=list)


class ObservationTable:
    def __init__(self, observations: list[Observation], coverage: list[Coverage], stats: list[FileStats],
                 identity: IdentityIndex):
        # A total order on content, so neither manifest nor parser order can
        # leak into which observation is seen first.
        key = lambda o: (o.observed_at, o.file_id, o.system_id, o.resource_id, o.source_detail,  # noqa: E731
                         o.state.value, str(o.data_category), o.identifiers, str(o.retain_until))
        # vision_only rows (a screenshot id nothing else knows) are held out of
        # state and linking until the agent re-reads the file.
        self.held = sorted((o for o in observations if o.source_detail.endswith(":vision_only")), key=key)
        self.observations = sorted((o for o in observations if not o.source_detail.endswith(":vision_only")),
                                   key=key)
        self.coverage = sorted(coverage, key=lambda c: (c.observed_at, c.file_id, c.source_detail, c.system_id,
                                                        sorted(c.resource_ids)))
        self.stats = stats
        self.identity = identity
        self.by_resource: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        self.by_user: dict[str, list[Observation]] = defaultdict(list)
        self.conflicting_owner: dict[tuple[str, str], set[str]] = defaultdict(set)
        for o in self.observations:
            self.by_resource[(o.system_id, o.resource_id)].append(o)
            for uid in self.owners(o):
                self.by_user[uid].append(o)

    def owners(self, o: Observation) -> set[str]:
        return {u for u in (self.identity.owner_of(i) for i in o.identifiers) if u}

    def eligible(self, res: tuple[str, str], audit_at: datetime) -> list[Observation]:
        """Observations of one resource from evidence captured at or before audit_at."""
        return [o for o in self.by_resource.get(res, ()) if o.captured_at <= audit_at]

    def held_for_user(self, user_id: str, audit_at: datetime) -> list[Observation]:
        return [o for o in self.held if o.captured_at <= audit_at and user_id in self.owners(o)]

    def user_resources(self, user_id: str, audit_at: datetime) -> set[tuple[str, str]]:
        """Resources linked to the user by an identifier in evidence that was
        available at audit_at."""
        return {(o.system_id, o.resource_id) for o in self.by_user.get(user_id, ())
                if o.captured_at <= audit_at}

    def case_observations(self, user_id: str, audit_at: datetime) -> list[Observation]:
        """All eligible observations of the user's resources, including ones
        that carry no identifier themselves (a 404, an anonymized row)."""
        out = []
        for res in sorted(self.user_resources(user_id, audit_at)):
            out.extend(o for o in self.by_resource[res] if o.captured_at <= audit_at)
        return out


def available_parsers() -> dict:
    import importlib

    parsers = {}
    for fmt, module in (("json", "parse_records"), ("txt", "parse_records"),
                        ("sql", "parse_sql"), ("log", "parse_logs")):
        try:
            parsers[fmt] = importlib.import_module(f"{__package__}.{module}").parse
        except ModuleNotFoundError:
            continue
    return parsers


def build_table(manifest: Manifest, identity: IdentityIndex, formats: set[str] | None = None,
                activity=None, llm=None) -> ObservationTable:
    """Text formats first, then PDFs and screenshots, whose resource ids are
    checked against what the text evidence and system_activity know.
    formats limits which parsers run (for staged measurement); files of other
    formats are reported as deferred."""
    parsers = {k: v for k, v in available_parsers().items() if formats is None or k in formats}
    observations: list[Observation] = []
    coverage: list[Coverage] = []
    stats: list[FileStats] = []

    from ..llm import LLMUnavailable

    def run(f: EvidenceFile, parser) -> None:
        try:
            res = parser(f)
        except LLMUnavailable:
            raise  # not a parse failure: the caller records the file as deferred
        except Exception as e:  # noqa: BLE001 - one bad file must not sink the run
            log.error("parse_failed", file_id=f.file_id, path=f.rel_path, error=repr(e))
            stats.append(FileStats(f.file_id, f.format, "failed", notes=[repr(e)]))
            return
        observations.extend(res.observations)
        coverage.extend(res.coverage)
        stats.append(FileStats(f.file_id, f.format, "parsed", res.rows, len(res.observations),
                               len(res.coverage), res.skipped, res.notes))

    visual = []
    for f in manifest.files:
        if f.format in ("pdf", "png"):
            visual.append(f)
            continue
        parser = parsers.get(f.format)
        if parser is None:
            stats.append(FileStats(f.file_id, f.format, "deferred"))
            continue
        run(f, parser)

    from . import parse_vision

    known = {o.resource_id for o in observations}
    labels: dict[str, str] = {}
    if activity is not None:
        known |= {e.resource_id for e in activity.events}
        labels = {lf.label: lf.resource_id for lf in activity.links if lf.label != lf.resource_id}
    universe = parse_vision.Universe(known, labels)
    for f in visual:
        if formats is not None and f.format not in formats:
            stats.append(FileStats(f.file_id, f.format, "deferred"))
            continue
        if f.format == "png" and llm is None:
            stats.append(FileStats(f.file_id, f.format, "deferred", notes=["no model credentials"]))
            continue
        try:
            run(f, lambda ff: parse_vision.parse(ff, universe, llm))
        except LLMUnavailable as e:
            stats.append(FileStats(f.file_id, f.format, "deferred", notes=[str(e)]))
    stats.sort(key=lambda st: st.file_id)
    return ObservationTable(observations, coverage, stats, identity)
