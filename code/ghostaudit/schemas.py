"""Shared data model.

ALLOWED_TRIPLES is the single source of truth for which (phase, operation,
state) combinations may appear in output and for the order they are emitted
in. Step ordering, validation and the contract suite all derive from it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Phase(StrEnum):
    FINDING = "FINDING"
    REMEDIATION = "REMEDIATION"
    VERIFICATION = "VERIFICATION"


class Operation(StrEnum):
    REPORT_REMAINING_RECORD = "REPORT_REMAINING_RECORD"
    REPORT_EXPOSED_PII = "REPORT_EXPOSED_PII"
    REPORT_ACTIVE_DATA_FLOW = "REPORT_ACTIVE_DATA_FLOW"
    REPORT_RESTORE_RISK = "REPORT_RESTORE_RISK"
    DISABLE_DATA_FLOW = "DISABLE_DATA_FLOW"
    BLOCK_RESTORE = "BLOCK_RESTORE"
    DELETE_RECORD = "DELETE_RECORD"
    REDACT_RECORD = "REDACT_RECORD"
    ANONYMIZE_RECORD = "ANONYMIZE_RECORD"
    VERIFY_RECORD_ABSENT = "VERIFY_RECORD_ABSENT"
    VERIFY_RECORD_PRESENT = "VERIFY_RECORD_PRESENT"
    VERIFY_RECORD_REDACTED = "VERIFY_RECORD_REDACTED"
    VERIFY_RECORD_ANONYMIZED = "VERIFY_RECORD_ANONYMIZED"
    VERIFY_DATA_FLOW_DISABLED = "VERIFY_DATA_FLOW_DISABLED"
    VERIFY_RESTORE_BLOCKED = "VERIFY_RESTORE_BLOCKED"


class StepState(StrEnum):
    """States allowed in an output step."""

    PRESENT = "PRESENT"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    BLOCKED = "BLOCKED"
    ABSENT = "ABSENT"
    REDACTED = "REDACTED"
    ANONYMIZED = "ANONYMIZED"


class ObservedState(StrEnum):
    """States a parser may assign to a resource seen in evidence.

    IDENTIFIABLE means the record exists and still carries identifiers that
    policy requires to be redacted or anonymized.
    """

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    REDACTED = "REDACTED"
    ANONYMIZED = "ANONYMIZED"
    IDENTIFIABLE = "IDENTIFIABLE"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    BLOCKED = "BLOCKED"


class DeletionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DataCategory(StrEnum):
    USER_PROFILE = "USER_PROFILE"
    SECONDARY_IDENTIFIER = "SECONDARY_IDENTIFIER"
    SEARCH_INDEX_ENTRY = "SEARCH_INDEX_ENTRY"
    CACHE_ENTRY = "CACHE_ENTRY"
    SUPPORT_MESSAGE = "SUPPORT_MESSAGE"
    SUPPORT_ATTACHMENT = "SUPPORT_ATTACHMENT"
    ANALYTICS_PROFILE = "ANALYTICS_PROFILE"
    PRODUCT_TELEMETRY = "PRODUCT_TELEMETRY"
    USER_EXPORT = "USER_EXPORT"
    MARKETING_PROFILE = "MARKETING_PROFILE"
    EMAIL_DELIVERY_LOG = "EMAIL_DELIVERY_LOG"
    SECURITY_AUDIT_LOG = "SECURITY_AUDIT_LOG"
    LEGAL_HOLD_RECORD = "LEGAL_HOLD_RECORD"
    BACKUP_ARCHIVE = "BACKUP_ARCHIVE"
    ACCOUNT_TOMBSTONE = "ACCOUNT_TOMBSTONE"
    # Not a deletion_rules.csv category: connectors are judged by whether they
    # are ACTIVE or DISABLED, not by a record-level action.
    DATA_FLOW = "DATA_FLOW"


class RequiredAction(StrEnum):
    DELETE = "DELETE"
    REDACT = "REDACT"
    ANONYMIZE = "ANONYMIZE"
    RETAIN = "RETAIN"


# Order matters: phases are contiguous and, within a phase, steps are sorted
# by the operation's position here, then system_id, then resource_id.
ALLOWED_TRIPLES: tuple[tuple[Phase, Operation, StepState], ...] = (
    (Phase.FINDING, Operation.REPORT_REMAINING_RECORD, StepState.PRESENT),
    (Phase.FINDING, Operation.REPORT_EXPOSED_PII, StepState.PRESENT),
    (Phase.FINDING, Operation.REPORT_ACTIVE_DATA_FLOW, StepState.ACTIVE),
    (Phase.FINDING, Operation.REPORT_RESTORE_RISK, StepState.ACTIVE),
    (Phase.REMEDIATION, Operation.DISABLE_DATA_FLOW, StepState.DISABLED),
    (Phase.REMEDIATION, Operation.BLOCK_RESTORE, StepState.BLOCKED),
    (Phase.REMEDIATION, Operation.DELETE_RECORD, StepState.ABSENT),
    (Phase.REMEDIATION, Operation.REDACT_RECORD, StepState.REDACTED),
    (Phase.REMEDIATION, Operation.ANONYMIZE_RECORD, StepState.ANONYMIZED),
    (Phase.VERIFICATION, Operation.VERIFY_RECORD_ABSENT, StepState.ABSENT),
    (Phase.VERIFICATION, Operation.VERIFY_RECORD_PRESENT, StepState.PRESENT),
    (Phase.VERIFICATION, Operation.VERIFY_RECORD_REDACTED, StepState.REDACTED),
    (Phase.VERIFICATION, Operation.VERIFY_RECORD_ANONYMIZED, StepState.ANONYMIZED),
    (Phase.VERIFICATION, Operation.VERIFY_DATA_FLOW_DISABLED, StepState.DISABLED),
    (Phase.VERIFICATION, Operation.VERIFY_RESTORE_BLOCKED, StepState.BLOCKED),
)

PHASE_ORDER: dict[str, int] = {p.value: i for i, p in enumerate(Phase)}
OPERATION_ORDER: dict[str, int] = {op.value: i for i, (_, op, _) in enumerate(ALLOWED_TRIPLES)}
OPERATION_PHASE: dict[str, str] = {op.value: ph.value for ph, op, _ in ALLOWED_TRIPLES}
OPERATION_STATE: dict[str, str] = {op.value: st.value for _, op, st in ALLOWED_TRIPLES}
TRIPLE_SET: frozenset[tuple[str, str, str]] = frozenset(
    (ph.value, op.value, st.value) for ph, op, st in ALLOWED_TRIPLES
)

STEP_FIELDS: tuple[str, ...] = (
    "step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id",
)
OUTPUT_HEADER: tuple[str, ...] = ("case_id", "deletion_status", "steps_json")

_CASE_NUM = re.compile(r"(\d+)$")


def case_sort_key(case_id: str) -> tuple[str, int, str]:
    """Natural order: case_33 before case_100."""
    m = _CASE_NUM.search(case_id)
    if not m:
        return (case_id, -1, case_id)
    return (case_id[: m.start()], int(m.group(1)), case_id)


def step_sort_key(phase: str, operation: str, system_id: str, resource_id: str) -> tuple:
    return (PHASE_ORDER[phase], OPERATION_ORDER[operation], system_id, resource_id)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Observation(_Frozen):
    """One fact about one resource, extracted from one evidence item.

    observed_at is the time the fact describes (a log line's timestamp, a SQL
    snapshot time); captured_at is the evidence file's capture time and decides
    whether the file may be used or cited for a given audit_at.
    """

    file_id: str
    captured_at: datetime
    observed_at: datetime
    system_id: str
    resource_id: str
    data_category: DataCategory | None
    state: ObservedState
    identifiers: tuple[str, ...]
    source_detail: str
    vision: bool = False


class Coverage(_Frozen):
    """A snapshot that lists a system's records completely, so a resource of
    that system missing from it was absent at observed_at."""

    file_id: str
    captured_at: datetime
    observed_at: datetime
    system_id: str
    source_detail: str
    resource_ids: frozenset[str]


class IdentitySet(_Frozen):
    """Current and historical identifiers for one user (raw values kept for
    display; *_norm sets are what matching uses)."""

    user_id: str
    name: str
    current_email: str
    current_phone: str
    current_email_hash: str
    emails: frozenset[str]
    phones: frozenset[str]
    email_hashes: frozenset[str]
    historical_emails: frozenset[str]

    def identifiers(self) -> frozenset[str]:
        """All normalised identifiers that link a record to this user."""
        return self.emails | self.phones | self.email_hashes | {self.user_id.lower()}


class IntentLabel(StrEnum):
    DELETION = "DELETION"
    ACCESS = "ACCESS"
    CORRECTION = "CORRECTION"
    EXPORT = "EXPORT"
    OPT_OUT = "OPT_OUT"
    OTHER = "OTHER"


class Intent(_Frozen):
    label: IntentLabel
    confidence: Literal["high", "low"]
    injection_suspected: bool = False

    @property
    def authorizes_deletion(self) -> bool:
        return self.label is IntentLabel.DELETION


class ResourceState(_Frozen):
    """A resource's state at audit_at after resolving all observations."""

    system_id: str
    resource_id: str
    data_category: DataCategory | None
    state: ObservedState
    evidence_id: str
    as_of: datetime
    conflicting: bool = False
    sources: tuple[str, ...] = ()


class Violation(_Frozen):
    """A resource whose state does not satisfy its required action."""

    resource: ResourceState
    required_action: RequiredAction
    finding: Operation
    remediation: Operation
    verification: Operation

    @model_validator(mode="after")
    def _phases(self) -> Violation:
        for op, phase in (
            (self.finding, Phase.FINDING),
            (self.remediation, Phase.REMEDIATION),
            (self.verification, Phase.VERIFICATION),
        ):
            if OPERATION_PHASE[op] != phase:
                raise ValueError(f"{op} is not a {phase} operation")
        return self


class Step(_Frozen):
    step_no: int
    phase: Phase
    system_id: str
    resource_id: str
    operation: Operation
    state: StepState
    evidence_id: str

    @field_validator("step_no")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step_no must be >= 1")
        return v

    @model_validator(mode="after")
    def _triple(self) -> Step:
        if (self.phase, self.operation, self.state) not in TRIPLE_SET:
            raise ValueError(f"disallowed triple {self.phase}/{self.operation}/{self.state}")
        return self

    def sort_key(self) -> tuple:
        return step_sort_key(self.phase, self.operation, self.system_id, self.resource_id)

    def as_json_dict(self) -> dict:
        return {k: (getattr(self, k).value if isinstance(getattr(self, k), StrEnum) else getattr(self, k))
                for k in STEP_FIELDS}


def number_steps(steps: list[Step]) -> list[Step]:
    """Sort by the canonical key and renumber from 1."""
    ordered = sorted(steps, key=Step.sort_key)
    return [s.model_copy(update={"step_no": i}) for i, s in enumerate(ordered, start=1)]


def serialize_steps(steps: list[Step]) -> str:
    return json.dumps([s.as_json_dict() for s in steps], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


class Decision(_Frozen):
    case_id: str
    status: DeletionStatus
    steps: tuple[Step, ...]
    flags: tuple[str, ...] = ()

    def to_row(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "deletion_status": self.status.value,
            "steps_json": serialize_steps(list(self.steps)),
        }
