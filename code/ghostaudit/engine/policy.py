"""deletion_rules.csv applied to resource states.

The rule file gives one action per category. Conditions are prose; the ones
that matter are encoded as states (a support message with shared context is
kept and redacted; an analytics record "still attributable" is IDENTIFIABLE).
Backups (RETAIN while restore can be blocked) and data flows (no rule; must
not be able to recreate data) are judged on restore/flow state.

Two conditions are read from the evidence rather than the state:
  - RETAIN holds only while the stated retention date (Observation.retain_until)
    is not before audit_at; past it the record is a remaining record (DELETE).
  - SUPPORT_ATTACHMENT is DELETE only when the attachment "belongs only to the
    deletion subject": one showing another subject's identifiers is not ours
    to delete, so it gets no step and the case is flagged SHARED_ATTACHMENT.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from .. import DATASET_DIR
from ..obs import get_logger
from ..schemas import DataCategory, Observation, ObservedState, Operation, RequiredAction, ResourceState

log = get_logger(__name__)

O = Operation
S = ObservedState

# (required action, observed state) -> (finding, remediation, verification);
# finding None means compliant and only the verification is emitted.
TREATMENT: dict[tuple[RequiredAction, ObservedState], tuple[O | None, O | None, O]] = {
    (RequiredAction.DELETE, S.ABSENT): (None, None, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.DELETE, S.PRESENT): (O.REPORT_REMAINING_RECORD, O.DELETE_RECORD, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.DELETE, S.IDENTIFIABLE): (O.REPORT_REMAINING_RECORD, O.DELETE_RECORD, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.DELETE, S.REDACTED): (O.REPORT_REMAINING_RECORD, O.DELETE_RECORD, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.DELETE, S.ANONYMIZED): (O.REPORT_REMAINING_RECORD, O.DELETE_RECORD, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.REDACT, S.REDACTED): (None, None, O.VERIFY_RECORD_REDACTED),
    (RequiredAction.REDACT, S.ABSENT): (None, None, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.REDACT, S.IDENTIFIABLE): (O.REPORT_EXPOSED_PII, O.REDACT_RECORD, O.VERIFY_RECORD_REDACTED),
    (RequiredAction.REDACT, S.PRESENT): (O.REPORT_EXPOSED_PII, O.REDACT_RECORD, O.VERIFY_RECORD_REDACTED),
    (RequiredAction.ANONYMIZE, S.ANONYMIZED): (None, None, O.VERIFY_RECORD_ANONYMIZED),
    (RequiredAction.ANONYMIZE, S.ABSENT): (None, None, O.VERIFY_RECORD_ABSENT),
    (RequiredAction.ANONYMIZE, S.IDENTIFIABLE): (O.REPORT_EXPOSED_PII, O.ANONYMIZE_RECORD,
                                                 O.VERIFY_RECORD_ANONYMIZED),
    (RequiredAction.ANONYMIZE, S.PRESENT): (O.REPORT_EXPOSED_PII, O.ANONYMIZE_RECORD, O.VERIFY_RECORD_ANONYMIZED),
    (RequiredAction.RETAIN, S.PRESENT): (None, None, O.VERIFY_RECORD_PRESENT),
}
RESTORE: dict[ObservedState, tuple[O | None, O | None, O]] = {
    S.BLOCKED: (None, None, O.VERIFY_RESTORE_BLOCKED),
    S.ACTIVE: (O.REPORT_RESTORE_RISK, O.BLOCK_RESTORE, O.VERIFY_RESTORE_BLOCKED),
}
FLOW: dict[ObservedState, tuple[O | None, O | None, O]] = {
    S.DISABLED: (None, None, O.VERIFY_DATA_FLOW_DISABLED),
    S.ACTIVE: (O.REPORT_ACTIVE_DATA_FLOW, O.DISABLE_DATA_FLOW, O.VERIFY_DATA_FLOW_DISABLED),
}
# End state to verify when a linked resource has no citable state.
PLANNED_END: dict[RequiredAction, O] = {
    RequiredAction.DELETE: O.VERIFY_RECORD_ABSENT,
    RequiredAction.REDACT: O.VERIFY_RECORD_REDACTED,
    RequiredAction.ANONYMIZE: O.VERIFY_RECORD_ANONYMIZED,
    RequiredAction.RETAIN: O.VERIFY_RECORD_PRESENT,
}


def load_rules(dataset_dir: Path = DATASET_DIR) -> dict[DataCategory, RequiredAction]:
    with open(dataset_dir / "deletion_rules.csv", newline="", encoding="utf-8") as f:
        return {DataCategory(r["data_category"]): RequiredAction(r["required_action"]) for r in csv.DictReader(f)}


class Scoped(NamedTuple):
    """A resource in a case's scope with what policy needs to judge it. Plain
    (system_id, resource_id, category, state) tuples are accepted too."""

    system_id: str
    resource_id: str
    category: DataCategory | None
    state: ResourceState | None
    retention_expired: bool = False
    shared: bool = False


RETENTION_EXPIRED = "retention expired"


def retention_expired(observations: Iterable[Observation], audit_at: datetime) -> bool:
    """True when the latest stated retention date is before audit_at. The
    latest statement wins (an extended hold supersedes an earlier date); no
    stated date means the rule's condition holds and the record is kept."""
    stated = sorted(((o.observed_at, o.file_id, o.retain_until) for o in observations
                     if o.retain_until is not None), key=lambda t: (t[0], t[1], t[2]))
    return bool(stated) and stated[-1][2] < audit_at


def attachment_subjects(o: Observation, owner_of: Callable[[str], str | None]) -> list[str]:
    """Distinct subjects an attachment observation shows: known users, plus
    each email no user holds (someone else's, by definition). Phones and
    hashes that resolve to nobody are not counted: they are too weak to
    assert a second person."""
    subjects = set()
    for ident in o.identifiers:
        uid = owner_of(ident)
        if uid:
            subjects.add(uid)
        elif "@" in ident:
            subjects.add("email:" + ident.strip().lower())
    return sorted(subjects)


def shared_attachment(observations: Iterable[Observation], owner_of: Callable[[str], str | None]) -> bool:
    """An attachment any single observation of which shows more than one
    subject does not belong only to the deletion subject (rule_06)."""
    return any(len(attachment_subjects(o, owner_of)) > 1 for o in observations
               if o.data_category is DataCategory.SUPPORT_ATTACHMENT)


@dataclass(frozen=True)
class Treatment:
    system_id: str
    resource_id: str
    category: DataCategory | None
    state: ResourceState | None
    finding: Operation | None
    remediation: Operation | None
    verification: Operation
    note: str = ""


def treat(system_id: str, resource_id: str, category: DataCategory | None, state: ResourceState | None,
          rules: dict[DataCategory, RequiredAction], retention_expired: bool = False) -> Treatment | None:
    note = ""
    if category is DataCategory.BACKUP_ARCHIVE:
        table, key = RESTORE, (state.state if state else None)
        planned = O.VERIFY_RESTORE_BLOCKED
    elif category is DataCategory.DATA_FLOW:
        table, key = FLOW, (state.state if state else None)
        planned = O.VERIFY_DATA_FLOW_DISABLED
    else:
        action = rules.get(category) if category else None
        if action is None:
            # Fallback: no rule, no step. Reporting a finding we cannot back
            # with policy would be a guess.
            log.warning("policy_no_rule", resource_id=resource_id, category=str(category))
            return None
        if action is RequiredAction.RETAIN and retention_expired:
            # The rule's condition ("within the supplied retention period",
            # "a supplied legal hold is active") no longer holds.
            action, note = RequiredAction.DELETE, RETENTION_EXPIRED
        table, key = TREATMENT, ((action, state.state) if state else None)
        planned = PLANNED_END[action]
    if state is None:
        return Treatment(system_id, resource_id, category, None, None, None, planned,
                         "; ".join(n for n in ("no citable state", note) if n))
    t = table.get(key)
    if t is None:
        log.warning("policy_state_unhandled", resource_id=resource_id, category=str(category), state=str(key))
        return None
    return Treatment(system_id, resource_id, category, state, *t, note)


def policy_diff(resources: list[Scoped | tuple[str, str, DataCategory | None, ResourceState | None]],
                rules: dict[DataCategory, RequiredAction]) -> tuple[list[Treatment], list[str]]:
    """Treatments for every resource policy covers, and the resources it does
    not (no rule for the category, or a state the rule table cannot judge).
    A shared attachment is in neither list: it is left alone on purpose and
    the caller flags it."""
    out, unmapped = [], []
    for r in resources:
        system_id, resource_id, category, state = r[:4]
        if getattr(r, "shared", False) and category is DataCategory.SUPPORT_ATTACHMENT:
            log.info("policy_shared_attachment_left", resource_id=resource_id)
            continue
        t = treat(system_id, resource_id, category, state, rules, getattr(r, "retention_expired", False))
        if t is None:
            unmapped.append(f"{system_id}/{resource_id} ({category}, {state.state.value if state else 'no state'})")
        else:
            out.append(t)
    return out, unmapped
