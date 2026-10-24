"""deletion_rules.csv applied to resource states.

The rule file gives one action per category. Conditions are prose; the ones
that matter are encoded as states (a support message with shared context is
kept and redacted; an analytics record "still attributable" is IDENTIFIABLE).
Backups (RETAIN while restore can be blocked) and data flows (no rule; must
not be able to recreate data) are judged on restore/flow state.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .. import DATASET_DIR
from ..obs import get_logger
from ..schemas import DataCategory, ObservedState, Operation, RequiredAction, ResourceState

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
          rules: dict[DataCategory, RequiredAction]) -> Treatment | None:
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
        table, key = TREATMENT, ((action, state.state) if state else None)
        planned = PLANNED_END[action]
    if state is None:
        return Treatment(system_id, resource_id, category, None, None, None, planned, "no citable state")
    t = table.get(key)
    if t is None:
        log.warning("policy_state_unhandled", resource_id=resource_id, category=str(category), state=str(key))
        return None
    return Treatment(system_id, resource_id, category, state, *t)


def policy_diff(resources: list[tuple[str, str, DataCategory | None, ResourceState | None]],
                rules: dict[DataCategory, RequiredAction]) -> tuple[list[Treatment], list[str]]:
    """Treatments for every resource policy covers, and the resources it does
    not (no rule for the category, or a state the rule table cannot judge)."""
    out, unmapped = [], []
    for system_id, resource_id, category, state in resources:
        t = treat(system_id, resource_id, category, state, rules)
        if t is None:
            unmapped.append(f"{system_id}/{resource_id} ({category}, {state.state.value if state else 'no state'})")
        else:
            out.append(t)
    return out, unmapped
