"""Treatments -> ordered, numbered steps. Order comes only from ALLOWED_TRIPLES."""

from __future__ import annotations

from ..schemas import OPERATION_PHASE, OPERATION_STATE, DeletionStatus, Step, number_steps
from .policy import Treatment


def _step(op, t: Treatment, evidence_id: str) -> Step:
    return Step(step_no=1, phase=OPERATION_PHASE[op], system_id=t.system_id, resource_id=t.resource_id,
                operation=op, state=OPERATION_STATE[op], evidence_id=evidence_id)


def build_steps(treatments: list[Treatment]) -> tuple[DeletionStatus, list[Step]]:
    """INCOMPLETE when any treatment has a finding: findings cite evidence,
    remediation and verification describe future work. Otherwise COMPLETE:
    verification only, each citing the evidence that proves it."""
    violations = [t for t in treatments if t.finding is not None]
    steps: list[Step] = []
    if violations:
        for t in violations:
            steps.append(_step(t.finding, t, t.state.evidence_id))
            steps.append(_step(t.remediation, t, ""))
        steps.extend(_step(t.verification, t, "") for t in treatments)
        return DeletionStatus.INCOMPLETE, number_steps(steps)
    # A COMPLETE verification must cite evidence; a resource with no citable
    # state cannot be verified and is left out (the caller flags it).
    steps.extend(_step(t.verification, t, t.state.evidence_id) for t in treatments if t.state is not None)
    return DeletionStatus.COMPLETE, number_steps(steps)
