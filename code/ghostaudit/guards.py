"""Output guardrails (L3).

Contract checks operate on raw serialised rows so the same code validates our
output, the sample answers, and hand-built bad rows. The evidence guards below
them need the loaded world:

  citation guard   every cited id is an evidence file, captured <= audit_at,
                   that observes the resource it is cited for: an explicit
                   observation of that (system_id, resource_id), or, for an
                   ABSENT claim, a complete snapshot of that system in the
                   same file that omits it.
  complete check   a COMPLETE row is re-derived from its cited files alone
                   (state, then policy). If the cited evidence shows a
                   violation, the re-derived row *is* the downgrade:
                   INCOMPLETE, its findings citing the file that shows them,
                   and Flag.COMPLETE_DOWNGRADED. Downgrade never goes the
                   other way.
  policy contract  DELETE_RECORD never on records policy keeps (audit,
                   legal hold, backup, tombstone, support message, analytics
                   profile) unless their retention date has passed; no
                   cache_* / email_delivery_* resource in output at all.
  cross-row        one row per request (518 for the evaluation file) and a
                   warning when the status mix drifts from the sample mix.

`guard_decision` applies them per row: downgrade if the cited evidence
supports it; otherwise a row that fails any check is regenerated once by the
engine, and if that still fails it is replaced by its re-derivation from
cited evidence (or, with nothing citable left, the COMPLETE-[]-NEEDS_REVIEW
fallback), flagged CONTRACT_REPAIRED.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .engine.policy import RETENTION_EXPIRED, Scoped, Treatment, policy_diff, retention_expired, shared_attachment
from .engine.state import state_at
from .engine.steps import build_steps
from .ingest.manifest import CitationRegistry, parse_ts
from .ingest.observations import category_for
from .schemas import (
    OUTPUT_HEADER,
    PHASE_ORDER,
    STEP_FIELDS,
    TRIPLE_SET,
    DataCategory,
    Decision,
    DeletionStatus,
    Flag,
    FlagSet,
    Operation,
    Phase,
    Step,
    case_sort_key,
    serialize_steps,
    step_sort_key,
)
from .telemetry import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Issue:
    code: str
    case_id: str
    detail: str


Bad = Callable[[str, str], None]


def check_row(case_id: str, status: str, steps_json: str, audit_at: datetime | None,
              registry: CitationRegistry | None) -> list[Issue]:
    issues: list[Issue] = []

    def bad(code: str, detail: str) -> None:
        issues.append(Issue(code, case_id, detail))

    if status not in {s.value for s in DeletionStatus}:
        bad("status_invalid", repr(status))
    steps = _parse_steps(steps_json, bad)
    if steps is None:
        return issues
    by_phase = _check_order(steps, bad)
    if status == DeletionStatus.NOT_APPLICABLE:
        if steps:
            bad("na_with_steps", f"{len(steps)} steps")
        return issues
    if status == DeletionStatus.COMPLETE:
        _check_complete(steps, by_phase, bad)
    if status == DeletionStatus.INCOMPLETE:
        _check_incomplete(by_phase, bad)
    if registry is not None:
        _check_citations(steps, audit_at, registry, bad)
    return issues


def _parse_steps(steps_json: str, bad: Bad) -> list[dict] | None:
    """JSON array of exactly-7-field steps with allowed triples, or None."""
    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError as e:
        bad("json_invalid", str(e))
        return None
    if not isinstance(steps, list):
        bad("json_not_array", type(steps).__name__)
        return None
    ok = True
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or set(s) != set(STEP_FIELDS):
            bad("step_fields", f"step[{i}] keys={sorted(s) if isinstance(s, dict) else type(s).__name__}")
        elif not isinstance(s["step_no"], int) or isinstance(s["step_no"], bool):
            bad("step_no_type", f"step[{i}] step_no={s['step_no']!r}")
        elif not all(isinstance(s[k], str) for k in STEP_FIELDS if k != "step_no"):
            bad("step_field_type", f"step[{i}]")
        elif (s["phase"], s["operation"], s["state"]) not in TRIPLE_SET:
            bad("triple", f"step[{i}] {s['phase']}/{s['operation']}/{s['state']}")
        else:
            continue
        ok = False
    return steps if ok else None


def _check_order(steps: list[dict], bad: Bad) -> dict[str, list[dict]]:
    """Numbering, phase order, canonical sort, one step per resource per phase."""
    if [s["step_no"] for s in steps] != list(range(1, len(steps) + 1)):
        bad("numbering", f"step_no={[s['step_no'] for s in steps]}")
    phases = [PHASE_ORDER[s["phase"]] for s in steps]
    if phases != sorted(phases):
        bad("phase_order", f"phases={[s['phase'] for s in steps]}")
    keys = [step_sort_key(s["phase"], s["operation"], s["system_id"], s["resource_id"]) for s in steps]
    if keys != sorted(keys):
        bad("step_order", "steps not sorted by (phase, operation, system_id, resource_id)")
    by_phase: dict[str, list[dict]] = {p.value: [] for p in Phase}
    for s in steps:
        by_phase[s["phase"]].append(s)
    for phase, items in by_phase.items():
        dup = [r for r, n in Counter((s["system_id"], s["resource_id"]) for s in items).items() if n > 1]
        if dup:
            bad("resource_repeated", f"{phase} {dup}")
    return by_phase


def _check_complete(steps: list[dict], by_phase: dict[str, list[dict]], bad: Bad) -> None:
    if not steps:
        bad("complete_empty", "COMPLETE needs evidence-backed verification")
    if by_phase["FINDING"] or by_phase["REMEDIATION"]:
        bad("complete_non_verification", "COMPLETE has FINDING/REMEDIATION steps")
    for s in by_phase["VERIFICATION"]:
        if not s["evidence_id"]:
            bad("complete_missing_evidence", f"step {s['step_no']}")


def _check_incomplete(by_phase: dict[str, list[dict]], bad: Bad) -> None:
    """Findings cite evidence; future steps do not; every finding has a
    remediation and a verification ending in the remediated state."""
    findings = by_phase["FINDING"]
    if not findings:
        bad("incomplete_no_finding", "INCOMPLETE needs at least one finding")
    for s in findings:
        if not s["evidence_id"]:
            bad("finding_missing_evidence", f"step {s['step_no']}")
    for s in by_phase["REMEDIATION"] + by_phase["VERIFICATION"]:
        if s["evidence_id"]:
            bad("future_step_has_evidence", f"step {s['step_no']} cites {s['evidence_id']}")
    rem = {(s["system_id"], s["resource_id"]): s for s in by_phase["REMEDIATION"]}
    ver = {(s["system_id"], s["resource_id"]): s for s in by_phase["VERIFICATION"]}
    found = {(s["system_id"], s["resource_id"]) for s in findings}
    for res in sorted(found):
        if res not in rem:
            bad("finding_without_remediation", str(res))
        if res not in ver:
            bad("finding_without_verification", str(res))
    for res, s in sorted(rem.items()):
        if res not in found:
            bad("remediation_without_finding", str(res))
        elif res in ver and ver[res]["state"] != s["state"]:
            bad("verification_state_mismatch", f"{res} remediation={s['state']} verification={ver[res]['state']}")


def _check_citations(steps: list[dict], audit_at: datetime | None, registry: CitationRegistry, bad: Bad) -> None:
    for s in steps:
        eid = s["evidence_id"]
        if not eid:
            continue
        if not registry.known(eid):
            bad("citation_unknown", f"step {s['step_no']} cites {eid}")
        elif audit_at is not None and not registry.citable_at(eid, audit_at):
            bad("citation_after_audit", f"step {s['step_no']} cites {eid} "
                f"({registry.times[eid].isoformat()} > {audit_at.isoformat()})")


def load_requests(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def check_output(output: Path, requests: dict[str, dict], registry: CitationRegistry) -> list[Issue]:
    issues: list[Issue] = []
    with open(output, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if header is None or tuple(header) != OUTPUT_HEADER:
        issues.append(Issue("header", "-", f"header={header}"))
        return issues

    ids = []
    for n, row in enumerate(rows, start=2):
        if len(row) != len(OUTPUT_HEADER):
            issues.append(Issue("row_width", f"line{n}", f"{len(row)} columns"))
            continue
        case_id, status, steps_json = row
        ids.append(case_id)
        req = requests.get(case_id)
        audit_at = parse_ts(req["audit_at"]) if req else None
        issues.extend(check_row(case_id, status, steps_json, audit_at, registry))

    if len(rows) != len(requests):
        issues.append(Issue("row_count", "-", f"{len(rows)} rows, expected {len(requests)}"))
    dup = sorted(c for c, n in Counter(ids).items() if n > 1)
    if dup:
        issues.append(Issue("case_duplicate", "-", ",".join(dup)))
    missing = sorted(set(requests) - set(ids), key=case_sort_key)
    extra = sorted(set(ids) - set(requests), key=case_sort_key)
    if missing:
        issues.append(Issue("case_missing", "-", ",".join(missing)))
    if extra:
        issues.append(Issue("case_unexpected", "-", ",".join(extra)))
    if ids != sorted(ids, key=case_sort_key):
        issues.append(Issue("case_order", "-", "rows not sorted by case_id"))
    return issues


# ---------------------------------------------------------------- policy contract

# Records policy keeps: DELETE_RECORD on them is a violation unless the trace
# records that their retention date had passed at audit_at.
KEPT_PREFIXES = ("audit_record_", "legal_hold_", "backup_", "tombstone_")
KEPT_CATEGORIES = frozenset({DataCategory.SUPPORT_MESSAGE.value, DataCategory.ANALYTICS_PROFILE.value})
# Systems that occur only in system_activity: no evidence file observes them,
# so nothing about them can be cited and they never belong in output.
ACTIVITY_ONLY_PREFIXES = ("cache_", "email_delivery_")


def categories_from_trace(trace: dict | None) -> dict[str, tuple[str, str]]:
    """'system/resource' -> (data category, treatment note) as the engine judged it."""
    return {t["resource"]: (t.get("category") or "", t.get("note") or "")
            for t in (trace or {}).get("treatments", [])}


def policy_issues(case_id: str, steps: list[dict], categories: dict[str, tuple[str, str]] | None) -> list[Issue]:
    issues = []
    for s in steps:
        rid, key = s["resource_id"], f"{s['system_id']}/{s['resource_id']}"
        if rid.startswith(ACTIVITY_ONLY_PREFIXES):
            issues.append(Issue("activity_only_resource", case_id, f"step {s['step_no']} {key}"))
        if s["operation"] != "DELETE_RECORD":
            continue
        category, note = (categories or {}).get(key, ("", ""))
        if RETENTION_EXPIRED in note:
            continue
        if rid.startswith(KEPT_PREFIXES):
            issues.append(Issue("delete_on_kept_record", case_id, f"step {s['step_no']} {key}"))
        elif category in KEPT_CATEGORIES:
            issues.append(Issue("delete_on_kept_category", case_id, f"step {s['step_no']} {key} ({category})"))
    return issues


# ---------------------------------------------------------------- evidence guards

Res = tuple[str, str]


class _CitedView:
    """The observation table seen through one row's citations: observations
    from the cited files, plus those from files captured after audit_at (so
    a record seen only in a later export stays 'unknown', exactly as
    state_at treats it), and the cited files' snapshot coverage."""

    def __init__(self, table, cited: set[str], audit_at: datetime, resources: list[Res]):
        self.coverage = [c for c in table.coverage if c.file_id in cited]
        self.by_resource = {res: [o for o in table.by_resource.get(res, ())
                                  if o.file_id in cited or o.captured_at > audit_at] for res in resources}

    def cited_observations(self, res: Res, audit_at: datetime) -> list:
        return [o for o in self.by_resource.get(res, ()) if o.captured_at <= audit_at]


def observes(table, file_id: str, res: Res, absent_claim: bool) -> bool:
    """Does this file say anything about this resource? An explicit
    observation always counts; an omission counts only for an ABSENT claim
    and only from a complete snapshot of the resource's system."""
    if any(o.file_id == file_id for o in table.by_resource.get(res, ())):
        return True
    return absent_claim and any(c.file_id == file_id and c.system_id == res[0] and res[1] not in c.resource_ids
                                for c in table.coverage)


def table_with_agent(world, trace: dict | None):
    """The observation table, plus any record the agent verified for this
    case (resource id present in an eligible file with one of the subject's
    identifiers; agent/tools.verified_observation). Agent records exist only
    for the case that found them."""
    extra = ((trace or {}).get("agent") or {}).get("observations") or []
    if not extra:
        return world.table
    from .ingest.observations import ObservationTable
    from .schemas import Observation

    t = world.table
    return ObservationTable(t.observations + [Observation.model_validate(o) for o in extra], t.coverage, t.stats,
                            t.identity)


def citation_issues(case_id: str, steps: list[dict], audit_at: datetime, world, table=None) -> list[Issue]:
    table = table if table is not None else world.table
    issues = []
    for s in steps:
        eid = s["evidence_id"]
        if not eid:
            continue
        res = (s["system_id"], s["resource_id"])
        f = world.manifest.by_id.get(eid)
        if f is None:
            issues.append(Issue("citation_not_evidence_file", case_id, f"step {s['step_no']} cites {eid}"))
        elif f.captured_at > audit_at:
            issues.append(Issue("citation_after_audit", case_id, f"step {s['step_no']} cites {eid}"))
        elif not observes(table, eid, res, s["state"] == "ABSENT" and s["phase"] == Phase.VERIFICATION):
            issues.append(Issue("citation_does_not_observe", case_id,
                                f"step {s['step_no']} cites {eid}, which does not observe {res[0]}/{res[1]}"))
    return issues


def rederive(steps: list[dict], audit_at: datetime, world, table=None) -> tuple[DeletionStatus, list[Step]]:
    """The row the engine's own state and policy rules produce from the
    row's cited files alone. Resources the row cites nothing for keep their
    verification steps as future checks (they only exist in INCOMPLETE rows,
    where they make no evidence claim)."""
    resources = sorted({(s["system_id"], s["resource_id"]) for s in steps})
    cited = {s["evidence_id"] for s in steps if s["evidence_id"]}
    table = table if table is not None else world.table
    view = _CitedView(table, cited, audit_at, resources)
    scoped, kept = [], []
    for res in resources:
        mine = [s for s in steps if (s["system_id"], s["resource_id"]) == res]
        if not any(s["evidence_id"] for s in mine):
            kept.extend(Treatment(res[0], res[1], None, None, None, None, Operation(s["operation"]), "uncited")
                        for s in mine if s["phase"] == Phase.VERIFICATION)
            continue
        st = state_at(res, view, audit_at)
        # The category is a property of the resource, not a state claim: when
        # the cited file does not name it, take it as the engine does.
        full = state_at(res, table, audit_at)
        category = ((st.data_category if st else None) or (full.data_category if full else None)
                    or category_for(res[0]))
        obs = view.cited_observations(res, audit_at)
        shared = category is DataCategory.SUPPORT_ATTACHMENT and shared_attachment(obs, world.identity.owner_of)
        scoped.append(Scoped(res[0], res[1], category, st, retention_expired(obs, audit_at), shared))
    treatments, _ = policy_diff(scoped, world.rules)
    status, out = build_steps(treatments + kept)
    return status, out


def _canon(steps: list[dict]) -> list[str]:
    return [json.dumps(s, sort_keys=True) for s in steps]


def complete_check(case_id: str, steps: list[dict], audit_at: datetime, world, table=None
                   ) -> tuple[list[Issue], DeletionStatus, list[Step]]:
    """Re-derive a COMPLETE row from its cited evidence alone. Returns the
    issues, and the re-derived status and steps (the downgrade, when the
    cited evidence shows a violation)."""
    status, derived = rederive(steps, audit_at, world, table)
    got = json.loads(serialize_steps(derived))
    if status is DeletionStatus.COMPLETE and _canon(got) == _canon(steps):
        return [], status, derived
    derived_by_res = {(s["system_id"], s["resource_id"]): s for s in got}
    issues = []
    for s in steps:
        res = (s["system_id"], s["resource_id"])
        d = derived_by_res.get(res)
        same = d is not None and d["operation"] == s["operation"] and d["evidence_id"] == s["evidence_id"]
        if not same:
            if d is None:
                why = "nothing"
            else:
                findings = [g["operation"] for g in got if (g["system_id"], g["resource_id"]) == res
                            and g["phase"] == Phase.FINDING]
                why = ", ".join(findings) if findings else d["operation"]
            issues.append(Issue("complete_refuted" if status is DeletionStatus.INCOMPLETE else "complete_unsupported",
                                case_id, f"step {s['step_no']} {s['operation']} {res[0]}/{res[1]} "
                                f"[{s['evidence_id']}]: cited evidence gives {why}"))
    if not issues:
        issues.append(Issue("complete_refuted" if status is DeletionStatus.INCOMPLETE else "complete_unsupported",
                            case_id, "re-derived row differs"))
    return issues, status, derived


# ---------------------------------------------------------------- per-row guard

@dataclass
class GuardContext:
    world: object  # engine.decide.World (not imported: decide must not depend on guards)
    registry: CitationRegistry


def _with_flags(d: Decision, *hits: tuple[Flag, str], status: DeletionStatus | None = None,
                steps: list[Step] | tuple[Step, ...] | None = None) -> Decision:
    flags = FlagSet(d.flags)
    for flag, detail in hits:
        flags.add(flag, detail)
    return Decision(case_id=d.case_id, status=status or d.status,
                    steps=tuple(d.steps if steps is None else steps), flags=flags.hits)


def _is_agreed_fallback(d: Decision, codes: set[str]) -> bool:
    """COMPLETE with no steps and NEEDS_REVIEW: the row written when a subject
    has no evidence-backed step. It breaks only complete_empty, and no valid
    row exists for it without inventing a finding, so it stays as it is."""
    return d.status is DeletionStatus.COMPLETE and not d.steps and Flag.NEEDS_REVIEW in FlagSet(d.flags) \
        and codes == {"complete_empty"}


def row_issues(d: Decision, trace: dict | None, audit_at: datetime, ctx: GuardContext) -> list[Issue]:
    """Every per-row check: contract, citations, policy contract."""
    row = d.to_row()
    issues = check_row(d.case_id, row["deletion_status"], row["steps_json"], audit_at, ctx.registry)
    steps = json.loads(row["steps_json"])
    if d.status is not DeletionStatus.NOT_APPLICABLE:
        issues += citation_issues(d.case_id, steps, audit_at, ctx.world, table_with_agent(ctx.world, trace))
        issues += policy_issues(d.case_id, steps, categories_from_trace(trace))
    return issues


def _guard_once(d: Decision, trace: dict | None, audit_at: datetime, ctx: GuardContext, report: dict
                ) -> tuple[Decision, list[Issue]]:
    if d.status is DeletionStatus.COMPLETE and d.steps:
        steps = json.loads(d.to_row()["steps_json"])
        refuted, status, derived = complete_check(d.case_id, steps, audit_at, ctx.world,
                                                  table_with_agent(ctx.world, trace))
        if refuted:
            report["complete_check"] = [i.detail for i in refuted]
            if status is not DeletionStatus.INCOMPLETE:
                # Not downgradable: no cited file shows a violation, and an
                # INCOMPLETE row needs a finding that cites one.
                return d, refuted
            detail = "; ".join(i.detail for i in refuted)
            log.warning("complete_downgraded", case_id=d.case_id, detail=detail)
            d = _with_flags(d, (Flag.COMPLETE_DOWNGRADED, detail), status=DeletionStatus.INCOMPLETE, steps=derived)
            report["action"] = "downgraded"
    issues = row_issues(d, trace, audit_at, ctx)
    if _is_agreed_fallback(d, {i.code for i in issues}):
        report["accepted_fallback"] = True
        return d, []
    return d, issues


def guard_decision(decision: Decision, trace: dict, request: dict, ctx: GuardContext,
                   regenerate: Callable[[], tuple[Decision, dict]] | None = None) -> tuple[Decision, dict]:
    """Apply every per-row guard; returns the row to write and its trace
    (with a 'guards' section and the final flags)."""
    audit_at = parse_ts(request["audit_at"])
    report: dict = {"action": "none"}
    d, issues = _guard_once(decision, trace, audit_at, ctx, report)
    if issues:
        report["issues"] = [f"{i.code}: {i.detail}" for i in issues]
        codes = ",".join(sorted({i.code for i in issues}))
        log.warning("guard_row_failed", case_id=d.case_id, codes=codes)
        base, base_trace = d, trace
        if regenerate is not None:
            regen, regen_trace = regenerate()
            r2: dict = {"action": "none"}
            d2, issues2 = _guard_once(regen, regen_trace, audit_at, ctx, r2)
            if not issues2:
                d = _with_flags(d2, (Flag.CONTRACT_REPAIRED, f"regenerated by the engine after {codes}"))
                trace = regen_trace
                report.update(action="regenerated", regenerated=r2)
                return d, _finish(trace, d, report)
            base, base_trace = regen, regen_trace
            report["regenerated_issues"] = [f"{i.code}: {i.detail}" for i in issues2]
        d = _conservative(base, audit_at, ctx, codes)
        trace = base_trace
        report["action"] = "repaired"
    return d, _finish(trace, d, report)


def _conservative(d: Decision, audit_at: datetime, ctx: GuardContext, codes: str) -> Decision:
    """The most conservative valid row the evidence supports: the row
    re-derived from its own cited files (every supportable finding kept, every
    unsupported claim dropped). With nothing citable left, the
    COMPLETE-[]-NEEDS_REVIEW fallback. Never invents a finding."""
    steps = json.loads(d.to_row()["steps_json"]) if d.status is not DeletionStatus.NOT_APPLICABLE else []
    hits = [(Flag.CONTRACT_REPAIRED, f"re-derived from cited evidence after {codes}")]
    if d.status is DeletionStatus.NOT_APPLICABLE:
        return _with_flags(d, *hits, steps=[])
    status, derived = rederive(steps, audit_at, ctx.world)
    if status is DeletionStatus.COMPLETE and d.status is DeletionStatus.INCOMPLETE:
        hits.append((Flag.NEEDS_REVIEW, "repair found no supportable finding"))
    repaired = _with_flags(d, *hits, status=status, steps=derived)
    left = {i.code for i in row_issues(repaired, None, audit_at, ctx)}
    if left and not _is_agreed_fallback(repaired, left):
        repaired = _with_flags(d, *hits, (Flag.NEEDS_REVIEW, "no valid row from cited evidence"),
                               status=DeletionStatus.COMPLETE, steps=[])
    elif not derived:
        repaired = _with_flags(repaired, (Flag.NEEDS_REVIEW, "no evidence-backed step"))
    return repaired


def _finish(trace: dict, d: Decision, report: dict) -> dict:
    trace = dict(trace)
    trace["guards"] = report
    trace.update(FlagSet(d.flags).trace())
    return trace


# ---------------------------------------------------------------- cross-row

EVAL_ROWS = 518
MIX_DRIFT_WARN = 0.15  # total-variation distance from the sample status mix


def status_mix(statuses: list[str]) -> dict[str, float]:
    n = len(statuses) or 1
    c = Counter(statuses)
    return {s.value: c[s.value] / n for s in DeletionStatus}


def sample_mix(dataset_dir: Path) -> dict[str, float]:
    with open(dataset_dir / "sample_output.csv", newline="", encoding="utf-8") as f:
        return status_mix([r["deletion_status"] for r in csv.DictReader(f)])


def cross_row_check(decisions: list[Decision], requests: list[dict], requests_path: Path,
                    reference_mix: dict[str, float]) -> dict:
    """Row-count invariants (errors) and status-mix drift (a warning only:
    the evaluation set may legitimately differ from the samples)."""
    errors, warnings = [], []
    ids = [d.case_id for d in decisions]
    want = [r["case_id"] for r in requests]
    if Counter(ids) != Counter(want):
        errors.append(f"rows {len(ids)} != requests {len(want)} or case ids differ")
    if requests_path.name == "deletion_requests.csv" and len(ids) != EVAL_ROWS:
        errors.append(f"evaluation file must produce {EVAL_ROWS} rows, got {len(ids)}")
    mix = status_mix([d.status.value for d in decisions])
    drift = sum(abs(mix[k] - reference_mix.get(k, 0.0)) for k in mix) / 2
    if drift > MIX_DRIFT_WARN:
        warnings.append(f"status mix drifted from the sample mix by {drift:.3f} (> {MIX_DRIFT_WARN})")
    for e in errors:
        log.error("cross_row_failed", detail=e)
    for w in warnings:
        log.warning("cross_row_drift", detail=w)
    return {"errors": errors, "warnings": warnings, "status_mix": {k: round(v, 4) for k, v in mix.items()},
            "sample_mix": {k: round(v, 4) for k, v in reference_mix.items()}, "drift": round(drift, 4),
            "drift_warn_threshold": MIX_DRIFT_WARN}
