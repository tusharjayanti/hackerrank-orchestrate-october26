"""Tool contract for the investigator agent (L2).

The agent runs only on gated cases. It can read and search evidence that was
available at the case's audit_at, and it can ask the policy engine what a
state means. It never writes a resource id or a step: the engine generates
candidate decisions for the case (agent/candidates.py), and submit_decision
accepts only one of those candidates, by id.

Every input model is strict (no type coercion) and forbids extra fields; the
same models produce the JSON schemas sent to the model with strict: true.
Tools run against a Session (the case under review, its engine trace and its
candidates); outside a session the read-only tools still work and
submit_decision refuses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ..engine.policy import treat
from ..identity import normalize_identifier
from ..ingest.manifest import parse_ts
from ..ingest.observations import category_for, text_identifiers
from ..obs import get_logger
from ..schemas import DataCategory, ObservedState, ResourceState

if TYPE_CHECKING:
    from ..engine.decide import World
    from .candidates import Candidate

log = get_logger(__name__)

MAX_FILE_CHARS = 20_000
MAX_SEARCH_HITS = 50


class ToolInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ListObservationsInput(ToolInput):
    case_id: str = Field(description="The case under review, e.g. case_61.")


class ReadEvidenceFileInput(ToolInput):
    case_id: str
    file_id: str = Field(description="An evidence_files.csv id, e.g. evidence_37.")


class SearchRawEvidenceInput(ToolInput):
    case_id: str
    identifier: str = Field(description="An exact email, phone, email hash, user_id or resource id.")


class ApplyPolicyInput(ToolInput):
    case_id: str
    system_id: str
    resource_id: str
    observed_state: ObservedState
    evidence_id: str = Field(description="The file or activity id that shows this state.")


class SubmitDecisionInput(ToolInput):
    case_id: str
    candidate_id: str = Field(description="The id of one candidate the engine generated for this case.")
    steps_json: str | None = Field(description="null, or that candidate's steps_json copied exactly.")
    rationale: str = Field(description="Two or three sentences: the evidence that decided the choice.")


class ToolError(ValueError):
    """Returned to the model as an error tool_result; never fatal to the run."""


@dataclass
class Session:
    """The case one agent invocation is working on."""

    request: dict
    trace: dict
    candidates: list[Candidate]
    accepted: Candidate | None = None
    rejections: list[str] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return self.request["case_id"]


def _request(world: World, case_id: str, session: Session | None) -> dict:
    if session is not None:
        if case_id != session.case_id:
            raise ToolError(f"this investigation is {session.case_id}, not {case_id}")
        return session.request
    req = world.requests.get(case_id)
    if req is None:
        raise ToolError(f"unknown case {case_id}")
    return req


def _eligible(world: World, req: dict, evidence_id: str) -> None:
    f = world.manifest.by_id.get(evidence_id)
    if f is None:
        raise ToolError(f"{evidence_id} is not an evidence file")
    if f.captured_at > parse_ts(req["audit_at"]):
        raise ToolError(f"{evidence_id} was captured after this case's audit_at and cannot be used")


def _engine_trace(world: World, req: dict, session: Session | None) -> dict:
    if session is not None:
        return session.trace
    from dataclasses import replace

    from ..engine.decide import decide_case

    # Engine-only (no model): never re-enters the agent.
    return decide_case(replace(world, llm=None), req)[1]


def list_observations(world: World, inp: ListObservationsInput, session: Session | None = None) -> dict:
    """What the engine already knows about the case.

    Input: case_id. Output: the resolved user, every linked resource with its
    link layer, resources the scope rule left out, the observations behind
    each (file, time, state, identifiers), the engine's treatments, held
    vision_only rows, the case flags, and the candidate decisions (id,
    description, status, number of steps).
    May decide: nothing. It is read-only.
    May not: return evidence captured after audit_at.
    """
    req = _request(world, inp.case_id, session)
    trace = _engine_trace(world, req, session)
    uid = (trace.get("identity") or {}).get("user_id")
    audit_at = parse_ts(req["audit_at"])
    held = [o.model_dump(mode="json") for o in world.table.held_for_user(uid, audit_at)] if uid else []
    out = {"identity": trace.get("identity"), "links": trace.get("links", []),
           "out_of_scope": trace.get("out_of_scope", []), "observations": trace.get("observations", []),
           "treatments": trace.get("treatments", []), "held_vision_only": held,
           "flags_fired": trace.get("flags_fired", []), "notes": trace.get("notes", [])}
    if session is not None:
        out["candidates"] = [c.summary() for c in session.candidates]
    return out


def read_evidence_file(world: World, inp: ReadEvidenceFileInput, session: Session | None = None) -> dict:
    """The raw content of one evidence file that existed at audit_at.

    Input: case_id, file_id. Output: text content (truncated at
    MAX_FILE_CHARS) for text formats and PDFs; for a PNG, the path the agent
    loop attaches as an image block.
    May decide: nothing. May not: open a file captured after audit_at, or any
    path outside the evidence manifest.
    """
    _eligible(world, _request(world, inp.case_id, session), inp.file_id)
    f = world.manifest.by_id[inp.file_id]
    if f.format == "png":
        return {"file_id": f.file_id, "format": "png", "captured_at": f.captured_at.isoformat(),
                "image_path": str(f.path)}
    if f.format == "pdf":
        import pypdf

        text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(f.path)).pages)
    else:
        text = f.path.read_text(encoding="utf-8")
    return {"file_id": f.file_id, "format": f.format, "captured_at": f.captured_at.isoformat(),
            "truncated": len(text) > MAX_FILE_CHARS, "content": text[:MAX_FILE_CHARS]}


def search_raw_evidence(world: World, inp: SearchRawEvidenceInput, session: Session | None = None) -> dict:
    """Exact lexical search for one identifier across eligible raw text files
    and system_activity rows.

    Input: case_id, identifier. Output: up to MAX_SEARCH_HITS hits of
    (file_id, line number, line text) or (activity_id, row text). Matching is
    exact on the normalised identifier (case-insensitive email, digits-only
    phone); there is no fuzzy or partial matching, so near-identical decoys
    never match. Activity rows are claims, not proof of state.
    May decide: nothing. May not: search files captured, or activity that
    happened, after audit_at.
    """
    req = _request(world, inp.case_id, session)
    audit_at = parse_ts(req["audit_at"])
    needle = normalize_identifier(inp.identifier)
    hits = []
    for f in world.manifest.files:
        if f.captured_at > audit_at or f.format in ("png", "pdf"):
            continue
        for n, line in enumerate(f.path.read_text(encoding="utf-8").splitlines(), start=1):
            if needle in _line_identifiers(line):
                hits.append({"file_id": f.file_id, "line": n, "text": line[:400]})
                if len(hits) >= MAX_SEARCH_HITS:
                    return {"hits": hits, "truncated": True}
    for e in world.activity.events:
        if e.event_time > audit_at:
            break
        if needle == e.resource_id.lower() or needle in _line_identifiers(e.details):
            hits.append({"activity_id": e.activity_id, "text": f"{e.event_time.isoformat()} {e.event_type} "
                         f"{e.source_system}->{e.target_system} {e.resource_id} {e.reported_result}: "
                         f"{e.details}"[:400]})
            if len(hits) >= MAX_SEARCH_HITS:
                return {"hits": hits, "truncated": True}
    return {"hits": hits, "truncated": False}


_TOKEN = re.compile(r"[\w.+@-]+")


def _line_identifiers(line: str) -> set[str]:
    """Normalised identifiers on a line: whole tokens (emails, hashes, ids)
    plus phone numbers, which contain spaces."""
    return {normalize_identifier(t) for t in _TOKEN.findall(line)} | \
        {normalize_identifier(p) for p in text_identifiers(line)}


def apply_policy(world: World, inp: ApplyPolicyInput, session: Session | None = None) -> dict:
    """What deletion_rules.csv requires for a resource in a given state.

    Input: case_id, system_id, resource_id, observed_state, evidence_id.
    Output: the data category, the finding (or none if compliant), the
    remediation, and the verification the engine would emit.
    May decide: nothing; it reports the rule. May not: accept evidence
    captured after audit_at or a state the policy table does not cover.
    """
    _eligible(world, _request(world, inp.case_id, session), inp.evidence_id)
    category: DataCategory | None = category_for(inp.system_id)
    state = ResourceState(system_id=inp.system_id, resource_id=inp.resource_id, data_category=category,
                          state=inp.observed_state, evidence_id=inp.evidence_id,
                          as_of=world.manifest.by_id[inp.evidence_id].captured_at)
    t = treat(inp.system_id, inp.resource_id, category, state, world.rules)
    if t is None:
        raise ToolError(f"policy does not cover {category} in state {inp.observed_state}")
    return {"category": str(category), "finding": t.finding, "remediation": t.remediation,
            "verification": t.verification}


def submit_decision(world: World, inp: SubmitDecisionInput, session: Session | None = None) -> dict:
    """End the investigation by choosing one engine-generated candidate.

    Input: case_id; candidate_id (one of the ids listed for this case);
    steps_json, either null or the chosen candidate's steps_json copied
    byte for byte; a short rationale. Output: the accepted candidate.
    May decide: which of the engine's candidates is the case's answer.
    May not: write, edit, reorder or number steps, name a resource, set the
    status directly, or submit an id or step list the engine did not
    generate for this case. A rejected submission returns the reason.
    """
    if session is None:
        raise ToolError("submit_decision is only available inside an agent investigation")
    _request(world, inp.case_id, session)
    by_id = {c.id: c for c in session.candidates}
    cand = by_id.get(inp.candidate_id)
    if cand is None:
        raise ToolError(f"{inp.candidate_id!r} is not a candidate for {session.case_id}; "
                        f"valid ids: {sorted(by_id)}")
    if inp.steps_json is not None and inp.steps_json != cand.steps_json:
        raise ToolError(f"steps_json does not match candidate {cand.id} exactly; submit the id with "
                        "steps_json null, or copy that candidate's steps_json unchanged")
    session.accepted = cand
    return {"accepted": True, "candidate_id": cand.id, "status": cand.status}


TOOLS = {
    "list_observations": (ListObservationsInput, list_observations),
    "read_evidence_file": (ReadEvidenceFileInput, read_evidence_file),
    "search_raw_evidence": (SearchRawEvidenceInput, search_raw_evidence),
    "apply_policy": (ApplyPolicyInput, apply_policy),
    "submit_decision": (SubmitDecisionInput, submit_decision),
}
ALLOWED = tuple(TOOLS)


def run_tool(world: World, name: str, payload: dict, session: Session | None = None) -> dict:
    """Validate a tool call exactly as the model sent it (JSON, strict) and
    run it. Unknown tools, validation errors and any exception a tool raises
    come back as {"error": ...} so the agent can correct itself; they never
    end the run."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name}; allowed: {list(ALLOWED)}"}
    model, fn = TOOLS[name]
    try:
        return fn(world, model.model_validate_json(json.dumps(payload)), session)
    except ValueError as e:  # pydantic ValidationError and ToolError are ValueErrors
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 - a failing tool is a result, not a crash
        log.error("agent_tool_failed", tool=name, error=repr(e))
        return {"error": f"tool {name} failed: {type(e).__name__}: {e}"}


# Keywords strict tool use does not accept; pydantic emits some of them.
_UNSUPPORTED = {"maxLength", "minLength", "minimum", "maximum", "multipleOf", "default"}


def _strict_schema(node):
    if isinstance(node, dict):
        out = {k: _strict_schema(v) for k, v in node.items() if k not in _UNSUPPORTED}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = sorted(out["properties"])
        return out
    if isinstance(node, list):
        return [_strict_schema(v) for v in node]
    return node


def tool_specs() -> list[dict]:
    """Tool definitions for the Messages API, strict and schema-complete."""
    return [{"name": name, "description": (fn.__doc__ or "").strip(), "strict": True,
             "input_schema": _strict_schema(model.model_json_schema())}
            for name, (model, fn) in TOOLS.items()]
