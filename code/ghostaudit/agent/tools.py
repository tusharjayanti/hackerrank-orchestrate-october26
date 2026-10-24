"""Tool contract for the investigator agent (L2).

The agent runs only on flagged cases. It can read and search evidence that
was available at the case's audit_at, and it can ask the policy engine what a
state means. It never writes steps: submit_decision hands the engine a set of
cited observations, and the engine generates, orders and numbers the steps.

Every input model is strict (no type coercion) and forbids extra fields; the
same models produce the JSON schemas sent to the model with strict: true.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..engine.decide import World
from ..engine.policy import treat
from ..identity import normalize_identifier
from ..ingest.manifest import parse_ts
from ..ingest.observations import category_for, text_identifiers
from ..schemas import DataCategory, ObservedState, ResourceState

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


class ProposedObservation(ToolInput):
    system_id: str
    resource_id: str
    observed_state: ObservedState
    evidence_id: str
    quote: str = Field(description="The exact text in the evidence that shows the state and the link to the user.")


class SubmitDecisionInput(ToolInput):
    case_id: str
    outcome: Literal["engine_default", "with_observations"]
    observations: tuple[ProposedObservation, ...] = ()
    rationale: str = Field(max_length=600)


class ToolError(ValueError):
    """Returned to the model as an error tool_result; never fatal to the run."""


def _request(world: World, case_id: str) -> dict:
    req = world.requests.get(case_id)
    if req is None:
        raise ToolError(f"unknown case {case_id}")
    return req


def _eligible(world: World, case_id: str, evidence_id: str) -> None:
    req = _request(world, case_id)
    f = world.manifest.by_id.get(evidence_id)
    if f is None:
        raise ToolError(f"{evidence_id} is not an evidence file")
    if f.captured_at > parse_ts(req["audit_at"]):
        raise ToolError(f"{evidence_id} was captured after this case's audit_at and cannot be used")


def list_observations(world: World, inp: ListObservationsInput) -> dict:
    """What the engine already knows about the case.

    Input: case_id. Output: the resolved user, every linked resource with its
    link layer, the observations behind each (file, time, state, identifiers),
    held vision_only rows, and the case flags.
    May decide: nothing. It is read-only.
    May not: return evidence captured after audit_at.
    """
    from ..engine.decide import case_context, decide_case

    req = _request(world, inp.case_id)
    _, trace = decide_case(world, req)
    ctx = case_context(world.identity, req)
    held = [o.model_dump(mode="json") for o in world.table.held_for_user(ctx.user_id, ctx.audit_at)] if ctx else []
    return {"identity": trace.get("identity"), "links": trace.get("links", []),
            "observations": trace.get("observations", []), "treatments": trace.get("treatments", []),
            "held_vision_only": held, "flags": trace.get("flags", [])}


def read_evidence_file(world: World, inp: ReadEvidenceFileInput) -> dict:
    """The raw content of one evidence file that existed at audit_at.

    Input: case_id, file_id. Output: text content (truncated at
    MAX_FILE_CHARS) for text formats and PDFs; for a PNG, the path the agent
    loop attaches as an image block.
    May decide: nothing. May not: open a file captured after audit_at, or any
    path outside the evidence manifest.
    """
    _eligible(world, inp.case_id, inp.file_id)
    f = world.manifest.by_id[inp.file_id]
    if f.format == "png":
        return {"file_id": f.file_id, "format": "png", "image_path": str(f.path)}
    if f.format == "pdf":
        import pypdf

        text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(f.path)).pages)
    else:
        text = f.path.read_text(encoding="utf-8")
    return {"file_id": f.file_id, "format": f.format, "truncated": len(text) > MAX_FILE_CHARS,
            "content": text[:MAX_FILE_CHARS]}


def search_raw_evidence(world: World, inp: SearchRawEvidenceInput) -> dict:
    """Exact lexical search for one identifier across eligible raw files.

    Input: case_id, identifier. Output: up to MAX_SEARCH_HITS hits of
    (file_id, line number, line text). Matching is exact on the normalised
    identifier (case-insensitive email, digits-only phone); there is no fuzzy
    or partial matching, so near-identical decoys never match.
    May decide: nothing. May not: search files captured after audit_at.
    """
    req = _request(world, inp.case_id)
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
    return {"hits": hits, "truncated": False}


_TOKEN = re.compile(r"[\w.+@-]+")


def _line_identifiers(line: str) -> set[str]:
    """Normalised identifiers on a line: whole tokens (emails, hashes, ids)
    plus phone numbers, which contain spaces."""
    return {normalize_identifier(t) for t in _TOKEN.findall(line)} | \
        {normalize_identifier(p) for p in text_identifiers(line)}


def apply_policy(world: World, inp: ApplyPolicyInput) -> dict:
    """What deletion_rules.csv requires for a resource in a given state.

    Input: case_id, system_id, resource_id, observed_state, evidence_id.
    Output: the data category, the finding (or none if compliant), the
    remediation, and the verification the engine would emit.
    May decide: nothing; it reports the rule. May not: accept evidence
    captured after audit_at or a state the policy table does not cover.
    """
    _eligible(world, inp.case_id, inp.evidence_id)
    category: DataCategory | None = category_for(inp.system_id)
    state = ResourceState(system_id=inp.system_id, resource_id=inp.resource_id, data_category=category,
                          state=inp.observed_state, evidence_id=inp.evidence_id,
                          as_of=world.manifest.by_id[inp.evidence_id].captured_at)
    t = treat(inp.system_id, inp.resource_id, category, state, world.rules)
    if t is None:
        raise ToolError(f"policy does not cover {category} in state {inp.observed_state}")
    return {"category": str(category), "finding": t.finding, "remediation": t.remediation,
            "verification": t.verification}


def submit_decision(world: World, inp: SubmitDecisionInput) -> dict:
    """End the investigation.

    Input: case_id; outcome "engine_default" (keep the engine's decision) or
    "with_observations" plus the cited observations the agent established;
    a short rationale. Output: the decision the engine generates.
    May decide: which cited observations exist (each must quote eligible
    evidence). May not: write, reorder or number steps, set the status
    directly, cite evidence captured after audit_at, or add a resource with
    no quote.
    """
    for o in inp.observations:
        _eligible(world, inp.case_id, o.evidence_id)
        if not o.quote.strip():
            raise ToolError(f"{o.resource_id}: every observation needs a quote from the evidence")
    raise NotImplementedError("engine re-run with agent observations is wired with the agent loop")


TOOLS = {
    "list_observations": (ListObservationsInput, list_observations),
    "read_evidence_file": (ReadEvidenceFileInput, read_evidence_file),
    "search_raw_evidence": (SearchRawEvidenceInput, search_raw_evidence),
    "apply_policy": (ApplyPolicyInput, apply_policy),
    "submit_decision": (SubmitDecisionInput, submit_decision),
}


def run_tool(world: World, name: str, payload: dict) -> dict:
    """Validate a tool call exactly as the model sent it (JSON, strict) and
    run it. Validation and tool errors come back as {"error": ...} so the
    agent can correct itself; they never end the run."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    model, fn = TOOLS[name]
    try:
        return fn(world, model.model_validate_json(json.dumps(payload)))
    except (ValueError, NotImplementedError) as e:  # pydantic ValidationError and ToolError are ValueErrors
        return {"error": str(e)}


def tool_specs() -> list[dict]:
    """Tool definitions for the Messages API, strict and schema-complete."""
    specs = []
    for name, (model, fn) in TOOLS.items():
        schema = model.model_json_schema()
        schema["additionalProperties"] = False
        specs.append({"name": name, "description": (fn.__doc__ or "").strip(), "strict": True,
                      "input_schema": schema})
    return specs
