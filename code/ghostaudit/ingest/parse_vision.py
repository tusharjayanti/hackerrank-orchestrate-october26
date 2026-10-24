"""PDF documents and PNG screenshots.

PDFs carry a text layer and are read deterministically with pypdf; no model
call. PNGs are read by the model twice with different prompts; a row is kept
when both readings agree on (resource, state, identifiers). On disagreement
the reading worse for the subject wins and the observation is flagged.

Every row passes a universe check before it becomes an observation: the id
must match the resource-id pattern, and a model-read id must also be known
from text evidence or system_activity (a misread digit must not invent a
record). Catalog labels from LINK_RESOURCE_IDENTITY rows (a statement number,
a file name) resolve to their resource id.
"""

from __future__ import annotations

import re

from ..identity import normalize_identifier
from ..llm import LLM, MODEL, SONNET, LLMUnavailable, image_block
from ..obs import get_logger
from ..schemas import EXPOSURE, ObservedState
from . import vision_prompts as vp
from .manifest import EvidenceFile
from .observations import ParseResult, observe_record, text_identifiers

log = get_logger(__name__)

RESOURCE_RE = re.compile(
    r"\b(?:account|search|analytics|support|export|data_flow|backup|audit_record|tombstone|legal_hold|marketing"
    r"|cache|email_delivery|telemetry|object)_\d+\b")
USER_RE = re.compile(r"\buser_\d+\b")
_ATTACHMENT_HINT = re.compile(r"attachment", re.IGNORECASE)


class Universe:
    def __init__(self, resource_ids: set[str], labels: dict[str, str]):
        self.resource_ids = resource_ids
        self.labels = labels  # printed label -> resource id

    def known(self, rid: str) -> bool:
        return rid in self.resource_ids


def parse(f: EvidenceFile, universe: Universe, llm: LLM | None) -> ParseResult:
    if f.format == "pdf":
        return _parse_pdf(f, universe)
    if llm is None:
        raise LLMUnavailable("no model client")
    return _parse_png(f, universe, llm)


# ---------------------------------------------------------------- PDF

def _parse_pdf(f: EvidenceFile, universe: Universe) -> ParseResult:
    import pypdf

    res = ParseResult()
    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(f.path)).pages)
    rids = set(RESOURCE_RE.findall(text))
    # A label that is itself a user reference (tombstone_17 is catalogued as
    # "user_17") names a person, not a record; it never resolves a resource.
    rids |= {rid for label, rid in universe.labels.items()
             if label and not USER_RE.fullmatch(label) and label in text}
    users = set(USER_RE.findall(text))
    contacts = set(text_identifiers(text))
    subjects = contacts | users
    if len(users) > 1 or sum("@" in c for c in contacts) > 1:
        # A register listing several people: identifiers cannot be assigned to
        # a single record, so none are attached.
        res.notes.append(f"multi-subject document ({len(users)} users, {len(contacts)} contacts); ids not attached")
        subjects = set()
    collection = "ticket_attachments" if _ATTACHMENT_HINT.search(text) else None
    for rid in sorted(rids):
        res.rows += 1
        record = {"resource_id": rid}
        emails = [c for c in contacts if "@" in c] if subjects else []
        phones = [c for c in contacts if "@" not in c] if subjects else []
        if emails:
            record["email"] = emails[0]
        if phones:
            record["phone"] = phones[0]
        o = observe_record(f, f.captured_at, record, collection=collection, source_detail="pdf:text",
                           extra_ids=sorted(subjects))
        if o is None:
            res.skipped += 1
            continue
        res.observations.append(o.model_copy(update={"vision": True}))
    if not rids:
        res.notes.append("no resource id or catalog label in document")
    return res


# ---------------------------------------------------------------- PNG

def _reading(f: EvidenceFile, llm: LLM, instruction: str, tag: str, model: str) -> dict[str, dict]:
    content = [image_block(f.path, "image/png"), {"type": "text", "text": instruction}]
    out = llm.json_call(purpose=f"vision:{tag}:{f.file_id}", prompt_version=f"{vp.PROMPT_VERSION}:{tag}",
                        system=vp.SYSTEM, content=content, schema=vp.SCHEMA, max_tokens=8000, effort="medium",
                        model=model)
    return {r["resource_id"].strip(): r for r in out.get("records", [])}


def _to_record(row: dict) -> dict:
    rec = {"resource_id": row["resource_id"].strip()}
    for key, field in (("emails", "email"), ("phones", "phone"), ("names", "full_name"),
                       ("email_hashes", "email_hash")):
        if row.get(key):
            rec[field] = row[key][0]
    if row.get("user_references"):
        rec["customer_reference"] = row["user_references"][0]
    if row.get("enabled") is not None:
        rec["enabled"] = row["enabled"]
    if row.get("restore_allowed") is not None:
        rec["restore_allowed"] = row["restore_allowed"]
    return rec


def _ids(row: dict) -> frozenset[str]:
    vals = [*row.get("emails", []), *row.get("phones", []), *row.get("email_hashes", []),
            *row.get("user_references", [])]
    return frozenset(normalize_identifier(v) for v in vals if v)


def _observe(f: EvidenceFile, row: dict | None, tag: str):
    if row is None:
        return None
    o = observe_record(f, f.captured_at, _to_record(row), absent=bool(row["not_found_or_deleted"]),
                       source_detail=f"png:reading{tag}")
    return None if o is None else (o, _ids(row))


def _same(a, b) -> bool:
    return a is not None and b is not None and a[0].state == b[0].state and a[1] == b[1]


def _parse_png(f: EvidenceFile, universe: Universe, llm: LLM) -> ParseResult:
    """Two Sonnet readings; any disagreement in the file triggers one Opus
    reading of the whole file as tiebreak."""
    res = ParseResult()
    readings = {"A": _reading(f, llm, vp.READING_A, "A", SONNET), "B": _reading(f, llm, vp.READING_B, "B", SONNET)}
    rids = sorted(set(readings["A"]) | set(readings["B"]))
    known = []
    for rid in rids:
        if not RESOURCE_RE.fullmatch(rid) or not universe.known(rid):
            log.warning("vision_resource_unknown", file_id=f.file_id, resource_id=rid)
            res.notes.append(f"dropped unknown resource {rid}")
            res.rows += 1
            res.skipped += 1
            continue
        known.append(rid)
    pairs = {rid: (_observe(f, readings["A"].get(rid), "A"), _observe(f, readings["B"].get(rid), "B"))
             for rid in known}
    disputed = [rid for rid, (a, b) in pairs.items() if not _same(a, b)]
    tiebreak: dict[str, dict] = {}
    if disputed:
        log.info("vision_escalated", file_id=f.file_id, disputed=disputed, model=MODEL)
        res.notes.append(f"escalated to {MODEL}: {disputed}")
        tiebreak = _reading(f, llm, vp.READING_A, "T", MODEL)
    for rid in known:
        res.rows += 1
        a, b = pairs[rid]
        if rid not in disputed:
            chosen, detail = a, "png:agreed"
        else:
            t = _observe(f, tiebreak.get(rid), "T")
            winner = next((x for x in (a, b) if _same(x, t)), None)
            if winner is not None:
                chosen, detail = winner, "png:tiebreak"
            else:
                # No two readings agree: keep the one worse for the subject.
                options = [x for x in (a, b, t) if x is not None]
                chosen = max(options, key=lambda c: (EXPOSURE[c[0].state], len(c[1])))
                detail = "png:disagreed"
                log.warning("vision_disagreement", file_id=f.file_id, resource_id=rid,
                            states=[x[0].state.value for x in options])
        if chosen is None:
            res.skipped += 1
            continue
        res.observations.append(chosen[0].model_copy(update={"vision": True, "source_detail": detail}))
    return res


__all__ = ["parse", "Universe", "RESOURCE_RE", "ObservedState"]
