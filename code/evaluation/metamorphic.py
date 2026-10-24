"""Metamorphic checks: perturbations of the input that must not change the
answer (or must change it in exactly one known way).

Each check copies dataset/ to a temp dir, perturbs the copy, loads
World.load(<copy>) and decides the 32 sample requests with the guards on.
Screenshots and intents replay from the content-keyed cache, so a check costs
no model call; the suite counts live calls before and after to prove it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ghostaudit.engine.decide import World, decide_case
from ghostaudit.guards import GuardContext, guard_decision
from ghostaudit.ingest.manifest import CitationRegistry
from ghostaudit.intent import classify
from ghostaudit.schemas import LinkLayer, case_sort_key

SEED = 20261024
FAR_FUTURE = "2099-01-01T00:00:00Z"
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TEXT_FORMATS = (".json", ".log", ".sql", ".txt")

Rows = dict[str, tuple[str, str]]  # case_id -> (status, steps_json)
Relied = dict[str, set[tuple[str, str]]]  # case_id -> {("system/resource", file_id)} the decision rests on


def requests_of(dataset: Path, name: str = "sample_deletion_requests.csv") -> list[dict]:
    with open(dataset / name, newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda r: case_sort_key(r["case_id"]))


def decide_rows(dataset: Path, requests: list[dict], relied: Relied | None = None) -> Rows:
    """Guarded rows for requests against the dataset at `dataset`. When
    `relied` is given it receives, per case, the (resource, file) pairs the
    engine took each resource's state from, cited or not (an INCOMPLETE row
    does not cite the file behind a compliant resource's verification)."""
    world = World.load(dataset)
    ctx = GuardContext(world, CitationRegistry.load(dataset))
    intents = classify([r["request_text"] for r in requests], world.llm)
    out: Rows = {}
    for r in requests:
        intent = intents[r["request_text"]]
        # The agent is off here: these checks are about the deterministic
        # engine, and each perturbed world would otherwise wake it (and pay)
        # on every case the perturbation empties.
        d, t = decide_case(world, r, LinkLayer.SIBLING, intent, use_agent=False)
        d, t = guard_decision(d, t, r, ctx,
                              regenerate=lambda: decide_case(world, r, LinkLayer.SIBLING, intent, use_agent=False))
        row = d.to_row()
        if relied is not None:
            relied[r["case_id"]] = {(tr["resource"], tr["evidence"]) for tr in t.get("treatments", [])
                                    if tr.get("evidence")}
        out[r["case_id"]] = (row["deletion_status"], row["steps_json"])
    return out


def cited(rows: Rows) -> dict[str, set[str]]:
    return {cid: {s["evidence_id"] for s in json.loads(sj) if s["evidence_id"]} for cid, (_, sj) in rows.items()}


def cited_pairs(rows: Rows) -> Relied:
    return {cid: {(f"{s['system_id']}/{s['resource_id']}", s["evidence_id"]) for s in json.loads(sj)
                  if s["evidence_id"]} for cid, (_, sj) in rows.items()}


def diff(a: Rows, b: Rows) -> list[str]:
    return sorted((c for c in a.keys() | b.keys() if a.get(c) != b.get(c)), key=case_sort_key)


# ---------------------------------------------------------------- perturbations

def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def shuffle_manifest(ds: Path, rng: random.Random) -> str:
    header, rows = _read_csv(ds / "evidence_files.csv")
    rng.shuffle(rows)
    _write_csv(ds / "evidence_files.csv", header, rows)
    return f"{len(rows)} manifest rows shuffled"


def _text_file(ds: Path, file_id: str) -> Path:
    _, rows = _read_csv(ds / "evidence_files.csv")
    return ds / next(r["file_path"] for r in rows if r["file_id"] == file_id)


def uppercase_emails(ds: Path, file_id: str) -> str:
    path = _text_file(ds, file_id)
    text = path.read_text(encoding="utf-8")
    n = len(_EMAIL.findall(text))
    path.write_text(_EMAIL.sub(lambda m: m.group(0).upper(), text), encoding="utf-8")
    return f"{n} emails uppercased in {file_id}"


def inject_decoy(ds: Path, subject_email: str) -> str:
    """A new user whose email differs from a real subject's by one character
    and appears in no evidence; phone and hash are fresh."""
    header, rows = _read_csv(ds / "user_identities.csv")
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="ignore").lower()
                       for p in (ds / "evidence").rglob("*") if p.suffix in TEXT_FORMATS)
    local, domain = subject_email.split("@", 1)
    known = {r["email"].lower() for r in rows}
    for k in range(1, 100):
        decoy = f"{local}{k}@{domain}"
        if decoy.lower() not in corpus and decoy.lower() not in known:
            break
    template = next(r for r in rows if r["email"].lower() == subject_email.lower()) if any(
        r["email"].lower() == subject_email.lower() for r in rows) else rows[0]
    phones = {re.sub(r"\D", "", r["phone"]) for r in rows}
    phone = next(f"+1 555 01{n:02d} 77{n:02d}" for n in range(100)
                 if re.sub(r"\D", "", f"+1 555 01{n:02d} 77{n:02d}") not in phones
                 and re.sub(r"\D", "", f"+1 555 01{n:02d} 77{n:02d}") not in corpus)
    rows.append({**template, "user_id": "user_9999001", "email": decoy, "phone": phone,
                 "email_hash": hashlib.sha256(decoy.lower().encode()).hexdigest()})
    _write_csv(ds / "user_identities.csv", header, rows)
    return f"decoy user_9999001 whose email is the subject's with '{k}' appended to the local part"


def delay_file(ds: Path, file_id: str) -> str:
    header, rows = _read_csv(ds / "evidence_files.csv")
    for r in rows:
        if r["file_id"] == file_id:
            r["captured_at"] = FAR_FUTURE
    _write_csv(ds / "evidence_files.csv", header, rows)
    return f"{file_id} captured_at -> {FAR_FUTURE}"


# ---------------------------------------------------------------- choice of targets

def pick_cited_text_file(base: Path, rows: Rows, with_email: bool) -> str | None:
    """The text-format evidence file cited by the most sample rows (ties by
    file_id), optionally one that contains an email."""
    counts: dict[str, int] = {}
    for ids in cited(rows).values():
        for e in ids:
            counts[e] = counts.get(e, 0) + 1
    for fid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        try:
            path = _text_file(base, fid)
        except StopIteration:
            continue
        if path.suffix not in TEXT_FORMATS:
            continue
        if with_email and not _EMAIL.search(path.read_text(encoding="utf-8")):
            continue
        return fid
    return None


# ---------------------------------------------------------------- runner

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    changed: list[str] = field(default_factory=list)


def _perturbed(base: Path, requests: list[dict], perturb: Callable[[Path], str],
               relied: Relied | None = None) -> tuple[Rows, str]:
    with tempfile.TemporaryDirectory(prefix="ghostaudit-meta-") as tmp:
        ds = Path(tmp) / "dataset"
        shutil.copytree(base, ds)
        what = perturb(ds)
        return decide_rows(ds, requests, relied), what


def run_checks(base: Path, requests: list[dict] | None = None) -> list[Check]:
    requests = requests or requests_of(base)
    rng = random.Random(SEED)
    relied_before: Relied = {}
    first = decide_rows(base, requests, relied_before)
    second = decide_rows(base, requests)
    checks = [Check("rerun_identical", first == second, "two runs on the same data", diff(first, second))]

    got, what = _perturbed(base, requests, lambda ds: shuffle_manifest(ds, rng))
    checks.append(Check("manifest_order", got == first, what, diff(first, got)))

    fid = pick_cited_text_file(base, first, with_email=True)
    if fid:
        got, what = _perturbed(base, requests, lambda ds: uppercase_emails(ds, fid))
        checks.append(Check("email_case", got == first, what, diff(first, got)))

    subject = requests[0]["email"]
    got, what = _perturbed(base, requests, lambda ds: inject_decoy(ds, subject))
    checks.append(Check("decoy_user", got == first, what, diff(first, got)))

    fid = pick_cited_text_file(base, first, with_email=False)
    if fid:
        # Moving one file past every audit_at removes its citations. It may
        # flip a row (its findings were in that file) and so surface a
        # citation the INCOMPLETE row left blank, but every citation after
        # must be a (resource, file) pair the engine already relied on.
        relied_after: Relied = {}
        got, what = _perturbed(base, requests, lambda ds: delay_file(ds, fid), relied_after)
        before, after = cited(first), cited(got)
        still = sorted((c for c, ids in after.items() if fid in ids), key=case_sort_key)
        relied_on_late = sorted((c for c, pairs in relied_after.items() if any(e == fid for _, e in pairs)),
                                key=case_sort_key)
        new_pairs = {c: sorted(f"{r}@{e}" for r, e in pairs - relied_before.get(c, set()))
                     for c, pairs in cited_pairs(got).items() if pairs - relied_before.get(c, set())}
        lost = sorted((c for c, ids in before.items() if fid in ids), key=case_sort_key)
        flipped = sorted((c for c in got if got[c][0] != first[c][0]), key=case_sort_key)
        ok = not still and not relied_on_late and not new_pairs
        detail = (f"{what}: cited by {len(lost)} rows before, {len(still)} after; relied on after: "
                  f"{len(relied_on_late)}; citations not relied on before: {len(new_pairs)}; "
                  f"status flips: {flipped or 'none'}")
        if new_pairs:
            detail += " " + json.dumps(new_pairs, sort_keys=True)
        checks.append(Check("late_file_vanishes", ok, detail, diff(first, got)))
    return checks
