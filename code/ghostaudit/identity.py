"""Identity reconstruction from the current snapshot plus the audit history.

Matching is exact on normalised identifiers only. Names are kept for
redaction checks but never used to link records: adjacent decoys such as
jonas.berg / jonas.berglund share name fragments, so any fuzzy or substring
match would merge different people.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from . import DATASET_DIR
from .schemas import IdentitySet
from .telemetry import get_logger

log = get_logger(__name__)

_NON_DIGIT = re.compile(r"\D")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_phone(value: str) -> str:
    # Formats vary by country ("+1 (312) 555-0165", "+55 11 96656-0765");
    # the digit sequence including country code is the identity.
    return _NON_DIGIT.sub("", value)


def normalize_hash(value: str) -> str:
    return value.strip().lower()


def normalize_identifier(value: str) -> str:
    """Normalise an identifier of unknown kind, as found in evidence."""
    v = value.strip()
    if "@" in v:
        return normalize_email(v)
    if _HEX64.match(v.lower()):
        return normalize_hash(v)
    if v.lower().startswith("user_"):
        return v.lower()
    digits = normalize_phone(v)
    # Only treat as a phone if it is mostly digits; otherwise keep as-is so
    # that free text never collapses to a short digit string.
    if digits and len(digits) >= 7 and len(digits) >= 0.6 * len(v.replace(" ", "")):
        return digits
    return v.lower()


class MatchKind(StrEnum):
    CURRENT = "current"
    HISTORICAL = "historical"
    NONE = "none"


@dataclass(frozen=True)
class Resolution:
    user_id: str | None
    kind: MatchKind


class IdentityIndex:
    def __init__(self, users: dict[str, IdentitySet], reverse: dict[str, str], current_emails: set[str],
                 ambiguous: frozenset[str] = frozenset()):
        self.users = users
        self._reverse = reverse
        self._current_emails = current_emails
        self.ambiguous = ambiguous  # normalised identifiers held by several users

    @classmethod
    def load(cls, dataset_dir: Path = DATASET_DIR) -> IdentityIndex:
        with open(dataset_dir / "user_identities.csv", newline="", encoding="utf-8") as f:
            snapshot = list(csv.DictReader(f))
        with open(dataset_dir / "user_audit_log.csv", newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        return cls.build(snapshot, history)

    @classmethod
    def build(cls, snapshot: list[dict], history: list[dict]) -> IdentityIndex:
        emails: dict[str, set[str]] = defaultdict(set)
        phones: dict[str, set[str]] = defaultdict(set)
        hashes: dict[str, set[str]] = defaultdict(set)
        historical: dict[str, set[str]] = defaultdict(set)
        base = {}
        for row in snapshot:
            uid = row["user_id"]
            base[uid] = row
            emails[uid].add(normalize_email(row["email"]))
            phones[uid].add(normalize_phone(row["phone"]))
            hashes[uid].add(normalize_hash(row["email_hash"]))

        for row in history:
            uid = row["user_id"]
            if uid not in base:
                # Fallback: history for a user absent from the snapshot cannot
                # be anchored to a current identity; skip rather than invent one.
                log.warning("audit_row_unknown_user", audit_id=row["audit_id"], user_id=uid)
                continue
            field = row["field_name"]
            for value in (row["old_value"], row["new_value"]):
                if not value:
                    continue
                if field == "email":
                    emails[uid].add(normalize_email(value))
                elif field == "phone":
                    phones[uid].add(normalize_phone(value))
                elif field == "email_hash":
                    hashes[uid].add(normalize_hash(value))
                else:
                    log.warning("audit_field_unmapped", audit_id=row["audit_id"], field=field)
            if field == "email" and row["old_value"]:
                historical[uid].add(normalize_email(row["old_value"]))

        users: dict[str, IdentitySet] = {}
        for uid, row in base.items():
            users[uid] = IdentitySet(
                user_id=uid,
                name=row["name"],
                current_email=row["email"],
                current_phone=row["phone"],
                current_email_hash=row["email_hash"],
                emails=frozenset(emails[uid]),
                phones=frozenset(phones[uid]),
                email_hashes=frozenset(hashes[uid]),
                historical_emails=frozenset(historical[uid] - {normalize_email(row["email"])}),
            )

        owners: dict[str, set[str]] = defaultdict(set)
        for uid, ident in users.items():
            for key in ident.identifiers():
                owners[key].add(uid)
        reverse: dict[str, str] = {}
        ambiguous = set()
        for key, uids in sorted(owners.items()):
            if len(uids) == 1:
                reverse[key] = next(iter(uids))
            else:
                # Fallback: an identifier shared by several users links to
                # nobody; attributing it to either would be a guess.
                # Not implemented (3 shared identifiers in this dataset):
                # resolving to the user who held it at the observation's
                # captured_at, from user_audit_log event_time.
                log.warning("identifier_ambiguous", identifier_kind=_kind(key), users=sorted(uids))
                ambiguous.add(key)
        current = {normalize_email(r["email"]) for r in base.values()}
        return cls(users, reverse, current, frozenset(ambiguous))

    def owner_of(self, identifier: str) -> str | None:
        """Reverse map: any identifier seen in evidence -> user_id."""
        return self._reverse.get(normalize_identifier(identifier))

    def is_ambiguous(self, identifier: str) -> bool:
        return normalize_identifier(identifier) in self.ambiguous

    def resolve_email(self, email: str) -> Resolution:
        key = normalize_email(email)
        uid = self._reverse.get(key)
        if uid is None:
            return Resolution(None, MatchKind.NONE)
        kind = MatchKind.CURRENT if key in self._current_emails else MatchKind.HISTORICAL
        return Resolution(uid, kind)


def _kind(identifier: str) -> str:
    if "@" in identifier:
        return "email"
    if _HEX64.match(identifier):
        return "email_hash"
    if identifier.startswith("user_"):
        return "user_id"
    return "phone"
