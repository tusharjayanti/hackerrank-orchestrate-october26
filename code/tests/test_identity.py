import hashlib

from ghostaudit.identity import (
    IdentityIndex,
    MatchKind,
    normalize_email,
    normalize_identifier,
    normalize_phone,
)


def _sha(e):
    return hashlib.sha256(e.encode()).hexdigest()


SNAPSHOT = [
    {"user_id": "user_1", "name": "Jonas Berg", "email": "jonas.berg@example.com",
     "phone": "+46 70 808 74 05", "email_hash": _sha("jonas.berg@example.com"), "updated_at": "2026-01-01T00:00:00Z"},
    {"user_id": "user_2", "name": "Jonas Berglund", "email": "jonas.berglund@example.com",
     "phone": "+1 (312) 555-0165", "email_hash": _sha("jonas.berglund@example.com"), "updated_at": "2026-01-01T00:00:00Z"},
]
HISTORY = [
    {"audit_id": "audit_1", "user_id": "user_1", "event_time": "2025-01-01T00:00:00Z",
     "field_name": "email", "old_value": "j.berg@old.example", "new_value": "jonas.berg@example.com"},
    {"audit_id": "audit_2", "user_id": "user_1", "event_time": "2025-01-01T00:00:00Z",
     "field_name": "email_hash", "old_value": _sha("j.berg@old.example"), "new_value": _sha("jonas.berg@example.com")},
    {"audit_id": "audit_3", "user_id": "user_1", "event_time": "2024-06-01T00:00:00Z",
     "field_name": "phone", "old_value": "+46 70 144 08 06", "new_value": "+46 70 808 74 05"},
]


def index():
    return IdentityIndex.build(SNAPSHOT, HISTORY)


def test_normalisation():
    assert normalize_email("  Jonas.BERG@Example.com ") == "jonas.berg@example.com"
    assert normalize_phone("+1 (312) 555-0165") == "13125550165"
    assert normalize_phone("+55 11 96656-0765") == "5511966560765"
    assert normalize_identifier("JONAS.BERG@EXAMPLE.COM") == "jonas.berg@example.com"
    assert normalize_identifier("+46 70 808 74 05") == "46708087405"
    assert normalize_identifier("USER_1") == "user_1"
    assert normalize_identifier(_sha("x").upper()) == _sha("x")


def test_current_and_historical_resolution():
    idx = index()
    assert idx.resolve_email("jonas.berg@example.com").kind is MatchKind.CURRENT
    r = idx.resolve_email("J.Berg@old.example")
    assert (r.user_id, r.kind) == ("user_1", MatchKind.HISTORICAL)
    assert idx.resolve_email("nobody@example.com").kind is MatchKind.NONE


def test_reverse_map_covers_every_identifier_kind():
    idx = index()
    assert idx.owner_of("+46 70 144 08 06") == "user_1"  # historical phone
    assert idx.owner_of("46701440806") == "user_1"
    assert idx.owner_of(_sha("j.berg@old.example")) == "user_1"  # historical hash
    assert idx.owner_of("user_1") == "user_1"
    assert idx.owner_of("+1 312-555-0165") == "user_2"


def test_decoys_never_match_by_prefix_or_name():
    idx = index()
    assert idx.owner_of("jonas.berg@example.co") is None
    assert idx.owner_of("jonas.berglund@example.com") == "user_2"
    assert idx.owner_of("Jonas Berg") is None


def test_shared_identifier_is_ambiguous_not_guessed():
    snap = SNAPSHOT + [{"user_id": "user_3", "name": "X", "email": "x@example.com",
                        "phone": "+46 70 808 74 05", "email_hash": _sha("x@example.com"), "updated_at": ""}]
    idx = IdentityIndex.build(snap, HISTORY)
    assert idx.owner_of("+46 70 808 74 05") is None


def test_real_dataset_resolves_every_request():
    import csv
    from ghostaudit import DATASET_DIR
    idx = IdentityIndex.load()
    with open(DATASET_DIR / "deletion_requests.csv", newline="", encoding="utf-8") as f:
        kinds = [idx.resolve_email(r["email"]).kind for r in csv.DictReader(f)]
    assert len(kinds) == 518
    assert kinds.count(MatchKind.NONE) == 0
