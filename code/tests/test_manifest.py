from datetime import datetime, timedelta, timezone

from ghostaudit.ingest.manifest import CitationRegistry, Manifest


def test_manifest_loads_every_file():
    m = Manifest.load()
    assert len(m.files) == 94
    assert all(f.path.is_file() for f in m.files)
    assert [f.file_id for f in m.files] == sorted(f.file_id for f in m.files)


def test_eligibility_is_inclusive():
    m = Manifest.load()
    f = m.files[0]
    assert f.eligible(f.captured_at)
    assert not f.eligible(f.captured_at - timedelta(seconds=1))


def test_registry_covers_all_citable_sources():
    r = CitationRegistry.load()
    assert r.known("evidence_01") and r.known("activity_01") and r.known("audit_01")
    assert not r.citable_at("evidence_01", datetime(2000, 1, 1, tzinfo=timezone.utc))
