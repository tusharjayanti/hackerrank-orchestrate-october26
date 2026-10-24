from ghostaudit.ingest.manifest import Manifest
from ghostaudit.ingest import parse_logs, parse_records, parse_sql
from ghostaudit.schemas import DataCategory, ObservedState

MANIFEST = Manifest.load()


def obs_for(parser, file_id):
    return parser.parse(MANIFEST.by_id[file_id])


def test_tokenizer_handles_every_syntax():
    lines = [
        '2026-07-28T05:51:21Z [INFO] [account_service] [tr-1] profile_lookup_completed: resource_id="account_1" rows="0"',
        '2026-07-03T07:51:32Z | DEBUG | security_service | tr-2 | audit_record_read_completed | resource_id="audit_record_1"',
        '2026-03-25T08:51:23Z <12>1 node-9 support_platform tr-3 - - level=WARN resource_id="support_1" event="message_pii_scan"',
        '2026-04-19T06:51:13Z {"severity":"DEBUG","service":"search_service","trace_id":"tr-4","event":"index_query_started"}',
        '2026-09-30T11:51:03Z requestId=tr-5 level=DEBUG function=profile_service message="resource_id=\\"account_2\\" event=\\"profile_lookup_completed\\""',
        '2026-09-15T08:51:45Z level=DEBUG service=search_service trace_id=tr-6 event=index_query_started match_key="a@b.com"',
        '2026-03-15T07:00:00Z INFO service=x event=record_lookup system="account_service" resource_id="account_3" result="FOUND"',
    ]
    traces = []
    for line in lines:
        ts, d = parse_logs.tokenize(line)
        assert d.get("event"), line
        traces.append(d.get("trace"))
    assert traces[:6] == ["tr-1", "tr-2", "tr-3", "tr-4", "tr-5", "tr-6"]


def test_log_trace_join_links_completed_line_to_started_identifier():
    # support_replica.log: lookup_key on *_started, resource on *_completed.
    res = obs_for(parse_logs, "evidence_61")
    support_002 = [o for o in res.observations if o.resource_id == "support_002"]
    assert support_002 and all("nadia.ionescu@yahoo.com" in o.identifiers for o in support_002)


def test_sql_replica_drift_keeps_logical_system_and_table_detail():
    res = obs_for(parse_sql, "evidence_54")
    a5 = [o for o in res.observations if o.resource_id == "account_005"]
    assert [(o.system_id, o.source_detail, o.state) for o in a5] == [
        ("account_service", "sql.customer_profiles_failover_replica", ObservedState.PRESENT)]
    primary = next(c for c in res.coverage if c.source_detail == "sql.customer_profiles_primary")
    assert "account_005" not in primary.resource_ids


def test_tombstone_states():
    res = obs_for(parse_records, "evidence_51")
    by_rid = {o.resource_id: o for o in res.observations}
    assert by_rid["tombstone_440"].data_category is DataCategory.ACCOUNT_TOMBSTONE
    assert by_rid["tombstone_440"].state is ObservedState.REDACTED  # subject_reference only
    assert by_rid["tombstone_9178"].state is ObservedState.IDENTIFIABLE  # name/email/phone/hash remain


def test_support_record_with_only_reference_is_redacted():
    res = obs_for(parse_records, "evidence_92")
    s = next(o for o in res.observations if o.resource_id == "support_4715")
    assert s.state is ObservedState.REDACTED and s.identifiers == ("user_26",)


def test_search_query_does_not_link_a_decoy_hit():
    res = obs_for(parse_sql, "evidence_57")
    by_rid = {o.resource_id: o for o in res.observations}
    assert "jonas.berg@outlook.com" in by_rid["search_001"].identifiers
    assert "jonas.berg@outlook.com" not in by_rid["search_8814"].identifiers


def test_restore_and_connector_heartbeat_states():
    res = obs_for(parse_logs, "evidence_01")
    by_rid = {o.resource_id: o for o in res.observations}
    assert by_rid["backup_3245"].state is ObservedState.ACTIVE
    assert by_rid["data_flow_9846"].state is ObservedState.ACTIVE
