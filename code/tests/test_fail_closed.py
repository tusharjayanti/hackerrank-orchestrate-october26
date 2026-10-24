"""An unknown connector or restore flag is the exposed reading, never the
compliant one. case_39 is the real case that shipped a false COMPLETE."""

from dataclasses import replace

from ghostaudit.engine.decide import World, decide_case
from ghostaudit.ingest.observations import ObservationTable, observe_record, state_for
from ghostaudit.schemas import DataCategory, ObservedState

WORLD = replace(World.load(), llm=None)


def test_missing_flags_read_as_exposed():
    assert state_for(DataCategory.DATA_FLOW, {"resource_id": "data_flow_1"}, False) is ObservedState.ACTIVE
    assert state_for(DataCategory.BACKUP_ARCHIVE, {"resource_id": "backup_1"}, False) is ObservedState.ACTIVE
    assert state_for(DataCategory.DATA_FLOW, {"enabled": False}, False) is ObservedState.DISABLED
    assert state_for(DataCategory.BACKUP_ARCHIVE, {"restore_allowed": False}, False) is ObservedState.BLOCKED


def test_case_39_running_connector_is_no_longer_a_complete_row():
    # evidence_14 (connector console) shows data_flow_1121 for the subject's
    # email with status "Healthy" and no enabled/disabled flag; both vision
    # readings returned enabled=null. Rebuild that row as the parser does.
    f = WORLD.manifest.by_id["evidence_14"]
    o = observe_record(f, f.captured_at, {"resource_id": "data_flow_1121", "customer_email": "jain.aditi@gmail.com"},
                       source_detail="png:agreed")
    assert o.state is ObservedState.ACTIVE and o.field_defaulted
    o = o.model_copy(update={"vision": True})
    t = WORLD.table
    world = replace(WORLD, table=ObservationTable(t.observations + [o], t.coverage, t.stats, t.identity))
    decision, trace = decide_case(world, world.requests["case_39"], use_agent=False)
    assert decision.status.value == "INCOMPLETE"
    finding = [s for s in decision.steps if s.phase.value == "FINDING"]
    assert [(s.operation.value, s.resource_id, s.evidence_id) for s in finding] == [
        ("REPORT_ACTIVE_DATA_FLOW", "data_flow_1121", "evidence_14")]
    assert "VISION_FIELD_DEFAULTED" in decision.fired()
