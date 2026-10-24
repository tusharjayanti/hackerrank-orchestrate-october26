from ghostaudit.ingest import parse_vision
from ghostaudit.ingest.activity import Activity
from ghostaudit.ingest.manifest import Manifest
from ghostaudit.llm import model_for
from ghostaudit.schemas import DataCategory, ObservedState

MANIFEST = Manifest.load()
ACTIVITY = Activity.load()
UNIVERSE = parse_vision.Universe({e.resource_id for e in ACTIVITY.events},
                                 {lf.label: lf.resource_id for lf in ACTIVITY.links if lf.label != lf.resource_id})


def test_pdf_attachment_is_a_remaining_record_with_subject_identifiers():
    res = parse_vision.parse(MANIFEST.by_id["evidence_58"], UNIVERSE, None)
    [o] = res.observations
    assert (o.resource_id, o.data_category, o.state) == ("support_001", DataCategory.SUPPORT_ATTACHMENT,
                                                         ObservedState.PRESENT)
    assert "priya.raman@gmail.com" in o.identifiers and o.vision


def test_pdf_catalog_label_resolves_resource():
    [o] = parse_vision.parse(MANIFEST.by_id["evidence_18"], UNIVERSE, None).observations
    assert o.resource_id == "export_1842" and "user_18" in o.identifiers


def test_pdf_user_reference_label_never_resolves_and_registers_attach_nothing():
    res = parse_vision.parse(MANIFEST.by_id["evidence_27"], UNIVERSE, None)
    assert res.observations == []


class FakeLLM:
    def __init__(self, a, b, t=None):
        self.answers = {"A": a, "B": b, "T": t or []}
        self.models = []

    def json_call(self, *, prompt_version, model, **_):
        self.models.append(model)
        return {"records": self.answers[prompt_version.rsplit(":", 1)[1]]}


def row(rid, **kw):
    base = {"resource_id": rid, "status_text": "", "not_found_or_deleted": False, "enabled": None,
            "restore_allowed": None, "emails": [], "phones": [], "names": [], "email_hashes": [],
            "user_references": []}
    base.update(kw)
    return base


PNG = MANIFEST.by_id["evidence_19"]


def test_png_agreement_keeps_row_without_escalation():
    llm = FakeLLM([row("analytics_7438", emails=["a@b.com"])], [row("analytics_7438", emails=["A@B.com"])])
    [o] = parse_vision.parse(PNG, UNIVERSE, llm).observations
    assert o.source_detail == "png:agreed" and o.state is ObservedState.IDENTIFIABLE
    assert llm.models == [model_for("VISION"), model_for("VISION")]


def test_png_disagreement_is_settled_by_one_opus_read():
    llm = FakeLLM([row("analytics_7438")], [row("analytics_7438", emails=["a@b.com"])], [row("analytics_7438")])
    [o] = parse_vision.parse(PNG, UNIVERSE, llm).observations
    assert o.source_detail == "png:tiebreak" and o.state is ObservedState.ANONYMIZED
    assert llm.models == [model_for("VISION"), model_for("VISION"), model_for("AGENT")]


def test_png_three_way_disagreement_keeps_worse_reading_and_flags():
    llm = FakeLLM([row("analytics_7438")], [row("analytics_7438", emails=["a@b.com"])],
                  [row("analytics_7438", user_references=["user_20"])])
    [o] = parse_vision.parse(PNG, UNIVERSE, llm).observations
    assert o.source_detail == "png:disagreed" and o.state is ObservedState.IDENTIFIABLE


def test_png_unknown_but_agreed_id_is_admitted_as_vision_only():
    llm = FakeLLM([row("analytics_99999999")], [row("analytics_99999999")])
    [o] = parse_vision.parse(PNG, UNIVERSE, llm).observations
    assert o.source_detail == "png:agreed:vision_only"


def test_png_unknown_and_unagreed_id_is_dropped():
    llm = FakeLLM([row("analytics_99999999")], [row("analytics_99999999", emails=["a@b.com"])],
                  [row("analytics_99999999", user_references=["user_1"])])
    assert parse_vision.parse(PNG, UNIVERSE, llm).observations == []


def test_png_value_that_is_not_a_resource_id_is_dropped():
    llm = FakeLLM([row("user_17")], [row("user_17")])
    res = parse_vision.parse(PNG, UNIVERSE, llm)
    assert res.observations == [] and any("non-resource" in n for n in res.notes)
