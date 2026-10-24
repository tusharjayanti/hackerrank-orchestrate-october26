import json
from types import SimpleNamespace

import pytest

from ghostaudit import llm as llm_mod
from ghostaudit.llm import LLM, LLMExtractionFailed


class FakeMessages:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def create(self, **_):
        self.calls += 1
        text = self.texts.pop(0)
        return SimpleNamespace(
            model="claude-sonnet-5-5", stop_reason="end_turn", _request_id="req_test",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            content=[SimpleNamespace(type="text", text=text)])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm_mod, "CALL_LOG", tmp_path / "calls.jsonl")

    def make(texts):
        model = LLM()
        model._client = SimpleNamespace(messages=FakeMessages(texts))
        return model
    return make


def call(model, tag="x"):
    return model.json_call(purpose=tag, prompt_version="t", system="s", content=[{"type": "text", "text": tag}],
                           schema={"type": "object"})


def test_malformed_json_is_retried_once(client):
    model = client(["not json", json.dumps({"ok": 1})])
    assert call(model) == {"ok": 1}
    assert model._client.messages.calls == 2


def test_malformed_json_twice_raises_extraction_failed(client):
    model = client(["nope", "still nope"])
    with pytest.raises(LLMExtractionFailed):
        call(model)


def test_answers_are_cached(client):
    model = client([json.dumps({"ok": 1})])
    assert call(model) == {"ok": 1}
    assert call(model) == {"ok": 1}
    assert model._client.messages.calls == 1
