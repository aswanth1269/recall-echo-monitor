"""Tests for the LLM helper that need no API key and make no network calls.

The cache and the JSON salvaging are the parts most likely to break quietly
and the parts that protect a small daily quota, so they are worth pinning.
"""

from __future__ import annotations

import json

import pytest

from src import llm


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def test_missing_key_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "replace-me")
    with pytest.raises(llm.LLMNotConfigured, match="aistudio.google.com"):
        llm._client()


def test_cache_prevents_a_second_call(isolated_cache, monkeypatch):
    """A repeat prompt must cost zero requests - the whole point on a free tier."""
    calls = []

    def fake(model, messages, max_tokens, temperature):
        calls.append(messages)
        return "cached answer"

    monkeypatch.setattr(llm, "_call_with_backoff", fake)

    first = llm.complete("same question")
    second = llm.complete("same question")

    assert first == second == "cached answer"
    assert len(calls) == 1, "second identical prompt should hit the disk cache"
    assert len(list(isolated_cache.glob("*.json"))) == 1


def test_cache_key_separates_different_prompts(isolated_cache, monkeypatch):
    monkeypatch.setattr(llm, "_call_with_backoff",
                        lambda model, messages, mt, t: messages[-1]["content"])
    assert llm.complete("question A") == "question A"
    assert llm.complete("question B") == "question B"
    assert len(list(isolated_cache.glob("*.json"))) == 2


def test_use_cache_false_skips_the_cache(isolated_cache, monkeypatch):
    monkeypatch.setattr(llm, "_call_with_backoff",
                        lambda model, messages, mt, t: "fresh")
    llm.complete("q", use_cache=False)
    assert not list(isolated_cache.glob("*.json"))


def test_system_prompt_is_sent_first(isolated_cache, monkeypatch):
    seen = {}

    def fake(model, messages, max_tokens, temperature):
        seen["messages"] = messages
        return "ok"

    monkeypatch.setattr(llm, "_call_with_backoff", fake)
    llm.complete("hello", system="be terse")
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert seen["messages"][0]["content"] == "be terse"


@pytest.mark.parametrize("reply, expected", [
    ('{"component": "airbag"}', {"component": "airbag"}),
    ('```json\n{"component": "airbag"}\n```', {"component": "airbag"}),
    ('```\n{"component": "airbag"}\n```', {"component": "airbag"}),
    ('Sure! Here you go:\n{"component": "airbag"}\nHope that helps.',
     {"component": "airbag"}),
    ('[{"a": 1}, {"a": 2}]', [{"a": 1}, {"a": 2}]),
])
def test_complete_json_survives_chatty_models(reply, expected, monkeypatch):
    """Cheap models fence their JSON and add prose; we must cope."""
    monkeypatch.setattr(llm, "complete", lambda prompt, **kw: reply)
    assert llm.complete_json("irrelevant") == expected


def test_complete_json_reports_unusable_replies(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda prompt, **kw: "no json here")
    with pytest.raises(ValueError, match="No JSON found"):
        llm.complete_json("irrelevant")


def test_batched_groups_and_keeps_the_remainder():
    assert list(llm.batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(llm.batched([], 3)) == []
    assert list(llm.batched([1], 10)) == [[1]]


def test_batched_rejects_a_bad_size():
    with pytest.raises(ValueError):
        list(llm.batched([1, 2], 0))


def test_batched_cuts_real_recall_volume_to_few_requests():
    """355 SIAM rows must not become 355 requests against a ~500/day cap."""
    assert len(list(llm.batched(range(355), 20))) == 18


def test_retry_after_header_is_honoured():
    class Resp:
        headers = {"retry-after": "7.5"}

    class Exc(Exception):
        response = Resp()

    assert llm._retry_after(Exc()) == 7.5
    assert llm._retry_after(Exception()) is None


def test_cache_file_is_readable_json(isolated_cache, monkeypatch):
    monkeypatch.setattr(llm, "_call_with_backoff",
                        lambda model, messages, mt, t: "answer")
    llm.complete("q")
    (path,) = isolated_cache.glob("*.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["text"] == "answer"
    assert saved["prompt"] == "q"
