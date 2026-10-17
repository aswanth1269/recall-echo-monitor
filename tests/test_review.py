"""Tests for src/review.py. The LLM is never called; the database is a small fake."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import review as rv

PAIR_ITEM = {"kind": "pair", "side": "", "make": "HONDA", "key": "20V001000-1",
             "fingerprint": "aaaaaaaaaaaaaaaa"}
FAMILY_ITEM = {"kind": "family", "side": "nhtsa", "make": "HONDA", "key": "FIT",
               "fingerprint": "bbbbbbbbbbbbbbbb"}
VOCAB_ITEM = {"kind": "vocabulary", "side": "", "make": "HONDA", "key": "CITY HYBRID",
              "fingerprint": "cccccccccccccccc"}


@pytest.fixture
def ref(tmp_path: Path) -> Path:
    d = tmp_path / "reference"
    d.mkdir()
    pd.DataFrame({"make": ["HONDA"] * 3, "family": ["AMAZE", "CITY", "CITY HYBRID"],
                  "source": ["india_models"] * 3}).to_csv(d / "model_families.csv", index=False)
    return d


def _files(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# ---------- record / undo / outstanding ----------

def test_record_pair_writes_one_row_with_fingerprint_and_date(ref):
    changes = rv.record(PAIR_ITEM, "same", "checked both texts", ref_dir=ref, today="2026-10-16")
    df = rv.read_decisions("pair", ref)
    assert df.to_dict("records") == [{"pair_id": "20V001000-1", "decision": "same",
                                      "note": "checked both texts",
                                      "fingerprint": "aaaaaaaaaaaaaaaa",
                                      "reviewed_at": "2026-10-16"}]
    assert changes == [{"kind": "pair", "keys": {"pair_id": "20V001000-1"}, "previous": None}]


def test_second_decision_replaces_the_first(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref, today="2026-10-16")
    changes = rv.record(PAIR_ITEM, "different", "on second look", ref_dir=ref, today="2026-10-17")
    df = rv.read_decisions("pair", ref)
    assert df.decision.tolist() == ["different"]
    assert changes[0]["previous"]["decision"] == "same"


@pytest.mark.parametrize("item, decision, message", [
    (PAIR_ITEM, "maybe", "same or different"),
    (FAMILY_ITEM, "CIVIC", "not in the HONDA vocabulary"),
    (VOCAB_ITEM, "merge", "distinct or alias"),
    ({**PAIR_ITEM, "kind": "nonsense"}, "same", "unknown item kind"),
    ({**PAIR_ITEM, "fingerprint": ""}, "same", "no fingerprint"),
])
def test_invalid_decisions_are_refused_and_nothing_is_written(ref, item, decision, message):
    before = _files(ref)
    with pytest.raises(ValueError, match=message):
        rv.record(item, decision, ref_dir=ref)
    assert _files(ref) == before


def test_family_decision_uses_vocabulary_spelling_or_none(ref):
    rv.record(FAMILY_ITEM, "city", ref_dir=ref, today="2026-10-16")
    rv.record({**FAMILY_ITEM, "key": "PILOT"}, "None", ref_dir=ref, today="2026-10-16")
    df = rv.read_decisions("family", ref)
    assert df[["side", "make", "key", "family"]].values.tolist() == [
        ["nhtsa", "HONDA", "FIT", "CITY"], ["nhtsa", "HONDA", "PILOT", "none"]]
    assert (df.fingerprint == "bbbbbbbbbbbbbbbb").all()


def test_alias_needs_a_source_and_writes_both_files(ref):
    with pytest.raises(ValueError, match="source_url"):
        rv.record(VOCAB_ITEM, "alias", from_name="CITY E:HEV", ref_dir=ref)
    with pytest.raises(ValueError, match="retrieved_on"):
        rv.record(VOCAB_ITEM, "alias", from_name="CITY E:HEV",
                  source_url="https://example.com/city", retrieved_on="16/10/2026", ref_dir=ref)
    changes = rv.record(VOCAB_ITEM, "alias", "same car", from_name="CITY E:HEV",
                        source_url="https://example.com/city", retrieved_on="2026-10-16",
                        ref_dir=ref, today="2026-10-16")
    assert [c["kind"] for c in changes] == ["alias", "vocabulary"]
    alias = pd.read_csv(ref / "model_aliases.csv", dtype=str)
    assert alias[["make", "from_name", "family"]].values.tolist() == [
        ["HONDA", "CITY E:HEV", "CITY HYBRID"]]
    assert rv.read_decisions("vocabulary", ref).decision.tolist() == ["alias"]


def test_distinct_writes_only_the_vocabulary_review_file(ref):
    rv.record(VOCAB_ITEM, "distinct", ref_dir=ref, today="2026-10-16")
    assert not (ref / "model_aliases.csv").exists()
    assert rv.read_decisions("vocabulary", ref).decision.tolist() == ["distinct"]


def test_note_is_made_formula_safe(ref):
    rv.record(PAIR_ITEM, "same", '=HYPERLINK("http://x")', ref_dir=ref)
    assert rv.read_decisions("pair", ref).note.iloc[0].startswith("'=")


def test_record_only_writes_the_decision_files(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    rv.record(FAMILY_ITEM, "CITY", ref_dir=ref)
    rv.record(VOCAB_ITEM, "alias", from_name="X", source_url="https://e.com",
              retrieved_on="2026-10-16", ref_dir=ref)
    assert _files(ref) == {"model_families.csv", *rv.FILES.values()}   # no .tmp left behind


def test_record_keeps_rows_added_by_hand_meanwhile(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    path = ref / rv.FILES["pair"]
    path.write_text(path.read_text(encoding="utf-8")
                    + "20V009000-9,different,by hand,,2026-10-16\n", encoding="utf-8")
    rv.record({**PAIR_ITEM, "key": "20V002000-2"}, "same", ref_dir=ref)
    assert set(rv.read_decisions("pair", ref).pair_id) == {
        "20V001000-1", "20V009000-9", "20V002000-2"}


def test_undo_restores_the_previous_state(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref, today="2026-10-16")
    changes = rv.record(PAIR_ITEM, "different", ref_dir=ref, today="2026-10-17")
    rv.undo(changes, ref)
    assert rv.read_decisions("pair", ref).decision.tolist() == ["same"]
    first = rv.record({**PAIR_ITEM, "key": "20V002000-2"}, "same", ref_dir=ref)
    rv.undo(first, ref)
    assert rv.read_decisions("pair", ref).pair_id.tolist() == ["20V001000-1"]


def test_outstanding_drops_items_decided_since_the_queue_was_written(ref):
    q = pd.DataFrame([{**PAIR_ITEM, "status": "open"},
                      {**PAIR_ITEM, "key": "20V002000-2", "status": "open"},
                      {**FAMILY_ITEM, "status": "stale"},
                      {**VOCAB_ITEM, "status": "reviewed"}])
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    out = rv.outstanding(q, ref)
    assert out[["kind", "key"]].values.tolist() == [["pair", "20V002000-2"], ["family", "FIT"]]
