"""Tests for src/matcher.py. The LLM is always faked - no network calls."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import matcher as mt

# Real text of ground-truth pair 22V680000-170 (Mercedes GLE/GLS rear door trim bar).
US = ("Component: STRUCTURE:BODY\n"
      "Defect: Mercedes-Benz USA, LLC (MBUSA) is recalling certain 2020-2022 GLE350, GLE450, "
      "GLS450 vehicles. Due to a manufacturing error, the rear door window trim bars may detach.")
INDIA = ("Model: GLE and GLS (167)\n"
         "Defect: In the process, we have identified that a fixation of the trim bar at the rear "
         "door windows unfortunately might not meet specifications. Thus the trim bar might "
         "separate from the vehicle while driving.")
PAIR = {"pair_id": "22V680000-170", "CAMPNO": "22V680000", "make": "MERCEDES-BENZ",
        "csv_row": 170, "us_text": US, "india_text": INDIA,
        "us_has_defect": True, "india_has_defect": True}


def _result(**kw):
    base = {"pair_id": "22V680000-170", "label": "same", "confidence": 0.9,
            "us_evidence": "the rear door window trim bars may detach",
            "india_evidence": "the trim bar might separate from the vehicle"}
    base.update(kw)
    return base


# ---------- the evidence check (requested: one pass, one fail) ----------

def test_evidence_check_passes_for_exact_phrases():
    # different case and a line break inside the phrase still count as the same text
    out = mt.check_result(_result(us_evidence="The Rear Door Window\n trim bars may DETACH"), PAIR)
    assert out["us_evidence_found"] and out["india_evidence_found"]
    assert out["label"] == "same" and out["review_reason"] is None


def test_evidence_check_fails_for_a_paraphrase():
    out = mt.check_result(_result(india_evidence="the trim may fall off"), PAIR)
    assert out["us_evidence_found"] and not out["india_evidence_found"]
    assert out["label"] == "needs_review"
    assert "india_evidence not found in INDIA text" in out["review_reason"]
    assert out["llm_label"] == "same"            # the LLM's answer is kept for review


# ---------- evidence edge cases ----------

def test_typographic_apostrophe_matches_plain():
    assert mt.evidence_found("the driver's airbag", "Inflator in the driver’s airbag may rupture")


def test_evidence_must_be_three_words():
    out = mt.check_result(_result(us_evidence="trim bars"), PAIR)
    assert out["us_evidence_found"] and out["label"] == "needs_review"
    assert "shorter than 3 words" in out["review_reason"]


def test_empty_evidence_is_not_found():
    assert not mt.evidence_found("", US)
    assert not mt.evidence_found(None, US)


@pytest.mark.parametrize("change, reason", [
    ({"label": "maybe"}, "invalid label"),
    ({"confidence": 1.4}, "invalid confidence"),
    ({"confidence": "high"}, "invalid confidence"),
    ({"confidence": True}, "invalid confidence"),
])
def test_bad_label_or_confidence_needs_review(change, reason):
    out = mt.check_result(_result(**change), PAIR)
    assert out["label"] == "needs_review" and reason in out["review_reason"]


def test_missing_result_needs_review():
    out = mt.check_result(None, PAIR)
    assert out["label"] == "needs_review" and "no result" in out["review_reason"]


def test_normalise():
    assert mt.normalise("  A–B\n\tC”D ") == 'a-b c"d'


# ---------- texts / prompt ----------

def test_texts_include_model_and_skip_empty_parts():
    row = pd.Series({"components": ["AIR BAGS", "SEATS"], "DESC_DEFECT": "Trim may detach.",
                     "CONEQUENCE_DEFECT": None, "model": "S-Class (223)",
                     "defect_description": "Trim bar may separate."})
    assert mt.us_text(row) == "Component: AIR BAGS; SEATS\nDefect: Trim may detach."
    assert mt.india_text(row) == "Model: S-Class (223)\nDefect: Trim bar may separate."


def test_prompt_contains_each_pair_and_rules():
    p = mt.build_prompt([PAIR])
    assert "=== PAIR 22V680000-170 (make MERCEDES-BENZ) ===" in p
    assert US in p and INDIA in p and "A different model-year range is fine" in p


# ---------- judge: batching, missing/unknown ids, empty text ----------

def _pairs(n):
    return pd.DataFrame([{**PAIR, "pair_id": f"C{i}-{i}", "CAMPNO": f"C{i}", "csv_row": i}
                         for i in range(n)])


def test_judge_batches_ten_per_call_at_temperature_zero():
    calls = []

    def fake(prompt, **kw):
        calls.append(kw)
        ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("=== PAIR")]
        return {"results": [_result(pair_id=i) for i in ids]}

    out = mt.judge(_pairs(25), ask=fake)
    assert len(calls) == 3 and all(kw["temperature"] == 0.0 for kw in calls)
    assert len(out) == 25 and set(out.label) == {"same"}


def test_judge_handles_missing_and_unknown_pair_ids(caplog):
    def fake(prompt, **kw):
        return {"results": [_result(pair_id="C0-0"), _result(pair_id="ZZZ-9")]}

    out = mt.judge(_pairs(2), ask=fake).set_index("pair_id")
    assert out.loc["C0-0", "label"] == "same"
    assert out.loc["C1-1", "label"] == "needs_review"      # LLM skipped it
    assert "ZZZ-9" in caplog.text                          # unknown id is logged, not used


def test_judge_does_not_send_pairs_without_defect_text():
    calls = []
    pairs = _pairs(1).assign(india_has_defect=False)
    out = mt.judge(pairs, ask=lambda p, **kw: calls.append(p) or {"results": []})
    assert calls == [] and out.label.iloc[0] == "needs_review"
    assert "no INDIA defect text" in out.review_reason.iloc[0]


# ---------- ground-truth comparison ----------

def test_compare_ground_truth_counts_and_suppresses_small_n():
    gt = pd.DataFrame({"pair_id": ["a", "b", "c", "d"], "label": ["same", "different",
                                                                   "different", "same"]})
    matches = pd.DataFrame({"pair_id": ["a", "b", "c", "d"],
                            "label": ["same", "different", "same", "needs_review"],
                            "review_reason": [None, None, None, "us_evidence not found"]})
    cmp = mt.compare_ground_truth(matches, gt)
    assert (cmp["judged"], cmp["agree"], cmp["needs_review"]) == (3, 2, 1)
    assert cmp["per_class"] == {"different": (1, 2), "same": (1, 1)}
    assert "rate suppressed" in mt.report(matches, cmp)


# ---------- build + main ----------

def test_build_writes_matches_table(tmp_path: Path):
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE india_relevant_campaigns AS SELECT '22V680000' AS CAMPNO,
        'MERCEDES-BENZ' AS make, ['STRUCTURE:BODY'] AS components,
        'Due to a manufacturing error, the rear door window trim bars may detach.' AS DESC_DEFECT,
        NULL AS CONEQUENCE_DEFECT""")
    con.execute("""CREATE TABLE siam_recalls AS SELECT 170 AS csv_row, 'MERCEDES-BENZ' AS make,
        'GLE and GLS (167)' AS model,
        'Thus the trim bar might separate from the vehicle while driving.' AS defect_description""")
    con.close()
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"pair_id": ["22V680000-170"], "label": ["same"]}).to_csv(gt, index=False)

    out = mt.build(db_path=db, gt_csv=gt, ask=lambda p, **kw: {"results": [_result()]})
    con = duckdb.connect(str(db), read_only=True)
    label, scope, found = con.execute(
        "SELECT label, scope, india_evidence_found FROM matches").fetchone()
    assert (label, scope, found) == ("same", "ground_truth", True)
    assert out["comparison"]["agree"] == 1
    cols = [c[0] for c in con.execute("DESCRIBE matches").fetchall()]
    assert {"fingerprint", "prompt_version"} <= set(cols)
    con.close()
    # a rerun reuses the stored verdict: the LLM is not asked again
    again = mt.build(db_path=db, gt_csv=gt, ask=lambda p, **kw: pytest.fail("must not ask"))
    assert (again["reused"], again["sent"]) == (1, 0)


def test_build_rejects_unknown_scope(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown scope"):
        mt.build(db_path=tmp_path / "x.duckdb", gt_csv=tmp_path / "none.csv", scope="some")


def test_main_runs_ground_truth_scope_by_default(monkeypatch, capsys):
    seen = {}

    def fake_build(scope):
        seen["scope"] = scope
        return {"matches": pd.DataFrame({"label": ["same"]}), "comparison": None,
                "reused": 0, "sent": 1}

    monkeypatch.setattr(mt, "build", fake_build)
    mt.main([])
    out = capsys.readouterr().out
    assert seen["scope"] == "ground_truth" and "same=1" in out
    assert "pairs sent to the LLM: 1" in out


# ---------- fingerprints and reuse ----------

from src.fingerprint import fingerprint  # noqa: E402


def test_prompt_rules_are_unchanged_text():
    # the rules moved into a constant; the prompt must stay byte-identical so the LLM cache hits
    assert mt.build_prompt([PAIR]).startswith(mt.PROMPT_RULES + "\n")
    assert mt.PROMPT_VERSION == fingerprint(mt.SYSTEM, mt.PROMPT_RULES)


def test_check_result_carries_the_pair_fingerprint():
    out = mt.check_result(_result(), {**PAIR, "fingerprint": "abc"})
    assert out["fingerprint"] == "abc"


def _fp_pairs(n):
    p = _pairs(n)
    p["fingerprint"] = [fingerprint(u, i) for u, i in zip(p.us_text, p.india_text)]
    return p


def _echo_fake(calls):
    def fake(prompt, **kw):
        ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("=== PAIR")]
        calls.append(ids)
        return {"results": [_result(pair_id=i) for i in ids]}
    return fake


def test_unchanged_pairs_are_reused_without_a_call():
    first = mt.judge(_fp_pairs(3), ask=_echo_fake([]))
    calls = []
    again = mt.judge(_fp_pairs(3), ask=_echo_fake(calls), previous=first)
    assert calls == [] and again.attrs == {"reused": 3, "sent": 0}
    assert again.label.tolist() == first.label.tolist()


def test_changed_text_or_prompt_version_is_sent_again():
    first = mt.judge(_fp_pairs(3), ask=_echo_fake([]))
    changed = _fp_pairs(3)
    changed.loc[0, "fingerprint"] = "different"
    calls = []
    mt.judge(changed, ask=_echo_fake(calls), previous=first)
    assert calls == [["C0-0"]]
    calls = []
    mt.judge(_fp_pairs(3), ask=_echo_fake(calls), previous=first.assign(prompt_version="old"))
    assert calls == [["C0-0", "C1-1", "C2-2"]]


def test_previous_without_fingerprints_reuses_nothing():
    calls = []
    out = mt.judge(_fp_pairs(2), ask=_echo_fake(calls),
                   previous=pd.DataFrame({"pair_id": ["C0-0"], "label": ["same"]}))
    assert len(calls) == 1 and out.attrs == {"reused": 0, "sent": 2}
