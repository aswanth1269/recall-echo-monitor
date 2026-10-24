"""Tests for src/evaluate.py (synthetic database; never touches real labels)."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import evaluate as ev


# ---------- confusion / metrics / fmt_rate ----------

def test_confusion_and_metrics():
    truth = pd.Series(["same", "same", "different", "different", "same"])
    pred = pd.Series(["same", "different", "same", "different", "same"])
    c = ev.confusion(truth, pred)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    m = ev.metrics(c)
    assert m["precision"] == pytest.approx((2 / 3, 2, 3))
    assert m["recall"] == pytest.approx((2 / 3, 2, 3))
    assert m["f1"][1:] == (4, 6)


def test_metrics_undefined_when_nothing_predicted():
    m = ev.metrics({"tp": 0, "fp": 0, "fn": 3, "tn": 5})
    assert math.isnan(m["precision"][0]) and m["recall"][1:] == (0, 3)


@pytest.mark.parametrize("args, text", [
    ((0.75, 3, 4), "3 / 4 (n=4 < 10, rate suppressed)"),
    ((27 / 29, 27, 29), "27 / 29 = 0.931 (n=29)"),
    ((float("nan"), 0, 0), "undefined (n=0)"),
])
def test_fmt_rate_shows_n_and_suppresses_small_n(args, text):
    assert ev.fmt_rate(*args) == text


# ---------- choose_threshold ----------

def test_choose_threshold_maximises_f1():
    scores = pd.Series([0.9, 0.5, 0.4, 0.1, 0.05])
    truth = pd.Series(["same", "same", "different", "different", "different"])
    t, f1 = ev.choose_threshold(scores, truth)
    assert t == 0.5 and f1 == 1.0


def test_choose_threshold_ties_go_to_higher_threshold():
    scores = pd.Series([0.9, 0.2, 0.1])
    truth = pd.Series(["same", "different", "different"])
    assert ev.choose_threshold(scores, truth)[0] == 0.9


# ---------- load_ground_truth / llm_predictions ----------

def test_load_ground_truth_parses_ids_and_rejects_other_labels(tmp_path: Path):
    p = tmp_path / "gt.csv"
    pd.DataFrame({"pair_id": ["20V543000-81"], "label": ["same"]}).to_csv(p, index=False)
    gt = ev.load_ground_truth(p)
    assert (gt.CAMPNO.iloc[0], gt.csv_row.iloc[0]) == ("20V543000", 81)
    pd.DataFrame({"pair_id": ["x-1"], "label": ["maybe"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="other than same/different"):
        ev.load_ground_truth(p)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE matches AS SELECT * FROM (VALUES
        ('A-1', 'same',         0.9, 'trim bars may detach',  'trim bar might separate', NULL),
        ('B-2', 'different',    0.8, 'brake hose may leak',   'airbag may rupture',      NULL),
        ('C-3', 'needs_review', 0.9, 'invented quote here',   'inflator may rupture',
              'us_evidence not found in US text'),
        ('D-4', 'same',         0.7, 'steering may shut off', 'power steering deactivated', NULL))
        AS t(pair_id, label, confidence, us_evidence, india_evidence, review_reason)""")
    con.execute("""CREATE TABLE candidate_pairs AS SELECT * FROM (VALUES
        ('A', 1, 0.40), ('B', 2, 0.02), ('C', 3, 0.30), ('D', 4, 0.05))
        AS t(CAMPNO, csv_row, score)""")
    con.execute("""CREATE TABLE india_relevant_campaigns AS SELECT * FROM (VALUES
        ('A', 'MB', ['BODY'], 'The rear door window trim bars may detach.', NULL),
        ('B', 'MB', ['BRAKES'], 'The brake hose may leak.', NULL),
        ('C', 'HONDA', ['AIR BAGS'], 'Inflators may explode.', NULL),
        ('D', 'MB', ['STEERING'], 'Power steering may shut off.', NULL))
        AS t(CAMPNO, make, components, DESC_DEFECT, CONEQUENCE_DEFECT)""")
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'MB', 'GLE', 'The trim bar might separate.'),
        (2, 'MB', 'GLE', 'The airbag may rupture.'),
        (3, 'HONDA', 'Civic', 'The inflator may rupture.'),
        (4, 'MB', 'C-Class', 'Power steering deactivated.'))
        AS t(csv_row, make, model, defect_description)""")
    con.close()
    return db


@pytest.fixture
def gt_file(tmp_path: Path) -> Path:
    p = tmp_path / "ground_truth.csv"
    pd.DataFrame({"pair_id": ["A-1", "B-2", "C-3", "D-4"],
                  "label": ["same", "different", "same", "different"],
                  "reason": ["trim bar", "brake vs airbag", "inflator", "different platform"],
                  "confidence": ["high", "high", "high", "low"],
                  "labelled_by": "test labeller"}).to_csv(p, index=False)
    return p


def test_llm_predictions_count_needs_review_as_different(tmp_path, gt_file):
    con = duckdb.connect(str(_db(tmp_path)), read_only=True)
    p = ev.llm_predictions(con, ev.load_ground_truth(gt_file)).set_index("pair_id")
    assert p.loc["C-3", "llm_raw"] == "needs_review" and p.loc["C-3", "llm_pred"] == "different"


def test_llm_predictions_require_every_ground_truth_pair(tmp_path, gt_file):
    con = duckdb.connect(str(_db(tmp_path)))
    con.execute("DELETE FROM matches WHERE pair_id = 'D-4'")
    with pytest.raises(ValueError, match="run python -m src.matcher"):
        ev.llm_predictions(con, ev.load_ground_truth(gt_file))


def test_baseline_scores_join_candidate_pairs(tmp_path, gt_file):
    con = duckdb.connect(str(_db(tmp_path)), read_only=True)
    s = ev.baseline_scores(con, ev.load_ground_truth(gt_file)).set_index("pair_id").score
    assert s["A-1"] == pytest.approx(0.40)


# ---------- evaluate / build / report ----------

def test_build_report_and_labels_untouched(tmp_path, gt_file):
    before = gt_file.read_bytes()
    out_md = tmp_path / "reports" / "eval.md"
    r = ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=out_md)

    assert gt_file.read_bytes() == before                     # labels never changed
    assert r["llm"]["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert r["threshold"] == pytest.approx(0.30)              # best F1 on these pairs
    assert r["baseline"]["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}

    md = out_md.read_text(encoding="utf-8")
    assert "flattered" in md and "needs_review` counts as `different`" in md
    assert "rate suppressed" in md and "test labeller" in md
    # every disagreement appears with both texts
    for pid, us, india in (("C-3", "Inflators may explode.", "The inflator may rupture."),
                           ("D-4", "Power steering may shut off.", "Power steering deactivated.")):
        assert f"### {pid}" in md and us in md and india in md
    assert "### A-1" not in md and "### B-2" not in md       # agreed pairs not listed


def test_report_lists_revised_labels_and_prompt_tuning(tmp_path, gt_file):
    g = pd.read_csv(gt_file, dtype=str, keep_default_na=False)
    g["revision_note"] = ["", "", "", "Round 1: kept different, platform note"]
    g.to_csv(gt_file, index=False)
    out_md = tmp_path / "eval.md"
    ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=out_md)
    md = out_md.read_text(encoding="utf-8")
    assert "LLM matcher is flattered too" in md
    assert "`D-4` (now `different`): Round 1: kept different" in md


def test_revision_lines_empty_without_notes():
    assert ev._revision_lines(pd.DataFrame({"pair_id": ["a"], "label": ["same"]})) == []


def test_build_writes_eval_summary_with_suppressed_rates(tmp_path, gt_file):
    out_md = tmp_path / "reports" / "eval.md"
    ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=out_md)
    s = pd.read_csv(tmp_path / "reports" / "eval_summary.csv").set_index("matcher")
    assert list(s.index) == ["llm", "baseline"]
    assert (s.loc["llm", "tp"], s.loc["llm", "fn"], s.loc["llm", "positives"]) == (1, 1, 2)
    assert s.loc["llm", "recall_n"] == 2 and pd.isna(s.loc["llm", "recall"])   # n < 10
    assert {"n", "precision", "recall", "f1"} <= set(s.columns)


def test_summary_frame_shows_rates_at_n_10():
    c = {"tp": 5, "fp": 0, "fn": 0, "tn": 25}
    result = {"n": 30, "threshold": 0.2, "llm": {"confusion": c, "metrics": ev.metrics(c)},
              "baseline": {"confusion": c, "metrics": ev.metrics(c)}}
    s = ev.summary_frame(result).set_index("matcher")
    assert s.loc["llm", "f1"] == 1.0 and s.loc["llm", "f1_n"] == 10    # 2*5+0+0 = 10
    assert pd.isna(s.loc["llm", "precision"])                          # n = 5
    assert s.loc["baseline", "threshold"] == 0.2


def test_build_refuses_to_write_over_ground_truth(tmp_path, gt_file):
    with pytest.raises(ValueError, match="refusing"):
        ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=gt_file)


def test_main_prints_summary(monkeypatch, capsys):
    c = {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    monkeypatch.setattr(ev, "build", lambda: {"n": 2, "threshold": 0.3,
                        "llm": {"confusion": c, "metrics": ev.metrics(c)},
                        "baseline": {"confusion": c, "metrics": ev.metrics(c)}})
    ev.main()
    out = capsys.readouterr().out
    assert "flatters the baseline" in out and "TP=1" in out


# ---------- human labels and reviewed precision ----------

def _human(gt_file: Path, rows: list[tuple[str, str, str]]) -> Path:
    p = gt_file.parent / "human_labels.csv"
    pd.DataFrame(rows, columns=["pair_id", "human_label", "note"]).to_csv(p, index=False)
    return p


def test_human_label_overrides_claude_label_for_that_pair_only(gt_file):
    _human(gt_file, [("A-1", "different", "trim differs"), ("B-2", "", "")])
    df = ev.load_ground_truth(gt_file).set_index("pair_id")
    assert df.loc["A-1", "label"] == "different" and df.loc["A-1", "label_source"] == "human"
    assert df.loc["A-1", "reason"] == "trim differs" and df.loc["A-1", "confidence"] == "human"
    assert df.loc["B-2", "label"] == "different" and df.loc["B-2", "label_source"] == "claude"
    assert df.loc["C-3", "label"] == "same"                   # Claude's label kept
    assert not ev.human_verified(df.reset_index())


def test_missing_human_labels_file_means_all_claude(gt_file):
    df = ev.load_ground_truth(gt_file)
    assert set(df.label_source) == {"claude"}
    assert "no human labels yet" in ev.labelled_by_text(df)


@pytest.mark.parametrize("rows, message", [
    ([("A-1", "maybe", "")], "same, different or blank"),
    ([("Z-9", "same", "")], "not in the ground truth"),
    ([("A-1", "same", ""), ("A-1", "different", "")], "listed twice"),
])
def test_bad_human_labels_are_refused(gt_file, rows, message):
    _human(gt_file, rows)
    with pytest.raises(ValueError, match=message):
        ev.load_ground_truth(gt_file)


def test_labelled_by_text_partial_and_complete(gt_file):
    _human(gt_file, [("A-1", "same", ""), ("B-2", "Different", "")])     # case-insensitive
    text = ev.labelled_by_text(ev.load_ground_truth(gt_file))
    assert text.startswith("a person for 2 of 4 pairs (human_labels.csv)")
    assert "test labeller for the other 2" in text
    _human(gt_file, [(p, "same", "") for p in ("A-1", "B-2", "C-3", "D-4")])
    assert ev.labelled_by_text(ev.load_ground_truth(gt_file)) ==         "a person for all 4 pairs (human_labels.csv)"


def test_report_names_who_labelled_and_leaves_human_file_untouched(tmp_path, gt_file):
    # A-1: the LLM says same, the person says different -> listed as a disagreement
    hl = _human(gt_file, [("A-1", "different", "checked"), ("C-3", "different", "")])
    before = hl.read_bytes()
    out = tmp_path / "eval.md"
    ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=out)
    text = out.read_text(encoding="utf-8")
    assert "**Who labelled it:** a person for 2 of 4 pairs (human_labels.csv)" in text
    assert "(labelled by a person) - note: checked" in text
    assert hl.read_bytes() == before
    summary = pd.read_csv(tmp_path / "eval_summary.csv")
    assert summary.labelled_by.iloc[0].startswith("a person for 2 of 4")


def test_reviewed_precision_none_without_table(tmp_path):
    con = duckdb.connect(str(tmp_path / "x.duckdb"))
    assert ev.reviewed_precision(con) is None


def test_report_and_summary_include_reviewed_precision(tmp_path, gt_file):
    db = _db(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE matches_reviewed AS
        SELECT 'same' AS label, 'reviewed' AS review_status,
               CASE WHEN i < 11 THEN 'same' ELSE 'different' END AS final_label
        FROM range(12) AS t(i)""")
    con.close()
    out = tmp_path / "eval.md"
    ev.build(db_path=db, gt_path=gt_file, out_md=out)
    text = out.read_text(encoding="utf-8")
    assert "## Matcher precision (human-reviewed)" in text and "11 of 12" in text
    summary = pd.read_csv(tmp_path / "eval_summary.csv")
    row = summary[summary.matcher == "llm_reviewed"].iloc[0]
    assert (row.n, row.tp, row.fp, row.precision, row.precision_n) == (12, 11, 1, 0.9167, 12)


def test_report_says_when_nothing_is_reviewed_yet(tmp_path, gt_file):
    out = tmp_path / "eval.md"
    ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=out)
    assert "No LLM `same` pairs have been reviewed yet" in out.read_text(encoding="utf-8")


def test_report_and_summary_include_the_random_sample(tmp_path, gt_file):
    db = _db(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE matches_reviewed AS
        SELECT 'P' || i AS pair_id, 'same' AS label, 'reviewed' AS review_status,
               CASE WHEN i < 10 THEN 'same' ELSE 'different' END AS final_label
        FROM range(14) AS t(i)""")
    con.close()
    pd.DataFrame({"pair_id": [f"P{i}" for i in range(2, 14)] + ["NEW-1"], "seed": 42,
                  "drawn_on": "2026-10-24"}).to_csv(gt_file.parent / "review_sample.csv",
                                                    index=False)
    out = tmp_path / "eval.md"
    ev.build(db_path=db, gt_path=gt_file, out_md=out)
    text = out.read_text(encoding="utf-8")
    assert "**Random sample** (13 LLM `same` pairs" in text
    assert "a person confirmed 8 of 12: precision 8 / 12 = 0.667 (n=12)" in text
    assert "no longer a pair: 1." in text
    summary = pd.read_csv(tmp_path / "eval_summary.csv")
    row = summary[summary.matcher == "llm_reviewed_sample"].iloc[0]
    assert (row.n, row.tp, row.fp, row.precision, row.precision_n) == (12, 8, 4, 0.6667, 12)


def test_report_without_a_saved_sample_has_no_sample_line(tmp_path, gt_file):
    out = tmp_path / "eval.md"
    ev.build(db_path=_db(tmp_path), gt_path=gt_file, out_md=out)
    assert "Random sample" not in out.read_text(encoding="utf-8")


# ---------- test pairs that are no longer candidate pairs ----------

def _drop_candidate(db: Path, pair: str) -> None:
    campno, row = pair.split("-")
    con = duckdb.connect(str(db))
    con.execute("DELETE FROM candidate_pairs WHERE CAMPNO = ? AND csv_row = ?", [campno, int(row)])
    con.execute("DELETE FROM matches WHERE pair_id = ?", [pair])
    con.close()


def test_dropped_pairs_lists_test_pairs_that_are_no_longer_candidates(tmp_path, gt_file):
    db = _db(tmp_path)
    _drop_candidate(db, "D-4")
    con = duckdb.connect(str(db), read_only=True)
    d = ev.dropped_pairs(con, ev.load_ground_truth(gt_file))
    assert d.pair_id.tolist() == ["D-4"] and d.label.tolist() == ["different"]
    assert "no longer a candidate pair" in d.reason.iloc[0]


def test_build_scores_the_rest_and_reports_the_dropped_pair(tmp_path, gt_file):
    db = _db(tmp_path)
    _drop_candidate(db, "D-4")
    out = tmp_path / "eval.md"
    r = ev.build(db_path=db, gt_path=gt_file, out_md=out)
    assert r["n"] == 3 and r["dropped"].pair_id.tolist() == ["D-4"]
    text = out.read_text(encoding="utf-8")
    assert "3 pairs scored, 1 dropped" in text
    assert "## Test pairs no longer scored" in text and "`D-4`" in text
    summary = pd.read_csv(tmp_path / "eval_summary.csv").set_index("matcher")
    assert summary.loc["llm", "n"] == 3 and summary.loc["llm", "n_dropped"] == 1


def test_candidate_without_a_matcher_result_still_stops(tmp_path, gt_file):
    db = _db(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("DELETE FROM matches WHERE pair_id = 'D-4'")     # still a candidate pair
    con.close()
    with pytest.raises(ValueError, match="run python -m src.matcher"):
        ev.build(db_path=db, gt_path=gt_file, out_md=tmp_path / "eval.md")
