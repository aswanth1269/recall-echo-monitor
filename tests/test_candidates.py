"""Tests for src/candidates.py on a small synthetic database, plus the same
three guarantees checked on the real database when it has been built."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import candidates as cd
from src.load_nhtsa import DB_PATH

REPORT = date(2020, 7, 1)
D = lambda days: REPORT + timedelta(days=days)  # noqa: E731
AIRBAG = "Driver airbag inflator may rupture and spray metal fragments"


def _make_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE nhtsa_focus AS SELECT * FROM (VALUES
        ('1', 'C1', 'HONDA',  'CITY X', NULL,         ?, 'Injury risk', 'Replace inflator'),
        ('2', 'C1', 'HONDA',  'CITY X', 'AIR BAGS',   ?, 'Injury risk', 'Replace inflator'),
        ('3', 'C1', 'HONDA',  'CITY Y', 'SEAT BELTS', ?, 'Injury risk', 'Replace inflator'),
        ('4', 'C2', 'TOYOTA', 'CITY',   'BRAKES',     'Brake hose may leak', 'Crash', 'Replace hose'),
        ('5', 'C3', 'HONDA',  'PILOT',  'ENGINE',     'Stall', 'Crash', 'Reflash'),
        ('6', 'C4', 'NISSAN', 'SENTRA', 'STEERING',   'Steering lock', 'Crash', 'Fix'),
        ('7', 'C5', 'HONDA',  'CITY X', 'WIPERS',     'Wiper motor fails', 'Low visibility', 'Fix'))
        AS t(RECORD_ID, CAMPNO, make, MODELTXT, COMPNAME, DESC_DEFECT, CONEQUENCE_DEFECT,
             CORRECTIVE_ACTION)""", [AIRBAG, AIRBAG, AIRBAG])
    con.execute("""CREATE TABLE nhtsa_model_family AS SELECT * FROM (VALUES
        ('HONDA', 'CITY X', 'CITY', 'rule'), ('HONDA', 'CITY Y', 'CITY', 'rule'),
        ('TOYOTA', 'CITY', 'CITY', 'rule'), ('HONDA', 'PILOT', NULL, 'llm'),
        ('NISSAN', 'SENTRA', 'SUNNY', 'llm')) AS t(make, MODELTXT, family, method)""")
    con.execute("""CREATE TABLE nhtsa_campaigns AS SELECT * FROM (VALUES
        ('C1', ['HONDA'], ?), ('C2', ['TOYOTA'], ?), ('C3', ['HONDA'], ?),
        ('C4', ['NISSAN'], ?), ('C5', ['HONDA'], DATE '2026-01-01'))
        AS t(CAMPNO, makes, report_date)""", [REPORT] * 4)
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'HONDA',  ?, 'Steering column noise'),       -- -180 days: in
        (2, 'HONDA',  ?, 'Steering column noise'),       -- -181 days: out
        (3, 'HONDA',  ?, 'Steering column noise'),       -- +730 days: in
        (4, 'HONDA',  ?, 'Steering column noise'),       -- +731 days: out
        (5, 'HONDA',  NULL, ?),                          -- NULL date: never paired
        (6, 'HONDA',  ?, ?),                             -- same text: should rank 1
        (7, 'TOYOTA', ?, ?),                             -- other make
        (8, 'HONDA',  ?, ?),                             -- no shared family
        (9, 'NISSAN', ?, 'Steering lock may engage'))
        AS t(csv_row, make, recall_date, defect_description)""",
                [D(-180), D(-181), D(730), D(731), AIRBAG, D(31), AIRBAG, D(31), AIRBAG,
                 D(31), AIRBAG, D(31)])
    con.execute("""CREATE TABLE siam_row_family AS SELECT * FROM (VALUES
        (1, 'HONDA', 'CITY', 'rule'), (2, 'HONDA', 'CITY', 'rule'), (3, 'HONDA', 'CITY', 'rule'),
        (4, 'HONDA', 'CITY', 'rule'), (5, 'HONDA', 'CITY', 'rule'), (6, 'HONDA', 'CITY', 'rule'),
        (7, 'TOYOTA', 'CITY', 'rule'), (8, 'HONDA', 'JAZZ', 'rule'), (9, 'NISSAN', 'SUNNY', 'rule'))
        AS t(csv_row, make, family, method)""")
    con.close()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "recalls.duckdb"
    _make_db(path)
    return path


@pytest.fixture
def vocab(tmp_path: Path) -> Path:
    path = tmp_path / "families.csv"
    pd.DataFrame({"make": ["HONDA", "HONDA", "TOYOTA", "NISSAN", "HONDA"],
                  "family": ["CITY", "JAZZ", "CITY", "SUNNY", "CITY"],  # dup on purpose
                  "source": "x", "evidence": "", "source_url": ""}).to_csv(path, index=False)
    return path


@pytest.fixture
def built(db: Path, vocab: Path):
    summary = cd.build(db_path=db, vocab_csv=vocab)
    return duckdb.connect(str(db), read_only=True), summary


# ---------- load_india_families / build_relevant ----------

def test_load_india_families_dedupes(vocab: Path):
    fams = cd.load_india_families(vocab)
    assert len(fams) == 4 and list(fams.columns) == ["make", "family"]


def test_relevant_needs_an_india_family_and_takes_first_non_null(db, vocab):
    con = duckdb.connect(str(db))
    assert cd.build_relevant(con, cd.load_india_families(vocab)) == 4   # C1 C2 C4 C5, not C3
    rel = {r[0]: r[1:] for r in con.execute(
        "SELECT CAMPNO, COMPNAME, components, llm_only FROM india_relevant_campaigns").fetchall()}
    assert "C3" not in rel                                # PILOT has no India family
    assert rel["C1"][0] == "AIR BAGS"                     # RECORD_ID 1 is NULL, 2 is first non-null
    assert rel["C1"][1] == ["AIR BAGS", "SEAT BELTS"]
    assert rel["C1"][2] is False and rel["C4"][2] is True  # SUNNY only via the LLM


# ---------- the three required guarantees (synthetic) ----------

def test_every_pair_has_the_same_make(built):
    con, _ = built
    mismatched = con.execute("""SELECT count(*) FROM candidate_pairs p
        JOIN siam_recalls s USING (csv_row) WHERE s.make <> p.make""").fetchone()[0]
    assert mismatched == 0
    assert con.execute("SELECT list(csv_row) FROM candidate_pairs WHERE CAMPNO = 'C2'"
                       ).fetchone()[0] == [7]          # Toyota CITY never meets Honda CITY rows


def test_date_window_boundaries_are_inclusive(built):
    con, _ = built
    rows = {r[0] for r in con.execute(
        "SELECT csv_row FROM candidate_pairs WHERE CAMPNO = 'C1'").fetchall()}
    assert {1, 3} <= rows          # exactly -180 and +730 days are kept
    assert not {2, 4} & rows       # -181 and +731 days are not
    assert 8 not in rows           # no shared family


def test_null_recall_date_never_paired_and_logged(db, vocab, caplog):
    with caplog.at_level(logging.INFO, logger="src.candidates"):
        summary = cd.build(db_path=db, vocab_csv=vocab)
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT count(*) FROM candidate_pairs WHERE csv_row = 5").fetchone()[0] == 0
    assert summary["null_date_skipped"] == 1
    assert "skipped 1 SIAM rows with a NULL recall_date" in caplog.text


# ---------- scoring and ranking ----------

def test_score_pairs_prefers_matching_text():
    pairs = pd.DataFrame({"CAMPNO": ["C1", "C1", "C1"], "csv_row": [1, 2, 3],
                          "days_after_report": [0, 0, 0]})
    scored = cd.score_pairs(pairs, {"C1": AIRBAG},
                            {1: AIRBAG, 2: "Steering column noise", 3: ""})
    s = dict(zip(scored.csv_row, scored.score))
    assert s[1] > s[2] and s[3] == 0.0
    assert s[1] == pytest.approx(1.0, abs=1e-3)


def test_score_pairs_empty_input():
    assert cd.score_pairs(pd.DataFrame(columns=["CAMPNO", "csv_row", "days_after_report"]),
                          {}, {}).empty


def test_top_k_keeps_three_and_breaks_ties_by_time_gap():
    scored = pd.DataFrame({"CAMPNO": ["C"] * 5, "make": ["HONDA"] * 5,
                           "csv_row": [1, 2, 3, 4, 5],
                           "days_after_report": [400, -10, 50, 0, 5],
                           "score": [0.9, 0.5, 0.5, 0.1, 0.5]})
    top = cd.top_k(scored, k=3)
    assert top.csv_row.tolist() == [1, 5, 2] and top["rank"].tolist() == [1, 2, 3]


def test_top_k_ranks_each_make_of_a_shared_campaign_separately():
    # e.g. 16V045000 covers Audi Q5 and VW Tiguan: VW's weaker row must not be crowded out
    scored = pd.DataFrame({"CAMPNO": ["16V045000"] * 5,
                           "make": ["AUDI"] * 4 + ["VOLKSWAGEN"],
                           "csv_row": [1, 2, 3, 4, 9], "days_after_report": [0] * 5,
                           "score": [0.9, 0.8, 0.7, 0.6, 0.1]})
    top = cd.top_k(scored, k=3)
    assert top[top.make == "AUDI"].csv_row.tolist() == [1, 2, 3]
    assert top[top.make == "VOLKSWAGEN"][["csv_row", "rank"]].values.tolist() == [[9, 1]]


# ---------- text preparation ----------

HONDA_US = ("Honda (American Honda Motor Co.) is recalling certain 2005-2006 Acura MDX and "
            "Honda CR-V vehicles, 2005-2011 Honda Element vehicles. These vehicles are "
            "equipped with certain air bag inflators assembled as part of the passenger "
            "frontal air bag modules. The inflator may rupture.")
HONDA_SIAM = ("During passenger side airbag deployment inflator body may get ruptured due to "
              "excessive internal pressure.")


@pytest.mark.parametrize("text, kept, dropped", [
    (HONDA_US, "air bag inflators", "Acura MDX"),
    ("Kia America, Inc. (Kia) is recalling certain 2022-2024 EV6 vehicles. The Integrated "
     "Charging Control Unit (ICCU) may become damaged.", "ICCU", "EV6"),
    ("Hyundai Motor America (Hyundai) recalled certain 2019-2021 Tucson vehicles on "
     "September 4, 2020. On December 30, 2020, Hyundai expanded the recall population, and "
     "added certain 2016-2018 Tucson vehicles. The HECU could corrode internally.",
     "HECU could corrode", "Tucson"),
    # a scope sentence that states the defect keeps its clause
    ("BMW is recalling certain 2019 X5 vehicles that may have a fuel pump that fails.",
     "fuel pump that fails", "X5"),
])
def test_strip_recall_scope_real_shapes(text, kept, dropped):
    out = cd.strip_recall_scope(text)
    assert kept in out and dropped not in out


def test_strip_recall_scope_never_empties_text():
    only_scope = "Honda is recalling certain 2019 Civic vehicles."
    assert cd.strip_recall_scope(only_scope) == only_scope
    assert cd.strip_recall_scope(None) == ""


def test_normalise_text_merges_compounds_and_drops_years():
    s = cd.normalise_text("Air Bags and SEAT BELT in 2019 models")
    assert "airbag" in s and "seatbelt" in s and "2019" not in s


def test_analyze_text_stems_and_drops_stopwords():
    tokens = cd.analyze_text(HONDA_SIAM)
    assert {"ruptur", "inflat", "airbag"} <= set(tokens)
    assert "during" not in tokens and "may" not in tokens
    assert "passeng airbag" in tokens                       # bigrams of stems
    assert cd.analyze_text("ruptured") == cd.analyze_text("rupture")


def test_preparation_lifts_honda_inflator_pair():
    """The baseline scored this same-defect pair 0.018; it must now clearly beat noise."""
    pairs = pd.DataFrame({"CAMPNO": ["17V029000"] * 2, "csv_row": [46, 99],
                          "days_after_report": [0, 0]})
    us = {"17V029000": cd._text("AIR BAGS:FRONTAL:PASSENGER SIDE:INFLATOR MODULE",
                                cd.strip_recall_scope(HONDA_US))}
    siam = {46: HONDA_SIAM, 99: "Steering column bolt may loosen over time."}
    s = dict(zip(*cd.score_pairs(pairs, us, siam)[["csv_row", "score"]].values.T))
    assert s[46] > 0.1 and s[99] == 0.0


def test_build_ranks_identical_text_first_and_counts_window(built):
    con, summary = built
    assert con.execute("SELECT csv_row FROM candidate_pairs WHERE CAMPNO = 'C1' AND rank = 1"
                       ).fetchone()[0] == 6
    n = dict(con.execute("SELECT CAMPNO, n_window_candidates FROM india_relevant_campaigns"
                         ).fetchall())
    assert n == {"C1": 3, "C2": 1, "C4": 1, "C5": 0}
    assert con.execute("SELECT llm_only FROM candidate_pairs WHERE CAMPNO = 'C4'"
                       ).fetchone()[0] is True
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert "candidate_window" not in tables


# ---------- report / main ----------

def test_report_lists_every_section(built):
    _, summary = built
    text = cd.report(summary)
    for part in ("India-relevant campaigns per make", "Campaigns with zero candidates: 1",
                 "C5", "Total candidate pairs: 5", "HIGH scores", "MIDDLE scores",
                 "LOW scores", "NULL recall_date: 1"):
        assert part in text
    assert "n<10" in text  # rates with n < 10 are suppressed


def test_report_shows_missing_text_plainly():
    pairs = pd.DataFrame({"CAMPNO": ["C1"], "csv_row": [5], "make": ["HONDA"],
                          "shared_families": [["CITY"]], "score": [0.0], "rank": [1],
                          "days_after_report": [3], "llm_only": [False],
                          "us_text": [AIRBAG], "siam_text": [float("nan")]})
    summary = {"relevant": pd.DataFrame({"make": ["HONDA"], "relevant_campaigns": [1],
                                         "llm_only": [0], "all_campaigns": [1]}),
               "zero": pd.DataFrame(columns=["make", "zero_candidate_campaigns", "examples"]),
               "by_rank": pd.DataFrame({"rank": [1], "pairs": [1]}),
               "pairs": pairs, "null_date_skipped": 0}
    text = cd.report(summary)
    assert "(no SIAM defect text)" in text and "nan" not in text


def test_main_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(cd, "build", lambda: {"x": 1})
    monkeypatch.setattr(cd, "report", lambda s: "REPORT OK")
    cd.main()
    assert "REPORT OK" in capsys.readouterr().out


# ---------- the same guarantees on the real database ----------

def _real():
    if not DB_PATH.exists():
        pytest.skip("data/clean/recalls.duckdb not built")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    if "candidate_pairs" not in {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}:
        pytest.skip("run python -m src.candidates first")
    return con


def test_real_pairs_same_make_window_and_no_null_dates():
    con = _real()
    bad_make, outside, null_date = con.execute("""
        SELECT count(*) FILTER (WHERE s.make <> p.make),
               count(*) FILTER (WHERE s.recall_date < p.report_date - INTERVAL 180 DAY
                                   OR s.recall_date > p.report_date + INTERVAL 730 DAY),
               count(*) FILTER (WHERE s.recall_date IS NULL)
        FROM candidate_pairs p JOIN siam_recalls s USING (csv_row)""").fetchone()
    assert (bad_make, outside, null_date) == (0, 0, 0)
    assert con.execute("SELECT max(rank) FROM candidate_pairs").fetchone()[0] <= 3
    over = con.execute("""SELECT count(*) FROM (SELECT CAMPNO, make FROM candidate_pairs
                          GROUP BY CAMPNO, make HAVING count(*) > 3)""").fetchone()[0]
    assert over == 0                                     # top 3 per campaign AND make


def test_real_campaign_make_rows_are_unique():
    con = _real()
    rows, distinct = con.execute("""SELECT count(*), count(DISTINCT (CAMPNO, make))
                                    FROM india_relevant_campaigns""").fetchone()
    assert rows == distinct
