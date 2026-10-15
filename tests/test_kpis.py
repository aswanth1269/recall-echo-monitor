"""Each KPI query run on a small fake database whose answers are worked out by hand.

Fake data
---------
Make A (luxury), 12 campaigns A00-A11, reported 2020-01-01.
  COMPNAME: A00-A09 'AIR BAGS:FRONTAL:...', A10-A11 'BRAKES:DISC'.
  A00 same  -> India 2020-01-11  (lag +10)
  A01 same  -> India 2020-03-01 and 2019-12-22, earliest counts (lag -10)
  A02 same  -> India 2020-02-20  (lag +50)
  A03 needs_review only, A04 different only, A05-A11 no pairs.
Make B (mass_market), 5 campaigns B0-B4, reported 2021-01-01, 'ENGINE:OIL'.
  B0 same -> India 2021-01-31 (lag +30).
OTA: A00-A05 true, A06-A09 false, A10 NULL, A11 absent; B0 true, B1-B4 false.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import kpis


def _fake(tmp_path: Path) -> tuple[Path, Path]:
    camps = ([(f"A{i:02d}", "A", date(2020, 1, 1),
               "AIR BAGS:FRONTAL:DRIVER" if i < 10 else "BRAKES:DISC") for i in range(12)]
             + [(f"B{i}", "B", date(2021, 1, 1), "ENGINE:OIL") for i in range(5)])
    matches = [("A00", "A", 1, "same"), ("A01", "A", 2, "same"), ("A01", "A", 3, "same"),
               ("A02", "A", 4, "same"), ("A03", "A", 5, "needs_review"),
               ("A04", "A", 6, "different"), ("B0", "B", 7, "same"),
               ("B1", "B", 8, "different")]
    siam = [(1, date(2020, 1, 11)), (2, date(2020, 3, 1)), (3, date(2019, 12, 22)),
            (4, date(2020, 2, 20)), (5, date(2020, 1, 5)), (6, date(2020, 1, 5)),
            (7, date(2021, 1, 31)), (8, date(2021, 2, 1))]
    ota = ([(f"A{i:02d}", True) for i in range(6)] + [(f"A{i:02d}", False) for i in range(6, 10)]
           + [("A10", None), ("B0", True)] + [(f"B{i}", False) for i in range(1, 5)])

    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    for name, rows, cols in (
            ("india_relevant_campaigns", camps, ["CAMPNO", "make", "report_date", "COMPNAME"]),
            ("matches", matches, ["CAMPNO", "make", "csv_row", "label"]),
            ("siam_recalls", siam, ["csv_row", "recall_date"]),
            ("campaign_ota", ota, ["CAMPNO", "ota"])):
        df = pd.DataFrame(rows, columns=cols)
        if name == "campaign_ota":
            df["ota"] = df.ota.astype("boolean")
        con.register("df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
    con.execute("ALTER TABLE matches ADD COLUMN scope VARCHAR DEFAULT 'all'")
    con.execute("ALTER TABLE matches ADD COLUMN confidence DOUBLE DEFAULT 0.9")
    con.execute("ALTER TABLE matches ADD COLUMN us_evidence VARCHAR DEFAULT 'us quote'")
    con.execute("ALTER TABLE matches ADD COLUMN india_evidence VARCHAR DEFAULT 'india quote'")
    con.execute("ALTER TABLE india_relevant_campaigns ADD COLUMN DESC_DEFECT VARCHAR DEFAULT 'us text'")
    con.execute("ALTER TABLE siam_recalls ADD COLUMN model VARCHAR DEFAULT 'M'")
    con.execute("ALTER TABLE siam_recalls ADD COLUMN defect_description VARCHAR DEFAULT 'india text'")
    # rank: A01's 2019-12-22 row (csv_row 3) is rank 2, the 2020-03-01 row rank 1
    con.execute("""CREATE TABLE candidate_pairs AS
        SELECT CAMPNO, make, csv_row, CASE WHEN csv_row = 3 THEN 2 ELSE 1 END AS rank
        FROM matches""")
    con.close()

    makes = tmp_path / "india_makes.csv"
    pd.DataFrame({"make": ["Make A", "Make B"], "segment": ["luxury", "mass_market"],
                  "nhtsa_maketxt_expected": ["A", "B"]}).to_csv(makes, index=False)
    return db, makes


@pytest.fixture
def results(tmp_path: Path) -> dict[str, pd.DataFrame]:
    db, makes = _fake(tmp_path)
    return kpis.build(db_path=db, out_dir=tmp_path / "out", makes_csv=makes)


def _row(df: pd.DataFrame, **match) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k, v in match.items():
        mask &= df[k].eq(v)
    assert mask.sum() == 1, f"expected one row for {match}, got {mask.sum()}"
    return df[mask].iloc[0]


# ---------- 1. echo_rate ----------

def test_echo_rate_by_make_segment_and_all(results):
    df = results["echo_rate"]
    a = _row(df, level="make", grp="A")
    assert (a.n, a.echoing, a.echo_rate, a.needs_review_not_echoing) == (12, 3, 0.25, 1)
    b = _row(df, level="make", grp="B")
    assert (b.n, b.echoing) == (5, 1) and pd.isna(b.echo_rate)          # n < 10 suppressed
    assert _row(df, level="segment", grp="luxury").echo_rate == 0.25
    assert pd.isna(_row(df, level="segment", grp="mass_market").echo_rate)
    total = _row(df, level="all", grp="ALL")
    assert (total.n, total.echoing) == (17, 4) and total.echo_rate == round(4 / 17, 4)


# ---------- 2. echo_lag ----------

def test_echo_lag_uses_earliest_match_and_counts_india_first(results):
    df = results["echo_lag"]
    a = _row(df, level="make", grp="A")                  # lags -10, 10, 50
    assert (a.n, a.median_days, a.p25_days, a.p75_days, a.india_first) == (3, 10, 0, 30, 1)
    b = _row(df, level="make", grp="B")
    assert (b.n, b.median_days, b.india_first) == (1, 30, 0)
    total = _row(df, level="all", grp="ALL")             # lags -10, 10, 30, 50
    assert (total.n, total.median_days, total.p25_days, total.p75_days) == (4, 20, 5, 35)
    assert _row(df, level="segment", grp="luxury").n == 3


# ---------- 3. echo_by_component ----------

def test_echo_by_component_keeps_only_groups_with_n_at_least_10(results):
    df = results["echo_by_component"]
    assert df.component_group.tolist() == ["AIR BAGS"]   # BRAKES n=2, ENGINE n=5 dropped
    row = df.iloc[0]
    assert (row.n, row.echoing, row.echo_rate) == (10, 3, 0.3)


# ---------- 4. ota ----------

def test_ota_share_counts_unknown_separately(results):
    df = results["ota"]
    share = df[df.part == "ota_share"]
    a = _row(share, level="make", grp="A")
    assert (a.n, a.n_known_ota, a.ota_true, a.ota_share) == (12, 10, 6, 0.6)
    b = _row(share, level="make", grp="B")
    assert (b.n, b.n_known_ota, b.ota_true) == (5, 5, 1) and pd.isna(b.ota_share)
    y2020 = _row(share, level="report_year", grp="2020")
    assert (y2020.n, y2020.ota_true) == (12, 6)
    total = _row(share, level="all", grp="ALL")
    assert (total.n, total.n_known_ota, total.ota_true) == (17, 15, 7)
    assert total.ota_share == round(7 / 15, 4)


def test_echo_rate_by_ota_group(results):
    df = results["ota"]
    by = df[df.part == "echo_by_ota"]
    ota = _row(by, grp="ota")              # A00-A05, B0 -> echo A00, A01, A02, B0
    assert (ota.n, ota.echoing) == (7, 4) and pd.isna(ota.echo_rate)
    not_ota = _row(by, grp="not_ota")
    assert (not_ota.n, not_ota.echoing) == (8, 0)
    unknown = _row(by, grp="unknown")      # A10 (NULL) and A11 (absent)
    assert unknown.n == 2


# ---------- 5. echo_explorer (row-level export) ----------

def test_echo_explorer_one_row_per_campaign_with_best_pair(results, tmp_path):
    ex = results["echo_explorer"]
    assert len(ex) == 17 and not ex[["CAMPNO", "make"]].duplicated().any()
    by = ex.set_index("CAMPNO")
    # A01 has two `same` pairs: the earliest India date wins even though it ranks 2nd
    assert by.loc["A01", "label"] == "same"
    assert str(by.loc["A01", "india_recall_date"])[:10] == "2019-12-22"
    assert by.loc["A01", "lag_days"] == -10                     # same as echo_lag uses
    assert by.loc["A03", "label"] == "needs_review" and pd.isna(by.loc["A03", "lag_days"])
    assert by.loc["A04", "label"] == "different"
    assert by.loc["A05", "label"] == "no_candidate"
    assert pd.isna(by.loc["A05", "india_recall_date"])          # never filled in
    assert by.loc["A00", "segment"] == "luxury" and by.loc["B0", "segment"] == "mass_market"
    out = pd.read_csv(tmp_path / "out" / "echo_explorer.csv")   # no kpi_ prefix
    assert list(out.columns) == ["CAMPNO", "make", "segment", "report_date", "component",
                                 "us_defect", "label", "india_recall_date", "india_model",
                                 "india_defect", "lag_days", "confidence", "us_evidence",
                                 "india_evidence"]


def test_explorer_echoes_agree_with_echo_rate(results):
    ex, rate = results["echo_explorer"], results["echo_rate"]
    assert int((ex.label == "same").sum()) == int(_row(rate, level="all", grp="ALL").echoing)


# ---------- exports, checks, helpers ----------

def test_every_kpi_is_exported_with_n(results, tmp_path):
    for name in kpis.KPIS:
        out = pd.read_csv(tmp_path / "out" / f"kpi_{name}.csv")
        assert "n" in out.columns and len(out) == len(results[name])


def test_every_sql_file_starts_with_a_one_line_comment():
    for name in kpis.KPIS:
        first = (kpis.SQL_DIR / f"{name}.sql").read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith(f"-- {name}:")


def test_check_inputs_rejects_partial_matcher_run(tmp_path):
    db, makes = _fake(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("UPDATE matches SET scope = 'ground_truth'")
    con.close()
    with pytest.raises(ValueError, match="--all"):
        kpis.build(db_path=db, out_dir=tmp_path / "o", makes_csv=makes)


def test_check_inputs_rejects_unjudged_pairs(tmp_path):
    db, makes = _fake(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO candidate_pairs VALUES ('B4', 'B', 99, 1)")
    con.close()
    with pytest.raises(ValueError, match="no verdict"):
        kpis.build(db_path=db, out_dir=tmp_path / "o", makes_csv=makes)


def test_check_inputs_requires_campaign_ota(tmp_path):
    db, makes = _fake(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("DROP TABLE campaign_ota")
    con.close()
    with pytest.raises(ValueError, match="campaign_ota is missing"):
        kpis.build(db_path=db, out_dir=tmp_path / "o", makes_csv=makes)


def test_load_make_segments_uses_nhtsa_spelling():
    seg = kpis.load_make_segments()
    assert set(seg.segment) == {"luxury", "mass_market"}
    assert "MERCEDES-BENZ" in set(seg.make) and len(seg) == 10


def test_run_kpi_executes_named_file(tmp_path):
    db, _ = _fake(tmp_path)
    con = duckdb.connect(str(db), read_only=True)
    con.register("make_segments", pd.DataFrame({"make": ["A", "B"],
                                                "segment": ["luxury", "mass_market"]}))
    assert len(kpis.run_kpi(con, "echo_by_component")) == 1


def test_main_prints_every_table(monkeypatch, capsys):
    tables = {n: pd.DataFrame({"n": [1]}) for n in kpis.KPIS}
    tables["echo_explorer"] = pd.DataFrame({"label": ["same", "no_candidate"]})
    monkeypatch.setattr(kpis, "build", lambda: tables)
    kpis.main()
    out = capsys.readouterr().out
    assert all(f"kpi_{n}.csv" in out for n in kpis.KPIS) and "echo_explorer.csv" in out
