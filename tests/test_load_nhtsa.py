"""Tests for src/load_nhtsa.py.

The three checks the tables must satisfy (no other makes, no pre-2015 report
dates, unique campaign numbers) run against a fresh build of the real NHTSA
file in a temp directory, so they never touch data/clean. Each function also
has a small test on synthetic records.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import duckdb
import pytest

from src import load_nhtsa as ln

ROOT = Path(__file__).resolve().parents[1]
NAMES = ln.read_field_names()


# ---------- synthetic fixtures ----------

def _record(**values: str) -> str:
    row = {n: "" for n in NAMES}
    row.update(values)
    return "\t".join(row[n] for n in NAMES)


SYNTHETIC = [
    _record(RECORD_ID="1", CAMPNO="20V001000", MAKETXT="TOYOTA", MODELTXT="CAMRY",
            YEARTXT="2019", RCDATE="20200105", COMPNAME="AIR BAGS"),
    _record(RECORD_ID="2", CAMPNO="20V001000", MAKETXT="TOYOTA", MODELTXT="CAMRY",
            YEARTXT="2020", RCDATE="20200105", COMPNAME="AIR BAGS"),
    _record(RECORD_ID="3", CAMPNO="20V001000", MAKETXT="TOYOTA", MODELTXT="RAV4",
            YEARTXT="9999", RCDATE="20200105", COMPNAME="AIR BAGS"),
    _record(RECORD_ID="4", CAMPNO="18V002000", MAKETXT="MERCEDES-BENz", MODELTXT="C300",
            YEARTXT="2017", RCDATE="20180101", MFGCAMPNO="MB-1"),
    _record(RECORD_ID="5", CAMPNO="19V003000", MAKETXT="HYUNDAI TRANSLEAD",
            MODELTXT="TRAILER", YEARTXT="2019", RCDATE="20190101"),
    _record(RECORD_ID="6", CAMPNO="14V004000", MAKETXT="HONDA", MODELTXT="CIVIC",
            YEARTXT="2014", RCDATE="20141231"),
    _record(RECORD_ID="7", CAMPNO="20V005000", MAKETXT="FORD", MODELTXT="F-150",
            YEARTXT="2020", RCDATE="20200301"),
    _record(RECORD_ID="8", CAMPNO="20V006000", MAKETXT="LEXUS", MODELTXT="RX",
            YEARTXT="2020", RCDATE="20200301"),
]


@pytest.fixture
def synthetic_zip(tmp_path: Path) -> Path:
    path = tmp_path / "recalls.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("FLAT_RCL.txt", "\r\n".join(SYNTHETIC) + "\r\n")
    return path


@pytest.fixture
def synthetic_db(tmp_path: Path, synthetic_zip: Path) -> tuple[Path, dict]:
    db = tmp_path / "clean" / "recalls.duckdb"
    counts = ln.build(db_path=db, zip_path=synthetic_zip)
    return db, counts


# ---------- read_field_names ----------

def test_read_field_names_matches_rcl():
    assert len(NAMES) == 29
    assert NAMES[:5] == ["RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT"]
    assert NAMES[15] == "RCDATE"
    assert "CONEQUENCE_DEFECT" in NAMES  # misspelt in the source, kept as-is
    assert NAMES[-1] == "PARK_OUTSIDE"


def test_read_field_names_rejects_gaps(tmp_path: Path):
    bad = tmp_path / "RCL.txt"
    bad.write_text("1        RECORD_ID  NUMBER(9)  x\n3        MAKETXT  CHAR(25)  y\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="1..N"):
        ln.read_field_names(bad)


# ---------- read_nhtsa ----------

def test_read_nhtsa_names_columns_and_nulls_empties(synthetic_zip: Path):
    df = ln.read_nhtsa(synthetic_zip, NAMES)
    assert list(df.columns) == NAMES
    assert len(df) == len(SYNTHETIC)
    assert df.loc[0, "MFGCAMPNO"] is None or df["MFGCAMPNO"].isna()[0]
    assert (df.astype(str) == "").sum().sum() == 0, "empty strings must become NULL"
    assert not df["PARK_OUTSIDE"].astype(str).str.contains("\r").any()


def test_read_nhtsa_rejects_multi_file_zip(tmp_path: Path):
    path = tmp_path / "two.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.txt", SYNTHETIC[0])
        zf.writestr("b.txt", SYNTHETIC[1])
    with pytest.raises(ValueError, match="expected one file"):
        ln.read_nhtsa(path, NAMES)


# ---------- focus_make_table ----------

def test_focus_make_table_covers_every_picked_make():
    table = ln.focus_make_table()
    with (ROOT / "data/raw/india_makes.csv").open(encoding="utf-8") as fh:
        makes = {r["make"] for r in csv.DictReader(fh)}
    assert set(table["make"]) == makes
    assert (table["maketxt"] == table["maketxt"].str.upper().str.strip()).all()


def test_focus_make_table_refuses_unmapped_make(tmp_path: Path):
    csv_path = tmp_path / "makes.csv"
    csv_path.write_text("make,segment\nTata,mass_market\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Tata"):
        ln.focus_make_table(csv_path)


# ---------- run_sql ----------

def test_run_sql_executes_named_file(tmp_path: Path, monkeypatch):
    (tmp_path / "demo.sql").write_text("CREATE TABLE demo AS SELECT 42 AS x;",
                                       encoding="utf-8")
    monkeypatch.setattr(ln, "SQL_DIR", tmp_path)
    con = duckdb.connect()
    ln.run_sql(con, "demo")
    assert con.execute("SELECT x FROM demo").fetchone() == (42,)


# ---------- build + record_issues on synthetic data ----------

def test_build_filters_makes_dates_and_lookalikes(synthetic_db):
    db, counts = synthetic_db
    assert counts["nhtsa_raw"] == 8
    con = duckdb.connect(str(db), read_only=True)
    kept = {r[0] for r in con.execute("SELECT RECORD_ID FROM nhtsa_focus").fetchall()}
    # Toyota x3 and the case-variant Mercedes are kept; the trailer maker,
    # the 2014 Honda, Ford and Lexus are not.
    assert kept == {"1", "2", "3", "4"}
    assert con.execute(
        "SELECT make FROM nhtsa_focus WHERE RECORD_ID = '4'").fetchone() == ("Mercedes-Benz",)


def test_build_keeps_unknown_year_null(synthetic_db):
    db, _ = synthetic_db
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute(
        "SELECT model_year FROM nhtsa_focus WHERE RECORD_ID = '3'").fetchone() == (None,)
    assert con.execute(
        "SELECT MFGCAMPNO FROM nhtsa_raw WHERE RECORD_ID = '1'").fetchone() == (None,)


def test_build_campaign_lists_models_and_years(synthetic_db):
    db, counts = synthetic_db
    assert counts["nhtsa_campaigns"] == 2
    con = duckdb.connect(str(db), read_only=True)
    models, years, n, unknown = con.execute(
        "SELECT models, model_years, n_records, n_records_unknown_year "
        "FROM nhtsa_campaigns WHERE CAMPNO = '20V001000'").fetchone()
    assert models == ["CAMRY", "RAV4"]
    assert years == [2019, 2020]  # the 9999 record is counted, not guessed
    assert (n, unknown) == (3, 1)


def test_record_issues_counts_unknown_years(synthetic_db):
    db, _ = synthetic_db
    con = duckdb.connect(str(db))
    issues = ln.record_issues(con)
    year_issue = issues[issues["issue"].str.startswith("YEARTXT 9999")]
    assert year_issue["n_rows"].tolist() == [1]
    assert con.execute("SELECT count(*) FROM load_issues").fetchone()[0] == len(issues)


def test_build_never_writes_into_data_raw(synthetic_zip, tmp_path):
    before = sorted(p.name for p in (ROOT / "data/raw").iterdir())
    ln.build(db_path=tmp_path / "x.duckdb", zip_path=synthetic_zip)
    assert sorted(p.name for p in (ROOT / "data/raw").iterdir()) == before


# ---------- main ----------

def test_main_prints_row_counts(monkeypatch, capsys):
    monkeypatch.setattr(ln, "build", lambda: {"nhtsa_raw": 10, "nhtsa_focus": 3})
    ln.main()
    out = capsys.readouterr().out
    assert "nhtsa_raw" in out and "10 rows" in out
    assert "nhtsa_focus" in out


# ---------- the requested checks, on the real NHTSA file ----------

@pytest.fixture(scope="module")
def real_db(tmp_path_factory) -> duckdb.DuckDBPyConnection:
    if not ln.ZIP_PATH.exists():
        pytest.skip("NHTSA zip not downloaded (gitignored)")
    db = tmp_path_factory.mktemp("real") / "recalls.duckdb"
    ln.build(db_path=db)
    con = duckdb.connect(str(db), read_only=True)
    yield con
    con.close()


def test_real_no_other_makes_remain(real_db):
    with (ROOT / "data/raw/india_makes.csv").open(encoding="utf-8") as fh:
        picked = {r["make"] for r in csv.DictReader(fh)}
    makes = {r[0] for r in real_db.execute("SELECT DISTINCT make FROM nhtsa_focus").fetchall()}
    assert makes == picked
    allowed = {a for spellings in ln.MAKETXT_ALIASES.values() for a in spellings}
    spellings = {r[0] for r in real_db.execute(
        "SELECT DISTINCT upper(trim(MAKETXT)) FROM nhtsa_focus").fetchall()}
    assert spellings <= allowed, f"unexpected MAKETXT: {spellings - allowed}"


def test_real_no_report_dates_before_2015(real_db):
    assert real_db.execute(
        "SELECT count(*) FROM nhtsa_focus "
        "WHERE report_date IS NULL OR report_date < DATE '2015-01-01'").fetchone()[0] == 0
    assert real_db.execute(
        "SELECT count(*) FROM nhtsa_campaigns "
        "WHERE report_date < DATE '2015-01-01'").fetchone()[0] == 0


def test_real_campaign_numbers_unique(real_db):
    total, distinct = real_db.execute(
        "SELECT count(*), count(DISTINCT CAMPNO) FROM nhtsa_campaigns").fetchone()
    assert total == distinct > 0
    # and every campaign in nhtsa_focus made it into nhtsa_campaigns
    assert distinct == real_db.execute(
        "SELECT count(DISTINCT CAMPNO) FROM nhtsa_focus").fetchone()[0]
