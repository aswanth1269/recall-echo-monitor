"""Tests for src/load_siam.py: date parsing, make values, cleaning and the load."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import load_siam as ls
from src.load_nhtsa import MAKETXT_ALIASES

ROOT = Path(__file__).resolve().parents[1]


# ---------- parse_date: formats that must parse ----------

@pytest.mark.parametrize("text, expected", [
    ("15-Mar-2022", date(2022, 3, 15)),       # voluntary_recall_date format
    ("02-Jul-18", date(2018, 7, 2)),          # two-digit year
    ("1-Jan-19", date(2019, 1, 1)),
    ("07 Dec 2007", date(2007, 12, 7)),
    ("01-June-2015", date(2015, 6, 1)),       # full month name
    ("12 September 2016", date(2016, 9, 12)),
    ("3 Sept 2016", date(2016, 9, 3)),
    ("1st Jan 2019", date(2019, 1, 1)),       # ordinal suffix
    ("21st November 2019", date(2019, 11, 21)),
    ("05Aug 2017", date(2017, 8, 5)),         # no space after the day
    ("1-Jan -2019", date(2019, 1, 1)),        # stray space
    ("24/10/2019", date(2019, 10, 24)),       # day > 12, so day-first is certain
    ("13-01-18", date(2018, 1, 13)),
    ("01/25/2020", date(2020, 1, 25)),        # month-first, but only one reading works
    ("05/05/2019", date(2019, 5, 5)),         # day == month: both readings agree
    ("  15-Mar-2022  ", date(2022, 3, 15)),
])
def test_parse_date_accepts_unambiguous_dates(text, expected):
    assert ls.parse_date(text) == (expected, None)


# ---------- parse_date: cells that must NOT be guessed ----------

@pytest.mark.parametrize("text, reason", [
    ("03/04/2021", "ambiguous day/month order"),
    ("01-12-09", "ambiguous day/month order"),
    ("30/15/2015", "not a real date"),        # real typo on the SIAM page
    ("31-Feb-2020", "not a real date"),
    ("Aug 2017", "unrecognised format"),
    ("2015", "unrecognised format"),
    ("Jul'19", "unrecognised format"),
    ("02-Jul-18; 05-Oct-18", "several dates"),
    ("03-Aug-17 & 12-Apr-18", "several dates"),
    ("CLA (117) - 9-Jan-15", "several dates"),
    ("01-Foo-2019", "unknown month name"),
    ("", "blank"),
    (None, "blank"),
])
def test_parse_date_refuses_to_guess(text, reason):
    value, why = ls.parse_date(text)
    assert value is None
    assert why.startswith(reason)


# ---------- make values ----------

def test_standardise_make_uses_nhtsa_spellings():
    assert ls.standardise_make("Mercedes-Benz") == "MERCEDES-BENZ"
    assert ls.standardise_make("Volkswagen") == "VOLKSWAGEN"
    assert ls.standardise_make("  BMW ") == "BMW"


def test_standardise_make_rejects_unknown_make():
    with pytest.raises(ValueError, match="Tata"):
        ls.standardise_make("Tata")


# ---------- clean_text ----------

@pytest.mark.parametrize("raw, expected", [
    ("  Audi   Q7 ", "Audi Q7"),
    ("a\t\nb", "a b"),
    ("", None),
    ("   ", None),
    (None, None),
    (float("nan"), None),
])
def test_clean_text(raw, expected):
    assert ls.clean_text(raw) == expected


# ---------- clean_siam ----------

def _frame(**overrides):
    base = {"make": "Audi", "brands_detected": "Audi", "make_resolution": "model_token",
            "manufacturer_name": "Skoda  Auto ",
            "voluntary_recall_date": "15-Mar-2022", "model": " Audi  Q7",
            "variant": "", "status": "", "target_recall_vehicles": "195",
            "manufactured_from": "03/04/2021", "manufactured_to": "24/10/2019",
            "defect_description": "x", "source_url": "u", "retrieved_on": "2026-10-13"}
    base.update(overrides)
    return pd.DataFrame([base])


def test_clean_siam_strips_spaces_and_sets_makes():
    clean, _ = ls.clean_siam(_frame())
    row = clean.iloc[0]
    assert row["make"] == "AUDI" and row["make_label"] == "Audi"
    assert row["model"] == "Audi Q7"
    assert row["manufacturer_name"] == "Skoda Auto"
    assert row["variant"] is None and row["status"] is None
    assert row["target_recall_vehicles"] == 195


def test_clean_siam_logs_ambiguous_and_keeps_raw_text():
    clean, issues = ls.clean_siam(_frame())
    row = clean.iloc[0]
    assert row["recall_date"] == date(2022, 3, 15)
    assert row["manufactured_from_date"] is None          # 03/04/2021 is ambiguous
    assert row["manufactured_from"] == "03/04/2021"        # raw text survives
    assert row["manufactured_to_date"] == date(2019, 10, 24)
    reasons = dict(zip(issues["column_name"], issues["reason"]))
    assert reasons["manufactured_from"] == "ambiguous day/month order"


def test_clean_siam_nulls_dates_outside_plausible_range():
    clean, issues = ls.clean_siam(_frame(manufactured_from="01-Jan-1985",
                                         manufactured_to="01-Jan-2023"))
    row = clean.iloc[0]
    assert row["manufactured_from_date"] is None   # before 1990
    assert row["manufactured_to_date"] is None     # after the 2022 recall
    assert (issues["reason"] == "parsed but outside 1990..recall date").sum() == 2


# ---------- build + main ----------

def test_build_writes_tables(tmp_path: Path):
    db = tmp_path / "recalls.duckdb"
    rows, n_issues = ls.build(db_path=db)
    con = duckdb.connect(str(db), read_only=True)
    assert rows == con.execute("SELECT count(*) FROM siam_recalls").fetchone()[0] == 355
    assert n_issues == con.execute("SELECT count(*) FROM siam_date_issues").fetchone()[0]
    types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns "
                             "WHERE table_name = 'siam_recalls'").fetchall())
    assert types["recall_date"] == "DATE" and types["manufactured_from_date"] == "DATE"


def test_build_every_make_is_an_nhtsa_spelling(tmp_path: Path):
    db = tmp_path / "recalls.duckdb"
    ls.build(db_path=db)
    con = duckdb.connect(str(db), read_only=True)
    makes = {r[0] for r in con.execute("SELECT DISTINCT make FROM siam_recalls").fetchall()}
    nhtsa = {spellings[0] for spellings in MAKETXT_ALIASES.values()}
    assert makes == nhtsa
    assert con.execute("SELECT count(*) FROM siam_recalls WHERE recall_date IS NULL"
                       ).fetchone()[0] == 0, "every voluntary_recall_date should parse"


def test_build_reads_without_touching_data_raw(tmp_path: Path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "data/raw").iterdir()}
    ls.build(db_path=tmp_path / "x.duckdb")
    after = {p.name: p.stat().st_mtime for p in (ROOT / "data/raw").iterdir()}
    assert before == after


def test_main_reports_counts(monkeypatch, capsys):
    monkeypatch.setattr(ls, "build", lambda: (355, 12))
    ls.main()
    out = capsys.readouterr().out
    assert "355 rows" in out and "12 rows" in out
