"""Tests for src/make_label_sheet.py (synthetic pairs; no real database needed)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import make_label_sheet as ls


def _pairs(n: int = 90) -> pd.DataFrame:
    return pd.DataFrame({
        "CAMPNO": [f"{20 + i % 7}V{i:03d}000" for i in range(n)],
        "make": "HONDA", "csv_row": range(1, n + 1),
        "score": [round(1 - i / n, 4) if i < 60 else 0.0 for i in range(n)],  # ties at 0
        "rank": 1, "us_report_date": date(2020, 1, 1), "us_component": "AIR BAGS",
        "us_defect_text": "Inflator may rupture", "siam_date": date(2020, 6, 1),
        "siam_model": "City", "siam_defect_text": "Airbag inflator may rupture"})


# ---------- assign_thirds ----------

def test_thirds_are_equal_and_ordered_by_score():
    banded = ls.assign_thirds(_pairs())
    assert banded.band.value_counts().to_dict() == {"top": 30, "middle": 30, "bottom": 30}
    assert banded[banded.band == "top"].score.min() >= banded[banded.band == "middle"].score.max()
    assert banded[banded.band == "middle"].score.min() >= banded[banded.band == "bottom"].score.max()


def test_thirds_are_deterministic_despite_tied_scores():
    a = ls.assign_thirds(_pairs())
    b = ls.assign_thirds(_pairs().sample(frac=1, random_state=1))  # input order shuffled
    assert a[["CAMPNO", "csv_row", "band"]].equals(b[["CAMPNO", "csv_row", "band"]])


# ---------- sample_pairs ----------

def test_sample_takes_ten_per_third():
    sample = ls.sample_pairs(_pairs())
    assert len(sample) == 30
    assert sample.band.value_counts().to_dict() == {"top": 10, "middle": 10, "bottom": 10}
    assert sample[["CAMPNO", "csv_row"]].duplicated().sum() == 0


def test_sample_is_reproducible_with_the_seed():
    a = ls.sample_pairs(_pairs(), seed=7)
    assert a.csv_row.tolist() == ls.sample_pairs(_pairs(), seed=7).csv_row.tolist()
    assert a.csv_row.tolist() != ls.sample_pairs(_pairs(), seed=8).csv_row.tolist()


def test_sample_order_does_not_reveal_the_band():
    bands = ls.sample_pairs(_pairs()).band.tolist()
    assert bands[:10] != ["top"] * 10                      # not written band by band
    assert sum(a != b for a, b in zip(bands, bands[1:])) > 5


def test_sample_refuses_when_a_third_is_too_small():
    with pytest.raises(ValueError, match="need 10"):
        ls.sample_pairs(_pairs(20))


# ---------- pair_id / to_sheet ----------

def test_pair_id_maps_back_to_candidate_pairs():
    assert ls.pair_id("20V543000", 81) == "20V543000-81"


def test_sheet_has_requested_columns_and_no_score():
    sheet, key = ls.to_sheet(ls.sample_pairs(_pairs()))
    assert list(sheet.columns) == ls.SHEET_COLUMNS
    assert not {"score", "rank", "band", "CAMPNO", "csv_row"} & set(sheet.columns)
    assert (sheet.label == "").all()
    assert {"score", "rank", "band"} <= set(key.columns)
    assert sheet.pair_id.tolist() == key.pair_id.tolist()


# ---------- has_labels / build / main ----------

def _db(tmp_path: Path) -> Path:
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    p = _pairs()
    con.register("p", p)
    con.execute("""CREATE TABLE candidate_pairs AS
        SELECT CAMPNO, make, csv_row, us_report_date AS report_date, score, rank FROM p""")
    con.execute("""CREATE TABLE india_relevant_campaigns AS
        SELECT DISTINCT CAMPNO, make, ['AIR BAGS', 'SEAT BELTS'] AS components,
               us_defect_text AS DESC_DEFECT FROM p""")
    con.execute("""CREATE TABLE siam_recalls AS
        SELECT csv_row, siam_date AS recall_date, siam_model AS model,
               siam_defect_text AS defect_description FROM p""")
    con.close()
    return db


def test_build_writes_sheet_and_key(tmp_path: Path):
    sheet_csv, key_csv = tmp_path / "sheet.csv", tmp_path / "key.csv"
    key = ls.build(db_path=_db(tmp_path), sheet_csv=sheet_csv, key_csv=key_csv)
    sheet = pd.read_csv(sheet_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(sheet.columns) == ls.SHEET_COLUMNS and len(sheet) == 30
    assert sheet.us_component.iloc[0] == "AIR BAGS; SEAT BELTS"
    assert "score" not in sheet.columns and len(key) == 30


def test_build_never_overwrites_filled_labels(tmp_path: Path):
    db, sheet_csv, key_csv = _db(tmp_path), tmp_path / "sheet.csv", tmp_path / "key.csv"
    ls.build(db_path=db, sheet_csv=sheet_csv, key_csv=key_csv)
    sheet = pd.read_csv(sheet_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sheet.loc[0, "label"] = "same"
    sheet.to_csv(sheet_csv, index=False, encoding="utf-8-sig")
    assert ls.has_labels(sheet_csv)
    with pytest.raises(FileExistsError, match="already has labels"):
        ls.build(db_path=db, sheet_csv=sheet_csv, key_csv=key_csv)
    ls.build(db_path=db, sheet_csv=sheet_csv, key_csv=key_csv, force=True)
    assert not ls.has_labels(sheet_csv)


def test_has_labels_false_for_missing_or_empty(tmp_path: Path):
    assert not ls.has_labels(tmp_path / "none.csv")
    empty = tmp_path / "e.csv"
    pd.DataFrame({"pair_id": ["a"], "label": [""]}).to_csv(empty, index=False)
    assert not ls.has_labels(empty)


def test_main_reports_and_passes_force(monkeypatch, capsys):
    seen = {}

    def fake_build(force=False):
        seen["force"] = force
        return pd.DataFrame({"band": ["top"] * 10 + ["middle"] * 10 + ["bottom"] * 10})

    monkeypatch.setattr(ls, "build", fake_build)
    ls.main(["--force"])
    out = capsys.readouterr().out
    assert seen["force"] is True and "30 pairs" in out and "10 top" in out
