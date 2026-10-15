"""Tests for src/mercedes_codes.py, using the real shapes SIAM's Mercedes rows take."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import mercedes_codes as mc


# ---------- split_segments ----------

def test_split_keeps_commas_inside_brackets():
    assert mc.split_segments("167 (GLE, GLS)") == ["167 (GLE, GLS)"]


@pytest.mark.parametrize("text, expected", [
    ("177 (A-Class); 247 (GLA)", ["177 (A-Class)", "247 (GLA)"]),
    ("CLE (236), AMG CLE (236)", ["CLE (236)", "AMG CLE (236)"]),
    ("G63 AMG (463 Platform) & E63 AMG(213 Platform)",
     ["G63 AMG (463 Platform)", "E63 AMG(213 Platform)"]),
    ("GLE and GLS (167)", ["GLE", "GLS (167)"]),
    ("S - Class \\ Maybach S - Class (223)", ["S - Class", "Maybach S - Class (223)"]),
    ("Model 205 : C220D and Model 222 : S350D", ["Model 205 : C220D", "Model 222 : S350D"]),
    ("", []),
    (None, []),
])
def test_split_segments(text, expected):
    assert mc.split_segments(text) == expected


# ---------- normalise_name ----------

@pytest.mark.parametrize("raw", ["C-Class", "C - Class", "C- Class", "C Class",
                                 "C – Class",   # en dash, as on the SIAM page
                                 "C � Class", "c-class"])
def test_normalise_name_merges_spelling_variants(raw):
    assert mc.normalise_name(raw) == "C-CLASS"


def test_normalise_name_keeps_source_typos():
    assert mc.normalise_name("AMD GLE") == "AMD GLE"
    assert mc.normalise_name("E-CLASS0") == "E-CLASS0"


# ---------- parse_segment ----------

@pytest.mark.parametrize("segment, expected", [
    ("205 (C-Class)", [("205", "C-CLASS")]),
    ("Model 206 (C 220 d)", [("206", "C 220 D")]),
    ("Model 177(AMG A)", [("177", "AMG A")]),
    ("167 (GLE, GLS)", [("167", "GLE"), ("167", "GLS")]),
    ("Model 166 (ML/GLE)", [("166", "ML"), ("166", "GLE")]),
    ("C � Class (205)", [("205", "C-CLASS")]),
    ("GLC (253 platform)", [("253", "GLC")]),
    ("C-Class ( Model 203)", [("203", "C-CLASS")]),
    ("AMGGTS (W190)", [("190", "AMGGTS")]),
    ("SLK (model 171)", [("171", "SLK")]),
    ("VAN V220DL/ V220D (447 Vehicle Platform)", [("447", "VAN V220DL/ V220D")]),
    ("Model 205 : C220D", [("205", "C220D")]),
])
def test_parse_segment_reads_stated_pairs(segment, expected):
    assert mc.parse_segment(segment) == expected


@pytest.mark.parametrize("segment", [
    "Model 167",                 # code with no name beside it
    "AMG C - Class (206087)",    # 6-digit model number: not cut to 206
    "Viano (639813)",
    "C-Class",                   # name only
    "AMGG63",
    "S560XL (Maybach S 560)",    # no platform code at all
    "167 (GLE, GLS), 205 (C-CLASS), 213 (E-CLASS0, 223 (S-CLASS, 253 (GLC)",  # broken brackets
    "213 (E-CLASS0, 223 (S-CLASS, 253 (GLC)",
])
def test_parse_segment_does_not_guess(segment):
    assert mc.parse_segment(segment) == []


# ---------- extract_pairs ----------

def _rows(*models):
    return pd.DataFrame({
        "csv_row": list(range(1, len(models) + 1)),
        "model": list(models),
        "source_url": "https://www.siam.in/siam-voluntary-recall.aspx",
        "retrieved_on": "2026-10-13",
    })


def test_extract_pairs_dedupes_and_keeps_provenance():
    pairs, _ = mc.extract_pairs(_rows("205 (C-Class)", "C - Class (205 Platform)",
                                      "S-Class(223), S-Class(223)"))
    c = pairs[(pairs.platform_code == "205") & (pairs.model_name == "C-CLASS")].iloc[0]
    assert c.n_siam_rows == 2 and c.siam_csv_rows == "1;2"
    assert c.source_text == "205 (C-Class)"
    s = pairs[pairs.platform_code == "223"].iloc[0]
    assert s.n_siam_rows == 1  # the same pair twice in one row counts once


def test_extract_pairs_never_puts_brackets_in_names():
    pairs, issues = mc.extract_pairs(_rows(
        "167 (GLE, GLS), 205 (C-CLASS), 213 (E-CLASS0, 223 (S-CLASS, 253 (GLC)"))
    assert not pairs.model_name.str.contains(r"[()]").any()
    assert (issues.reason == "unbalanced brackets in source text").any()


def test_extract_pairs_logs_codes_it_could_not_map():
    _, issues = mc.extract_pairs(_rows("Model 167; GLE,GLS", "AMG C - Class (206087)",
                                       "CLA; GLA"))
    reasons = dict(zip(issues.segment, issues.reason))
    assert reasons["Model 167"].startswith("3-digit number with no model name")
    assert reasons["AMG C - Class (206087)"].startswith("number longer than 3 digits")
    assert "CLA" not in reasons  # name-only segments are not issues


# ---------- build + main ----------

def test_build_reads_only_mercedes_and_writes_both_csvs(tmp_path: Path):
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'MERCEDES-BENZ', '205 (C-Class)', 'u', DATE '2026-10-13'),
        (2, 'MERCEDES-BENZ', 'Model 167', 'u', DATE '2026-10-13'),
        (3, 'BMW', 'X3 (999)', 'u', DATE '2026-10-13'))
        AS t(csv_row, make, model, source_url, retrieved_on)""")
    con.close()

    out, issues = tmp_path / "ref" / "codes.csv", tmp_path / "clean" / "issues.csv"
    assert mc.build(db, out, issues) == (1, 1)
    ref = pd.read_csv(out, dtype=str)
    assert list(ref.columns) == ["platform_code", "model_name", "n_siam_rows",
                                 "siam_csv_rows", "source_text", "source_url", "retrieved_on"]
    assert ref.platform_code.str.fullmatch(r"\d{3}").all()
    assert ref.siam_csv_rows.notna().all()
    assert "999" not in set(ref.platform_code)  # BMW row ignored via make


def test_main_reports_counts(monkeypatch, capsys):
    monkeypatch.setattr(mc, "build", lambda: (40, 7))
    mc.main()
    out = capsys.readouterr().out
    assert "40 code-name pairs" in out and "7 unparsed segments" in out
