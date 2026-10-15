"""Tests for src/model_family.py.

Splitting and Mercedes code-expansion tests use exact SIAM model strings
(csv_row noted) together with the committed vocabulary and code table. The
LLM is always replaced by a fake: no test makes a network call.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import model_family as fam

VOCAB = fam.load_vocab()
_codes_df = pd.read_csv(fam.CODES_CSV, dtype=str, keep_default_na=False)
CODES = _codes_df.groupby("platform_code").model_name.apply(list).to_dict()


def _families(csv_row: int, make: str, model: str) -> set[str]:
    siam = pd.DataFrame({"csv_row": [csv_row], "make": [make], "model": [model],
                         "source_url": ["https://www.siam.in/siam-voluntary-recall.aspx"]})
    rows, _zero = fam.siam_families(siam, VOCAB, CODES)
    return set(rows.family)


# ---------- SIAM splitting, real rows ----------

@pytest.mark.parametrize("csv_row, make, model, expected", [
    (2, "BMW", "BMW 1/3 Series, BMW X1/3/5/6/, M3",
     {"1 SERIES", "3 SERIES", "X1", "X3", "X5", "X6"}),
    (6, "BMW", "BMW 2 & 5 Series, X3 and Z", {"2 SERIES", "5 SERIES", "X3"}),
    (32, "HONDA", "Amaze; City; WR-V; Jazz; Civic; BR-V; CR-V",
     {"AMAZE", "CITY", "WR-V", "JAZZ", "CIVIC", "BR-V", "CR-V"}),
    (293, "VOLKSWAGEN", "Polo, Vento, Jetta, Pasat", {"POLO", "VENTO", "JETTA", "PASAT"}),
    (354, "VOLVO", "S90; V90 Cross Country; XC60; XC90 & XC40",
     {"S90", "V90", "XC60", "XC90", "XC40"}),
    (85, "HYUNDAI", "Grand i10 (Non ABS) & Xcent (Non ABS)", {"GRAND I10", "XCENT"}),
])
def test_siam_splitting_real_rows(csv_row, make, model, expected):
    assert _families(csv_row, make, model) == expected


@pytest.mark.parametrize("csv_row, make, model", [
    (252, "AUDI", "Audi"),
    (262, "VOLKSWAGEN", "Volkswagen 4 Wheeler"),
])
def test_siam_rows_with_no_model_give_zero_families(csv_row, make, model):
    siam = pd.DataFrame({"csv_row": [csv_row], "make": [make], "model": [model],
                         "source_url": ["u"]})
    rows, zero = fam.siam_families(siam, VOCAB, CODES)
    assert rows.empty and zero.csv_row.tolist() == [csv_row]


# ---------- Mercedes platform-code expansion, real rows ----------

@pytest.mark.parametrize("csv_row, model, expected", [
    (90, "GLE (167); GLC (253); E - Class (213); C - Class (205); S - Class (223)",
     {"GLE", "GLC", "E-CLASS", "C-CLASS", "S-CLASS"}),
    (165, "Model 167; GLE,GLS", {"GLE", "GLS"}),            # bare code expands via the table
    (210, "Model 205 : C220D and Model 222 : S350D", {"C-CLASS", "S-CLASS"}),
    (92, "CLE (236), AMG CLE (236), AMG C - Class (206)", {"CLE", "C-CLASS"}),
    # 6-digit model numbers are not cut to codes; the stated names still count
    (93, "AMG C - Class (206087) ; AMG GLC (254387) ; GLC (254605)", {"C-CLASS", "GLC"}),
])
def test_mercedes_code_expansion_real_rows(csv_row, model, expected):
    assert _families(csv_row, "MERCEDES-BENZ", model) == expected


def test_bare_code_alone_uses_code_table():
    # code 167 in the reference table lists GLE and GLS (and the SIAM typo AMD GLE)
    assert _families(999, "MERCEDES-BENZ", "Model 167") == {"GLE", "GLS"}


# ---------- load_vocab / rule_map_nhtsa ----------

def test_load_vocab_is_sorted_per_make():
    assert VOCAB["BMW"] == sorted(VOCAB["BMW"])
    assert "3 SERIES" in VOCAB["BMW"] and "CR-V" in VOCAB["HONDA"]


def test_rule_map_marks_rule_or_unmapped():
    models = pd.DataFrame({"make": ["BMW", "HONDA", "HONDA"],
                           "MODELTXT": ["330I", "CR-V HYBRID", "PILOT"]})
    out = fam.rule_map_nhtsa(models, VOCAB)
    assert out.family.tolist()[:2] == ["3 SERIES", "CR-V"]
    assert out.method.tolist()[:2] == ["rule", "rule"] and pd.isna(out.method.iloc[2])


# ---------- build_prompt / parse_llm_answer ----------

def test_prompt_lists_allowed_families_and_names():
    p = fam.build_prompt("NISSAN", ["ROGUE", "ALTIMA"], ["MICRA", "X-TRAIL"])
    assert "MICRA, X-TRAIL" in p and "- ROGUE" in p and '"none"' in p


def test_parse_llm_answer_validates_every_value():
    allowed = ["MICRA", "X-TRAIL"]
    answer = {"ROGUE": "x-trail", "TITAN": "none", "ALTIMA": "TEANA"}
    got = fam.parse_llm_answer(answer, ["ROGUE", "TITAN", "ALTIMA", "LEAF"], allowed)
    assert [g[1] for g in got] == ["X-TRAIL", None, None, None]
    assert "not in the allowed list" in got[2][2]   # invented family rejected
    assert "no answer" in got[3][2]


def test_parse_llm_answer_handles_non_dict():
    assert fam.parse_llm_answer(["junk"], ["A"], ["X"])[0][1] is None


# ---------- llm_map batching ----------

def test_llm_map_batches_40_per_call_one_make_per_call():
    calls = []

    def fake(prompt, **kw):
        calls.append(prompt)
        return {}

    unmapped = pd.DataFrame({"make": ["BMW"] * 85 + ["KIA"] * 3,
                             "MODELTXT": [f"M{i}" for i in range(85)] + ["K1", "K2", "K3"]})
    out = fam.llm_map(unmapped, {"BMW": ["X1"], "KIA": ["EV6"]}, ask=fake)
    assert len(calls) == 4                      # 40 + 40 + 5 for BMW, 3 for KIA
    assert all(p.startswith("Make: BMW") for p in calls[:3])
    assert len(out) == 88 and set(out.method) == {"llm"}


# ---------- overrides ----------

def test_overrides_apply_last(tmp_path: Path):
    path = tmp_path / "o.csv"
    pd.DataFrame([["nhtsa", "HONDA", "PILOT", "none", ""],
                  ["nhtsa", "NISSAN", "ROGUE", "X-Trail", "sourced"],
                  ["siam", "AUDI", "252", "A6", "checked page"]],
                 columns=fam.OVERRIDE_COLUMNS).to_csv(path, index=False)
    nhtsa = pd.DataFrame({"make": ["HONDA", "NISSAN"], "MODELTXT": ["PILOT", "ROGUE"],
                          "family": ["CR-V", None], "method": ["llm", "llm"]})
    siam = pd.DataFrame({"csv_row": [1], "make": ["HONDA"], "family": ["CITY"], "method": ["rule"]})
    n, s = fam.apply_overrides(nhtsa, siam, fam.load_overrides(path),
                               {"NISSAN": ["X-TRAIL"], "AUDI": ["A6"], "HONDA": ["CR-V"]})
    assert pd.isna(n.family.iloc[0]) and n.family.iloc[1] == "X-TRAIL"
    assert set(n.method) == {"override"}
    assert s[s.csv_row == 252].values.tolist() == [[252, "AUDI", "A6", "override"]]


def test_override_outside_vocabulary_is_refused(tmp_path: Path):
    path = tmp_path / "o.csv"
    pd.DataFrame([["nhtsa", "NISSAN", "ROGUE", "ROGUE", ""]],
                 columns=fam.OVERRIDE_COLUMNS).to_csv(path, index=False)
    nhtsa = pd.DataFrame({"make": ["NISSAN"], "MODELTXT": ["ROGUE"],
                          "family": [None], "method": ["llm"]})
    with pytest.raises(ValueError, match="not in the vocabulary"):
        fam.apply_overrides(nhtsa, nhtsa.iloc[0:0], fam.load_overrides(path),
                            {"NISSAN": ["X-TRAIL"]})


def test_load_overrides_missing_file_and_bad_side(tmp_path: Path):
    assert fam.load_overrides(tmp_path / "none.csv").empty
    bad = tmp_path / "bad.csv"
    pd.DataFrame([["usa", "HONDA", "PILOT", "", ""]],
                 columns=fam.OVERRIDE_COLUMNS).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="side must be"):
        fam.load_overrides(bad)


# ---------- review_frame ----------

def test_review_frame_has_llm_rows_zero_rows_and_empty_confirmed():
    llm_rows = pd.DataFrame({"make": ["NISSAN"], "MODELTXT": ["ROGUE"], "family": ["X-TRAIL"],
                             "method": ["llm"], "note": ["llm choice from allowed list"]})
    zero = pd.DataFrame({"csv_row": [252], "make": ["AUDI"], "model": ["Audi"]})
    r = fam.review_frame(llm_rows, zero)
    assert r.side.tolist() == ["nhtsa", "siam"]
    assert (r.confirmed == "").all()


# ---------- sql_statements ----------

def test_sql_statements_ignores_semicolons_in_comments(tmp_path: Path):
    f = tmp_path / "q.sql"
    f.write_text("-- note; with a semicolon\nSELECT 1;\n-- two\nSELECT 2;\n", encoding="utf-8")
    assert fam.sql_statements(f) == ["SELECT 1", "SELECT 2"]
    assert len(fam.sql_statements(fam.SQL_DIR / "model_family_report.sql")) == 2


# ---------- build + main ----------

def test_build_end_to_end_with_fake_llm(tmp_path: Path):
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE nhtsa_focus AS SELECT * FROM (VALUES
        ('A', 'HONDA', 'CR-V HYBRID'), ('A', 'HONDA', 'CR-V HYBRID'),
        ('B', 'HONDA', 'PILOT'), ('C', 'NISSAN', 'ROGUE')) AS t(CAMPNO, make, MODELTXT)""")
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'HONDA', 'CR-V; City', 'u'), (2, 'AUDI', 'Audi', 'u'))
        AS t(csv_row, make, model, source_url)""")
    con.execute("CREATE TABLE nhtsa_model_families AS SELECT 1 AS x")  # old-version table
    con.close()
    vocab_csv = tmp_path / "v.csv"
    pd.DataFrame({"make": ["HONDA", "HONDA", "NISSAN", "AUDI"],
                  "family": ["CR-V", "CITY", "X-TRAIL", "A6"],
                  "source": "x", "evidence": "", "source_url": ""}).to_csv(vocab_csv, index=False)

    def fake(prompt, **kw):
        return {"PILOT": "none"} if "Make: HONDA" in prompt else {"ROGUE": "X-TRAIL"}

    out = fam.build(db_path=db, vocab_csv=vocab_csv, aliases_csv=tmp_path / "a.csv",
                    overrides_csv=tmp_path / "missing.csv",
                    review_csv=tmp_path / "review.csv", ask=fake)
    con = duckdb.connect(str(db), read_only=True)
    got = dict(con.execute("SELECT MODELTXT, method FROM nhtsa_model_family").fetchall())
    assert got == {"CR-V HYBRID": "rule", "PILOT": "llm", "ROGUE": "llm"}
    assert set(con.execute("SELECT family FROM siam_row_family WHERE csv_row = 1"
                           ).fetchall()) == {("CR-V",), ("CITY",)}
    review = pd.read_csv(tmp_path / "review.csv", keep_default_na=False)
    assert set(review.side) == {"nhtsa", "siam"} and "confirmed" in review.columns
    total = out["share"][out["share"].make.isna()].iloc[0]
    assert (total.records, total.with_family) == (4, 3)   # records, not distinct names
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    assert not tables & set(fam.OLD_TABLES), "old-version tables must be dropped"


def test_build_refuses_without_vocabulary(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="--vocab"):
        fam.build(vocab_csv=tmp_path / "absent.csv")


# ---------- write_vocabulary ----------

def test_write_vocabulary_writes_vocab_review_and_alias_template(tmp_path: Path):
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'VOLKSWAGEN', 'Polo, Vento, Jetta, Pasat', 'u')) AS t(csv_row, make, model, source_url)""")
    con.execute("""CREATE TABLE nhtsa_focus AS SELECT * FROM (VALUES
        ('A', 'VOLKSWAGEN', 'PASSAT')) AS t(CAMPNO, make, MODELTXT)""")
    con.close()
    india = tmp_path / "india.csv"
    pd.DataFrame({"make": ["Volkswagen"], "model": ["Virtus"],
                  "source_url": ["https://www.volkswagen.co.in/"]}).to_csv(india, index=False)
    codes = tmp_path / "codes.csv"
    pd.DataFrame(columns=["platform_code", "model_name"]).to_csv(codes, index=False)

    n_fam, n_review = fam.write_vocabulary(
        db_path=db, india_csv=india, codes_csv=codes, vocab_csv=tmp_path / "ref" / "v.csv",
        review_csv=tmp_path / "clean" / "r.csv", aliases_csv=tmp_path / "ref" / "a.csv")
    vocab = pd.read_csv(tmp_path / "ref" / "v.csv")
    assert set(vocab.family) == {"VIRTUS", "POLO", "VENTO", "JETTA", "PASAT"} and n_fam == 5
    review = pd.read_csv(tmp_path / "clean" / "r.csv")
    assert "PASSAT" in review.review_note.iloc[0] and n_review == 1
    assert list(pd.read_csv(tmp_path / "ref" / "a.csv").columns) == fam.ALIAS_COLUMNS


# ---------- main ----------

def test_main_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(fam, "build", lambda: {
        "by_method": pd.DataFrame({"method": ["rule"], "n": [1]}),
        "share": pd.DataFrame({"make": ["BMW"], "pct": [50.0]})})
    fam.main([])
    out = capsys.readouterr().out
    assert "Mappings by method" in out and "BMW" in out


def test_main_vocab_flag_only_rebuilds_vocabulary(monkeypatch, capsys):
    monkeypatch.setattr(fam, "write_vocabulary", lambda: (167, 18))
    monkeypatch.setattr(fam, "build", lambda: pytest.fail("build must not run with --vocab"))
    fam.main(["--vocab"])
    assert "167 families" in capsys.readouterr().out
