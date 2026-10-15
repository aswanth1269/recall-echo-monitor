"""Rule tests for src/model_family.py, using model text seen in NHTSA and SIAM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import model_family as mf


# ---------- normalise_model / family_name ----------

@pytest.mark.parametrize("raw, expected", [
    ("Audi 4 Wheeler Audi e-tron GT", "E-TRON GT"),
    ("Grand i10 (Non ABS)", "GRAND I10"),
    ("Santa Fe (DMI)", "SANTA FE"),
    ("GLE (167)", "GLE (167)"),          # brackets holding a code are kept
    ("Volkswagen 4 Wheeler", None),
    ("Toyota  Land  Cruiser", "LAND CRUISER"),
    ("", None),
    (None, None),
])
def test_normalise_model(raw, expected):
    assert mf.normalise_model(raw) == expected


@pytest.mark.parametrize("make, name, expected", [
    ("VOLVO", "XC90 HYBRID", "XC90"),
    ("VOLVO", "S60 CROSS COUNTRY", "S60"),
    ("HYUNDAI", "VENUE N LINE", "VENUE"),
    ("HYUNDAI", "CRETA ELECTRIC", "CRETA"),
    ("KIA", "CARENS CLAVIS EV", "CARENS CLAVIS"),
    ("HYUNDAI", "IONIQ 5", "IONIQ 5"),   # a number is not a variant suffix
    ("BMW", "X5 M", "X5 M"),             # rule makes are left to their rules
])
def test_family_name_drops_variant_suffixes(make, name, expected):
    assert mf.family_name(make, name) == expected


# ---------- expand_siam_shorthand ----------

def test_expand_shorthand_bmw_lists():
    out = mf.expand_siam_shorthand("BMW 1/3 Series, BMW X1/3/5/6/, M3")
    assert "1 Series, 3 Series" in out
    assert "X1, X3, X5, X6" in out
    assert mf.expand_siam_shorthand("BMW 2 & 5 Series, X3 and Z") == \
        "BMW 2 Series, 5 Series, X3 and Z"


def test_expand_shorthand_leaves_plain_text():
    assert mf.expand_siam_shorthand("S60 & S60 Cross Country") == "S60 & S60 Cross Country"


# ---------- reduce_family: make rules (vocab=None means no gating) ----------

@pytest.mark.parametrize("text, family", [
    ("330I", "3 SERIES"), ("740LI", "7 SERIES"), ("M340I", "3 SERIES"),
    ("228I GRAN COUPE", "2 SERIES"), ("840I", "8 SERIES"), ("M3", "3 SERIES"),
    ("M5 COMPETITION", "5 SERIES"), ("X5 M", "X5"), ("X3M", "X3"), ("IX", "IX"),
    ("IX1", "IX1"), ("I4", "I4"), ("Z4", "Z4"), ("3 Series Gran Limousine", "3 SERIES"),
])
def test_bmw_rules(text, family):
    assert mf.reduce_family("BMW", text) == (family, None)


@pytest.mark.parametrize("text, family", [
    ("GLE450", "GLE"), ("GLE 450", "GLE"), ("C 300", "C-CLASS"), ("C300", "C-CLASS"),
    ("AMG C63", "C-CLASS"), ("AMGC63", "C-CLASS"), ("GLS 580 4MATIC", "GLS"),
    ("G 550", "G-CLASS"), ("EQS SUV", "EQS SUV"), ("EQE SUV 500 4MATIC", "EQE SUV"),
    ("SPRINTER 2500", "SPRINTER"), ("METRIS", "METRIS"), ("AMG GT 53", "AMG GT"),
    ("AMGGTS", "AMG GT"), ("MAYBACH S 580", "S-CLASS"), ("S-CLASS", "S-CLASS"),
    ("C – Class", "C-CLASS"), ("GLE COUPE", "GLE"), ("V220D", "V-CLASS"),
])
def test_mercedes_rules(text, family):
    assert mf.reduce_family("MERCEDES-BENZ", text) == (family, None)


@pytest.mark.parametrize("text, family", [
    ("S6", "A6"), ("RS 7", "A7"), ("RS Q8", "Q8"), ("SQ5", "Q5"), ("A4 ALLROAD", "A4"),
    ("RS 6 AVANT", "A6"), ("E-TRON GT", "E-TRON GT"), ("Audi Etron", "E-TRON"),
    ("TT RS", "TT"), ("A3 CABRIOLET", "A3"), ("E TRON GT", "E-TRON GT"),
])
def test_audi_rules(text, family):
    assert mf.reduce_family("AUDI", text) == (family, None)


@pytest.mark.parametrize("make, text", [
    ("BMW", "MINI COOPER S"), ("BMW", "Z"), ("BMW", "E8X E9X"),
    ("MERCEDES-BENZ", "E-CLASS0"), ("MERCEDES-BENZ", "EOS"), ("MERCEDES-BENZ", "VAN"),
    ("AUDI", "Audi"),
])
def test_rules_do_not_guess(make, text):
    family, reason = mf.reduce_family(make, text)
    assert family is None and reason


# ---------- reduce_family: vocabulary gating and prefix makes ----------

VOCAB = {
    "HONDA": {"CR-V", "CITY", "CIVIC"},
    "VOLVO": {"V60", "XC90"},
    "HYUNDAI": {"IONIQ", "IONIQ 5"},
    "BMW": {"X3", "3 SERIES"},
}


@pytest.mark.parametrize("make, text, family", [
    ("HONDA", "CR-V HYBRID", "CR-V"), ("HONDA", "CIVIC TYPE R", "CIVIC"),
    ("VOLVO", "V60CC", "V60"), ("VOLVO", "XC90 HYBRID", "XC90"),
    ("HYUNDAI", "IONIQ5", "IONIQ 5"), ("HYUNDAI", "IONIQ HYBRID", "IONIQ"),
    ("BMW", "330I", "3 SERIES"),
])
def test_prefix_and_gated_matches(make, text, family):
    assert mf.reduce_family(make, text, VOCAB) == (family, None)


@pytest.mark.parametrize("make, text, reason_part", [
    ("HONDA", "GL1800", "no family in the vocabulary"),   # motorcycle
    ("VOLVO", "VNL", "no family in the vocabulary"),      # truck
    ("BMW", "X4", "reduces to X4"),                       # real family, not in India vocab
])
def test_unmatched_get_a_reason(make, text, reason_part):
    family, reason = mf.reduce_family(make, text, VOCAB)
    assert family is None and reason_part in reason


def test_alias_applies_only_exact_names():
    aliases = {("NISSAN", "ROGUE"): "X-TRAIL"}
    vocab = {"NISSAN": {"X-TRAIL"}}
    assert mf.reduce_family("NISSAN", "ROGUE", vocab, aliases) == ("X-TRAIL", None)
    assert mf.reduce_family("NISSAN", "ROGUE SPORT", vocab, aliases)[0] is None


# ---------- load_aliases ----------

def test_load_aliases_requires_a_source(tmp_path: Path):
    good = tmp_path / "a.csv"
    pd.DataFrame([["NISSAN", "Rogue", "X-Trail", "https://example.org", "2026-10-15", ""]],
                 columns=mf.ALIAS_COLUMNS).to_csv(good, index=False)
    assert mf.load_aliases(good) == {("NISSAN", "ROGUE"): "X-TRAIL"}
    bad = tmp_path / "b.csv"
    pd.DataFrame([["NISSAN", "Rogue", "X-Trail", "", "", ""]],
                 columns=mf.ALIAS_COLUMNS).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="without source_url"):
        mf.load_aliases(bad)
    assert mf.load_aliases(tmp_path / "missing.csv") == {}


# ---------- siam_segments ----------

def test_siam_segments_bmw_shorthand():
    names = {n for _, n, _ in mf.siam_segments("BMW", "BMW 1/3 Series, BMW X1/3/5/6/, M3", {})}
    fams = {mf.reduce_family("BMW", n)[0] for n in names}
    assert fams == {"1 SERIES", "3 SERIES", "X1", "X3", "X5", "X6"}


def test_siam_segments_mercedes_codes():
    codes = {"167": ["GLE", "GLS"]}
    segs = mf.siam_segments("MERCEDES-BENZ", "Model 167", codes)
    assert {n for _, n, _ in segs} == {"GLE", "GLS"}
    assert {h for _, _, h in segs} == {"platform_code"}
    segs = mf.siam_segments("MERCEDES-BENZ", "GLE (167); C - Class (205)", codes)
    assert {mf.reduce_family("MERCEDES-BENZ", n)[0] for _, n, _ in segs} == {"GLE", "C-CLASS"}


# ---------- build_vocabulary / siam_row_families / nhtsa_families ----------

def _india():
    return pd.DataFrame({"make": ["Honda", "Volvo", "BMW"],
                         "model": ["City", "XC90", "3 Series Gran Limousine"],
                         "source_url": ["https://h", "https://v", "https://b"]})


def _siam():
    return pd.DataFrame({"csv_row": [1, 2, 3, 4],
                         "make": ["HONDA", "VOLVO", "BMW", "AUDI"],
                         "model": ["CR-V; City", "XC90 Hybrid", "BMW X5, X6", "Audi"],
                         "source_url": "https://siam"})


def test_build_vocabulary_merges_sources_with_evidence():
    vocab, issues = mf.build_vocabulary(_india(), _siam(), {})
    by = {(r.make, r.family): r for r in vocab.itertuples(index=False)}
    assert by[("HONDA", "CITY")].source == "india_models;siam_text"
    assert "siam csv_row: 1" in by[("HONDA", "CITY")].evidence
    assert by[("HONDA", "CR-V")].source == "siam_text"
    assert ("VOLVO", "XC90") in by and ("VOLVO", "XC90 HYBRID") not in by
    assert ("BMW", "3 SERIES") in by and ("BMW", "X6") in by
    assert not any(r.make == "AUDI" for r in vocab.itertuples())  # bare 'Audi' adds nothing


def test_build_vocabulary_merges_spelling_by_key_keeping_india_name():
    india = pd.DataFrame({"make": ["Hyundai"], "model": ["Ioniq 5"], "source_url": ["https://h"]})
    siam = pd.DataFrame({"csv_row": [7], "make": ["HYUNDAI"], "model": ["IONIQ5"],
                         "source_url": "https://siam"})
    vocab, _ = mf.build_vocabulary(india, siam, {})
    assert vocab.family.tolist() == ["IONIQ 5"]
    assert vocab.source.iloc[0] == "india_models;siam_text"


def test_siam_row_families_logs_unusable_segments():
    vocab_df, _ = mf.build_vocabulary(_india(), _siam(), {})
    vocab = vocab_df.groupby("make").family.apply(set).to_dict()
    rows, issues = mf.siam_row_families(_siam(), vocab, {})
    assert set(rows[rows.csv_row == 1].family) == {"CR-V", "CITY"}
    assert any(i[2] == "4" for i in issues)  # 'Audi' alone


def test_siam_issue_names_the_expanded_code_name():
    codes = {"167": ["GLE", "AMD GLE"]}  # 'AMD GLE' is a SIAM typo in the code table
    siam = pd.DataFrame({"csv_row": [9], "make": ["MERCEDES-BENZ"], "model": ["Model 167"],
                         "source_url": "u"})
    rows, issues = mf.siam_row_families(siam, {"MERCEDES-BENZ": {"GLE"}}, codes)
    assert rows.family.tolist() == ["GLE"]
    assert issues[0][3] == "Model 167 -> AMD GLE"


def test_review_skips_rule_makes():
    vocab = pd.DataFrame({"make": ["MERCEDES-BENZ"] * 2, "family": ["GL", "GLA"],
                          "source": "siam_text", "evidence": "", "source_url": ""})
    assert mf.vocabulary_review(vocab, pd.DataFrame(columns=["make", "MODELTXT"])).empty


# ---------- vocabulary_review ----------

def test_review_flags_prefix_collisions_and_typos():
    vocab = pd.DataFrame({"make": ["TOYOTA", "TOYOTA", "VOLKSWAGEN"],
                          "family": ["LAND CRUISER", "LAND CRUISER 300", "PASAT"],
                          "source": ["siam_text", "india_models", "siam_text"],
                          "evidence": "", "source_url": ""})
    models = pd.DataFrame({"make": ["VOLKSWAGEN"], "MODELTXT": ["PASSAT"]})
    notes = mf.vocabulary_review(vocab, models)
    assert (notes.family == "LAND CRUISER 300").any()
    typo = notes[notes.family == "PASAT"].review_note.iloc[0]
    assert "PASSAT" in typo and "not merged" in typo


# ---------- build + main ----------
