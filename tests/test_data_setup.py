"""Day 0 guard rails: the environment imports and the raw data is the shape
docs/SOURCES.md claims. These are cheap and catch a silently truncated
download or a SIAM page change before any analysis is built on top.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
NHTSA_ZIP = RAW / "FLAT_RCL_POST_2010.zip"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RAW / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_required_packages_import():
    import anthropic, duckdb, dotenv, pandas, plotly, rapidfuzz, streamlit  # noqa: F401


def test_env_example_tracked_and_env_ignored():
    assert (ROOT / ".env.example").exists(), ".env.example is the tracked template"
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored
    assert "!.env.example" in ignored


def test_makes_file_has_ten_makes_split_across_segments():
    makes = read_csv("india_makes.csv")
    assert len(makes) == 10
    segments = {m["segment"] for m in makes}
    assert segments == {"mass_market", "luxury"}
    # a mix, not 9-and-1
    luxury = sum(m["segment"] == "luxury" for m in makes)
    assert 3 <= luxury <= 5, f"expected a balanced mix, got {luxury} luxury"
    assert len({m["make"] for m in makes}) == 10, "make names must be unique"


def test_every_make_has_india_models_and_no_duplicates():
    makes = {m["make"] for m in read_csv("india_makes.csv")}
    models = read_csv("india_models.csv")
    assert models, "india_models.csv is empty"

    covered = {m["make"] for m in models}
    assert covered == makes, f"make mismatch: {covered ^ makes}"

    pairs = [(m["make"], m["model"]) for m in models]
    assert len(pairs) == len(set(pairs)), "duplicate make+model rows"
    assert all(m["model"].strip() for m in models), "blank model name"
    assert all(m["source_url"].startswith("https://") for m in models)


def test_siam_recalls_cover_only_the_picked_makes():
    makes = {m["make"] for m in read_csv("india_makes.csv")}
    rows = read_csv("siam_recalls.csv")
    assert len(rows) == 355, "SIAM row count drifted - re-read docs/SOURCES.md"
    assert {r["make"] for r in rows} == makes
    assert all(r["make_resolution"] in ("entity", "model_token") for r in rows)
    assert all(r["retrieved_on"] == "2026-10-13" for r in rows)


def test_siam_all_file_is_a_superset():
    picked = read_csv("siam_recalls.csv")
    every = read_csv("siam_recalls_all.csv")
    assert len(every) == 546
    assert len(every) > len(picked)
    assert any(r["make_resolution"] == "out_of_scope" for r in every)


@pytest.mark.skipif(not NHTSA_ZIP.exists(),
                    reason="NHTSA zip not downloaded (gitignored)")
def test_nhtsa_flat_file_parses_to_29_fields():
    """RCL.txt documents 29 fields; a ragged row means a bad download."""
    with zipfile.ZipFile(NHTSA_ZIP) as z:
        (member,) = z.namelist()
        with z.open(member) as fh:
            stream = io.TextIOWrapper(fh, encoding="latin-1", newline="")
            for i, raw in enumerate(stream):
                if i >= 5000:
                    break
                assert len(raw.rstrip("\r\n").split("\t")) == 29


@pytest.mark.skipif(not NHTSA_ZIP.exists(),
                    reason="NHTSA zip not downloaded (gitignored)")
def test_every_make_resolves_against_nhtsa_maketxt():
    """The US-side join key must exist for all 10 makes, or the premise fails."""
    expected = {m["nhtsa_maketxt_expected"] for m in read_csv("india_makes.csv")}
    seen: set[str] = set()
    with zipfile.ZipFile(NHTSA_ZIP) as z:
        (member,) = z.namelist()
        with z.open(member) as fh:
            for raw in io.TextIOWrapper(fh, encoding="latin-1", newline=""):
                fields = raw.rstrip("\r\n").split("\t")
                if len(fields) == 29:
                    seen.add(fields[2].strip().upper())
    assert expected <= seen, f"no NHTSA records for: {sorted(expected - seen)}"
