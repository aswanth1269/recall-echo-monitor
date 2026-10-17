"""Tests for app/pipeline_3d.py: counts are read, never invented, and the page is assembled."""

import json
import subprocess
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pipeline_3d  # noqa: E402


def test_missing_database_gives_all_none(tmp_path):
    counts = pipeline_3d.stage_counts(tmp_path / "nope.duckdb")
    assert set(counts) == set(pipeline_3d.STAGE_QUERIES)
    assert all(v is None for v in counts.values())


def test_counts_only_for_tables_that_exist(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE nhtsa_campaigns AS SELECT * FROM range(5)")
    con.execute("CREATE TABLE matches_reviewed (CAMPNO VARCHAR, final_label VARCHAR)")
    con.execute("INSERT INTO matches_reviewed VALUES ('A','same'),('A','same'),"
                "('B','different'),('C','same'),('D',NULL)")
    con.execute("CREATE TABLE india_relevant_campaigns (CAMPNO VARCHAR, make VARCHAR)")
    con.execute("INSERT INTO india_relevant_campaigns VALUES ('X','AUDI'),('X','VOLKSWAGEN'),('Y','BMW')")
    con.close()
    counts = pipeline_3d.stage_counts(db)
    assert counts["us_campaigns"] == 5
    assert counts["matched"] == 2          # distinct campaigns a person confirmed
    assert counts["relevant"] == 2         # a campaign under two makes counts once
    assert counts["pairs"] is None         # table missing -> None, not 0


def test_locked_database_gives_none_instead_of_crashing(tmp_path):
    db = tmp_path / "busy.duckdb"
    duckdb.connect(str(db)).execute("CREATE TABLE nhtsa_campaigns AS SELECT 1").close()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         f"import duckdb, time; c = duckdb.connect(r'{db}'); print('locked', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        counts = pipeline_3d.stage_counts(db)
        assert all(v is None for v in counts.values())
    finally:
        holder.kill()
        holder.wait()


def test_headline_numbers_read_pipeline_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_ECHO_ROOT", str(tmp_path))
    clean = tmp_path / "data/clean"
    clean.mkdir(parents=True)
    (clean / "kpi_echo_rate.csv").write_text(
        "level,grp,n,echoing,echo_rate,needs_review_not_echoing\n"
        "all,ALL,40,10,0.25,0\nmake,X,40,10,0.25,0\n")
    out = pipeline_3d.headline_numbers()
    assert out["echo_rate"] == 0.25
    assert out["median_lag_days"] is None


def test_f1_is_not_a_headline_number_or_in_the_page():
    # F1 is shown only on the dashboard's Method tab
    assert "f1" not in pipeline_3d.headline_numbers()
    page = (Path(pipeline_3d.__file__).parent / "assets/pipeline_3d.html").read_text(encoding="utf-8")
    assert "DATA.f1" not in page and "F1" not in page


def test_malformed_kpi_file_gives_none_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_ECHO_ROOT", str(tmp_path))
    clean = tmp_path / "data/clean"
    clean.mkdir(parents=True)
    (clean / "kpi_echo_rate.csv").write_text("group_type,group\noverall,all\n")
    assert pipeline_3d.headline_numbers()["echo_rate"] is None


def test_build_html_inlines_data_and_library(tmp_path):
    tpl = tmp_path / "t.html"
    tpl.write_text("<script>/*__THREE__*/</script><script>var DATA = /*__DATA__*/null;</script>")
    lib = tmp_path / "lib.js"
    lib.write_text("var THREE = {};")
    page = pipeline_3d.build_html({"pairs": 7, "note": "</script>"}, tpl, lib)
    assert "var THREE = {};" in page
    assert "/*__DATA__*/" not in page
    payload = page.split("var DATA = ")[1].split(";</script>")[0]
    assert json.loads(payload.replace("<\\/", "</"))["pairs"] == 7
    assert "</script>\"" not in page       # data cannot close the script tag


def test_real_page_has_no_untrue_claims():
    page = pipeline_3d.TEMPLATE.read_text(encoding="utf-8")
    assert "confirmed India echo" not in page and "then a human review" not in page
