"""Tests for src/fetch_ota.py. The HTTP call and sleep are always faked."""

from __future__ import annotations

import json
import logging
import urllib.error
from pathlib import Path

import duckdb
import pytest

from src import fetch_ota as fo


def _payload(*flags):
    """API-shaped response; one result per flag (None = field absent)."""
    rows = [{"NHTSACampaignNumber": "X", "ReportReceivedDate": "20/04/2022",
             **({} if f is None else {"overTheAirUpdate": f})} for f in flags]
    return {"Count": len(rows), "Message": "Results returned successfully", "results": rows}


# ---------- ota_from_response ----------

@pytest.mark.parametrize("payload, expected", [
    (_payload(True, True), (True, None)),
    (_payload(False, False, False), (False, None)),
    (_payload(None, True), (True, None)),                 # absent field ignored, not guessed
    (_payload(), (None, "API has no results for this campaign")),
    (_payload(None), (None, "no result carries a true/false overTheAirUpdate")),
    (_payload(True, False), (None, "results disagree on overTheAirUpdate")),
    (_payload("true"), (None, "no result carries a true/false overTheAirUpdate")),  # strings not trusted
])
def test_ota_from_response(payload, expected):
    assert fo.ota_from_response(payload) == expected


# ---------- fetch_campaign: caching ----------

def test_fetch_caches_and_rerun_makes_no_request(tmp_path: Path):
    calls = []

    def get(url):
        calls.append(url)
        return _payload(True)

    p1, cached1 = fo.fetch_campaign("24V200000", tmp_path, get)
    p2, cached2 = fo.fetch_campaign("24V200000", tmp_path, get)
    assert (cached1, cached2) == (False, True) and p1 == p2
    assert calls == ["https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber=24V200000"]
    assert json.loads((tmp_path / "24V200000.json").read_text())["Count"] == 1


def test_fetch_caches_empty_answers_too(tmp_path: Path):
    fo.fetch_campaign("99V999999", tmp_path, lambda u: _payload())
    assert (tmp_path / "99V999999.json").exists()


def test_fetch_rejects_and_does_not_cache_odd_responses(tmp_path: Path):
    with pytest.raises(ValueError, match="unexpected API response"):
        fo.fetch_campaign("20V000001", tmp_path, lambda u: {"error": "boom"})
    assert not (tmp_path / "20V000001.json").exists()


# ---------- fetch_all: pacing, failures ----------

def test_fetch_all_pauses_between_real_requests_only(tmp_path: Path):
    fo.fetch_campaign("20V000004", tmp_path, lambda u: _payload(False))   # pre-cached
    sleeps = []
    rows, requests = fo.fetch_all(["20V000001", "20V000004", "20V000002", "20V000003"], tmp_path,
                                  get=lambda u: _payload(True), sleep=sleeps.append)
    assert requests == 3 and sleeps == [0.3, 0.3]        # before B and C, not before A or CACHED
    assert rows.set_index("CAMPNO").ota.to_dict() == {"20V000001": True, "20V000004": False,
                                                      "20V000002": True, "20V000003": True}


def test_failed_request_is_null_and_retried_next_run(tmp_path: Path):
    def boom(url):
        raise urllib.error.URLError("network down")

    rows, _ = fo.fetch_all(["20V000001"], tmp_path, get=boom, sleep=lambda s: None)
    assert rows.ota.isna().all() and rows.null_reason.iloc[0].startswith("request failed")
    assert not (tmp_path / "20V000001.json").exists()            # nothing cached, so a rerun retries


# ---------- campaign_numbers / build / main ----------

def _db(tmp_path: Path) -> Path:
    db = tmp_path / "recalls.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE india_relevant_campaigns AS SELECT * FROM (VALUES
        ('16V045000', 'AUDI'), ('16V045000', 'VOLKSWAGEN'), ('24V200000', 'KIA'),
        ('99V999999', 'HONDA')) AS t(CAMPNO, make)""")
    con.close()
    return db


def test_campaign_numbers_are_distinct(tmp_path: Path):
    con = duckdb.connect(str(_db(tmp_path)), read_only=True)
    assert fo.campaign_numbers(con) == ["16V045000", "24V200000", "99V999999"]


def test_build_writes_table_and_logs_nulls(tmp_path: Path, caplog):
    answers = {"16V045000": _payload(True), "24V200000": _payload(False, False),
               "99V999999": _payload()}
    get = lambda url: answers[url.rsplit("=", 1)[1]]      # noqa: E731
    db = _db(tmp_path)
    with caplog.at_level(logging.INFO, logger="src.fetch_ota"):
        s = fo.build(db_path=db, cache_dir=tmp_path / "api", get=get, sleep=lambda x: None)
    assert (s["true"], s["false"], s["nulls"], s["requests"]) == (1, 1, 1, 3)
    assert "ota is NULL for 1 of 3 campaigns" in caplog.text
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT * FROM campaign_ota ORDER BY CAMPNO").fetchall() == [
        ("16V045000", True), ("24V200000", False), ("99V999999", None)]
    assert [c[0] for c in con.execute("DESCRIBE campaign_ota").fetchall()] == ["CAMPNO", "ota"]
    con.close()

    # second run: everything cached, no requests
    s2 = fo.build(db_path=db, cache_dir=tmp_path / "api",
                  get=lambda u: pytest.fail("must not request"), sleep=lambda x: None)
    assert s2["requests"] == 0


def test_main_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(fo, "build", lambda: {"campaigns": 3, "true": 1, "false": 1,
                                              "nulls": 1, "requests": 0, "null_reasons": {}})
    fo.main()
    assert "null=1" in capsys.readouterr().out
