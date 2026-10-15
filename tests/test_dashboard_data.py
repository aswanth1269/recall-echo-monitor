"""Tests for app/data_sources.py and an end-to-end run of app/dashboard.py.

The dashboard is run headless with Streamlit's AppTest against a small fake
project (RECALL_ECHO_ROOT), so these tests never read the real outputs.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import data_sources as ds  # noqa: E402

FILES = {
    "data/clean/kpi_echo_rate.csv":
        "level,grp,n,echoing,echo_rate,needs_review_not_echoing\n"
        "all,ALL,40,10,0.25,1\nmake,BIGCO,32,9,0.2812,1\nmake,TINYCO,8,1,,0\n"
        "segment,luxury,32,9,0.2812,1\nsegment,mass_market,8,1,,0\n",
    "data/clean/kpi_echo_lag.csv":
        "level,grp,n,median_days,p25_days,p75_days,india_first\n"
        "all,ALL,10,90.0,30.0,200.0,2\nmake,BIGCO,9,95.0,31.0,210.0,2\nmake,TINYCO,1,40.0,40.0,40.0,0\n"
        "segment,luxury,9,95.0,31.0,210.0,2\nsegment,mass_market,1,40.0,40.0,40.0,0\n",
    "data/clean/kpi_echo_by_component.csv":
        "component_group,n,echoing,echo_rate\nAIR BAGS,20,6,0.3\nSEATS,12,1,0.0833\n",
    "data/clean/kpi_ota.csv":
        "part,level,grp,ota,n,n_known_ota,ota_true,ota_share,echoing,echo_rate\n"
        "ota_share,all,ALL,,40,38,4,0.1053,,\n"
        "ota_share,make,BIGCO,,32,30,4,0.1333,,\n"
        "ota_share,make,TINYCO,,8,8,0,,,\n"
        "echo_by_ota,ota,not_ota,False,34,,,,9,0.2647\n"
        "echo_by_ota,ota,ota,True,4,,,,1,\n"
        "echo_by_ota,ota,unknown,,2,,,,0,\n",
    "data/clean/echo_explorer.csv":
        "CAMPNO,make,segment,report_date,component,us_defect,label,india_recall_date,"
        "india_model,india_defect,lag_days,confidence,us_evidence,india_evidence\n"
        "20V001000,BIGCO,luxury,2020-01-01,AIR BAGS,The driver’s air bag may rupture.,"
        "same,2020-03-01,City,Driver airbag inflator may rupture.,60,0.9,"
        "the driver's air  bag may rupture,airbag inflator may rupture\n"
        "20V002000,TINYCO,mass_market,2020-02-01,SEATS,Seat may detach.,no_candidate,,,,,,,\n",
    "reports/eval_summary.csv":
        "matcher,n,positives,tp,fp,fn,tn,precision,recall,f1,precision_n,recall_n,f1_n,threshold,note\n"
        "llm,30,5,5,0,0,25,,,1.0,5,5,10,,tuned\n"
        "baseline,30,5,4,0,1,25,,,,4,5,9,0.2177,flatters\n",
    "docs/SOURCES.md": "| Retrieved | 2026-10-13 |\n",
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    for rel, content in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    monkeypatch.setenv("RECALL_ECHO_ROOT", str(tmp_path))
    return tmp_path


# ---------- data_sources: reading and reshaping ----------

def test_missing_files_give_none(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_ECHO_ROOT", str(tmp_path))
    for fn in (ds.echo_rate, ds.echo_lag, ds.components, ds.ota_share, ds.ota_echo,
               ds.explorer, ds.eval_summary):
        assert fn() is None


def test_echo_rate_regroups_levels(project):
    df = ds.echo_rate()
    assert list(df.columns[:2]) == ["group_type", "group"]
    assert df.group_type.tolist() == ["overall", "make", "make", "segment", "segment"]
    assert pd.isna(df.set_index("group").loc["TINYCO", "echo_rate"])   # never filled


def test_echo_lag_regroups_levels(project):
    assert ds.echo_lag().query("group_type == 'overall'").median_days.iloc[0] == 90.0


def test_components_pass_through(project):
    assert ds.components().component_group.tolist() == ["AIR BAGS", "SEATS"]


def test_ota_share_uses_known_flags_as_denominator(project):
    o = ds.ota_share().set_index("group")
    assert (o.loc["ALL", "n"], o.loc["ALL", "ota_n"], o.loc["ALL", "n_all"]) == (38, 4, 40)
    assert o.loc["ALL", "group_type"] == "overall"


def test_ota_echo_names_groups(project):
    assert ds.ota_echo().ota.tolist() == ["Not OTA", "OTA", "Unknown"]


def test_explorer_and_eval_summary_read(project):
    assert len(ds.explorer()) == 2
    assert ds.eval_summary().set_index("matcher").loc["llm", "f1"] == 1.0


def test_wrong_format_raises_format_error_naming_columns(project):
    (project / "data/clean/kpi_echo_rate.csv").write_text(
        "group_type,group,n,echoing,echo_rate\noverall,all,40,10,0.25\n")   # old contract
    with pytest.raises(ds.FormatError, match=r"kpi_echo_rate.csv is missing columns \['level', 'grp'\]"):
        ds.echo_rate()


def test_stamp_changes_when_a_file_is_rewritten(project):
    before = ds.stamp()
    p = project / "data/clean/kpi_echo_lag.csv"
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert ds.stamp() != before


def test_root_defaults_to_repo(monkeypatch):
    monkeypatch.delenv("RECALL_ECHO_ROOT", raising=False)
    assert ds.root() == APP.parent


# ---------- data_sources: display helpers ----------

@pytest.mark.parametrize("value, expected", [(None, ""), (float("nan"), ""), ("x", "x"), (3, "3")])
def test_text_never_returns_nan(value, expected):
    assert ds.text(value) == expected


def test_find_evidence_matches_like_the_evidence_check():
    src = "The driver’s air bag  may\nrupture."
    i, j = ds.find_evidence(src, "THE DRIVER'S AIR BAG MAY RUPTURE")
    assert src[i:j] == "The driver’s air bag  may\nrupture"


def test_find_evidence_none_for_missing_or_absent_phrase():
    assert ds.find_evidence("financial loss", float("nan")) is None   # no 'nan' highlight
    assert ds.find_evidence("brake hose may leak", "airbag may rupture") is None
    assert ds.find_evidence(None, "x") is None


# ---------- the dashboard, end to end ----------

def _run(project):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP / "dashboard.py"), default_timeout=120).run()


def test_dashboard_renders_every_section_from_pipeline_files(project):
    at = _run(project)
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "yet. Run" not in text                       # no false "not run yet" states
    assert "nan" not in text.lower().replace("financial", "")
    assert "25.0%" in text                              # overall echo rate tile
    assert "Data retrieved 2026-10-13" in text
    assert len(at.get("plotly_chart")) >= 5             # makes, segments, lag, components, OTA


def test_small_n_rates_are_not_shown_anywhere(project):
    at = _run(project)
    specs = " ".join(c.proto.spec for c in at.get("plotly_chart"))
    assert "too few campaigns to show a rate" in specs
    assert "TINYCO</b><br>Echo rate" not in specs       # no rate in the hover for n < 10


def test_matcher_tile_states_who_labelled_and_tuning(project):
    text = " ".join(m.value for m in _run(project).markdown)
    assert "labelled by Claude and tuned on" in text
    assert "labelled by hand" not in text


def test_malformed_file_is_named_not_a_crash(project):
    (project / "data/clean/kpi_echo_rate.csv").write_text("group_type,group\noverall,all\n")
    at = _run(project)
    assert not at.exception
    assert any("kpi_echo_rate.csv is missing columns" in m.value for m in at.markdown)
