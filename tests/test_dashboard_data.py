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
        "matcher,n,positives,tp,fp,fn,tn,precision,recall,f1,precision_n,recall_n,f1_n,threshold,"
        "note,labelled_by\n"
        "llm,30,5,5,0,0,25,,,1.0,5,5,10,,tuned,"
        "a person for 2 of 30 pairs (human_labels.csv); Claude for the other 28\n"
        "baseline,30,5,4,0,1,25,,,,4,5,9,0.2177,flatters,"
        "a person for 2 of 30 pairs (human_labels.csv); Claude for the other 28\n",
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


def test_dashboard_renders_every_section_from_pipeline_files(review_project):
    at = _run(review_project)          # every pipeline output present, review queue included
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


def test_overview_has_pairs_reviewed_tile_not_f1(review_project):
    at = _run(review_project)
    overview = " ".join(m.value for m in at.tabs[0].markdown)
    assert "Pairs reviewed" in overview and "1 of 2" in overview
    assert "n&lt;10, share not shown" in overview or "n<10, share not shown" in overview
    assert "F1" not in overview


def test_f1_only_on_method_tab_with_who_labelled(review_project):
    at = _run(review_project)
    assert "F1" not in " ".join(m.value for m in at.tabs[1].markdown)
    method = at.tabs[-1]
    assert any("f1" in d.value.columns.tolist() for d in method.dataframe)
    assert "a person for 2 of 30 pairs" in " ".join(c.value for c in method.caption)


def test_pair_review_summary_counts_llm_same_pairs(review_project):
    assert ds.pair_review_summary() == {"n_llm_same": 2, "n_reviewed": 1, "n_confirmed": 1,
                                        "n_cannot_judge": 0}


def test_pairs_reviewed_tile_shows_share_from_ten(review_project):
    rows = "".join(
        f"pair||BIGCO|21V{i:03d}000-1,5,pair,,BIGCO,21V{i:03d}000-1,reviewed,same,0.9,a,b,,,,"
        f"{'d' * 16},{'same' if i < 9 else 'different'},,,,,\n" for i in range(10))
    q = review_project / "data/clean/review_queue.csv"
    q.write_text(q.read_text(encoding="utf-8") + rows, encoding="utf-8")
    overview = " ".join(m.value for m in _run(review_project).tabs[0].markdown)
    assert "11 of 12" in overview and "10 confirmed" in overview and "n=11" in overview


def test_malformed_file_is_named_not_a_crash(project):
    (project / "data/clean/kpi_echo_rate.csv").write_text("group_type,group\noverall,all\n")
    at = _run(project)
    assert not at.exception
    assert any("kpi_echo_rate.csv is missing columns" in m.value for m in at.markdown)


# ---------- review ----------

QUEUE = ("item_id,order,kind,side,make,key,status,llm_answer,confidence,us_text,india_text,"
         "us_evidence,india_evidence,reason,fingerprint,previous_decision,decision,note,"
         "from_name,source_url,retrieved_on\n"
         "family|nhtsa|BIGCO|FIT,1,family,nhtsa,BIGCO,FIT,open,CITY,,FIT,,,,an LLM chose this family,"
         "aaaaaaaaaaaaaaaa,,,,,,\n"
         "pair||BIGCO|20V001000-1,5,pair,,BIGCO,20V001000-1,open,same,0.9,"
         "Defect: The driver air bag may rupture.,Defect: Driver airbag inflator may rupture.,"
         "driver air bag may rupture,airbag inflator may rupture,,bbbbbbbbbbbbbbbb,,,,,,\n"
         "pair||BIGCO|20V003000-3,5,pair,,BIGCO,20V003000-3,reviewed,same,0.8,a,b,,,,"
         "cccccccccccccccc,same,,,,,\n")


@pytest.fixture
def review_project(project):
    (project / "data/clean/review_queue.csv").write_text(QUEUE, encoding="utf-8")
    ref = project / "data/reference"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "model_families.csv").write_text("make,family,source\nBIGCO,CITY,india_models\n",
                                            encoding="utf-8")
    return project


def test_review_queue_reads_as_text(review_project):
    q = ds.review_queue()
    assert len(q) == 3 and q.key.tolist()[0] == "FIT" and q.fingerprint.iloc[0] == "a" * 16


def test_decisions_newer_than_queue(review_project):
    assert not ds.decisions_newer_than_queue()
    time.sleep(0.05)
    (review_project / "data/reference/pair_reviews.csv").write_text(
        "pair_id,decision,note,fingerprint,reviewed_at\n", encoding="utf-8")
    assert ds.decisions_newer_than_queue()


def test_review_tab_shows_progress_and_first_item(review_project):
    at = _run(review_project)
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "1 of 2" in text                      # echo candidates: one reviewed of two
    assert "FIT" in text                         # families come first


def test_review_tab_records_a_decision_through_review_record(review_project):
    at = _run(review_project)
    at.selectbox(key="review_family_choice").select("CITY").run()
    at.button(key="review_confirm").click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = pd.read_csv(review_project / "data/reference/model_family_overrides.csv", dtype=str)
    assert df[["side", "make", "key", "family"]].values.tolist() == [["nhtsa", "BIGCO", "FIT", "CITY"]]
    text = " ".join(m.value for m in at.markdown)
    assert "src.review --rebuild" in text        # family decisions need a rebuild
    assert "20V001000-1" in text                 # moved on to the next item


def test_review_tab_without_queue_says_what_to_run(project):
    text = " ".join(m.value for m in _run(project).markdown)
    assert "python -m src.review" in text


def test_explorer_shows_awaiting_review(project):
    path = project / "data/clean/echo_explorer.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",same,", ",awaiting_review,"),
                    encoding="utf-8")
    at = _run(project)
    assert not at.exception, [e.value for e in at.exception]


def test_review_tab_sample_filter_shows_only_sampled_same_pairs(review_project):
    at = _run(review_project)
    at.checkbox(key="review_only_sample").check().run()
    assert not at.exception, [e.value for e in at.exception]
    review = at.tabs[3]
    text = " ".join(m.value for m in review.markdown)
    assert "20V001000-1" in text and "Echo candidate" in text     # family item FIT filtered out
    assert "FIT" not in text.split("item(s) to review")[1]
    assert any("Sample: 1 of 2 reviewed" in c.value for c in review.caption)
    saved = pd.read_csv(review_project / "data/reference/review_sample.csv", dtype=str)
    assert sorted(saved.pair_id) == ["20V001000-1", "20V003000-3"] and set(saved.seed) == {"42"}


def test_review_tab_sample_is_read_back_not_redrawn(review_project):
    (review_project / "data/reference/review_sample.csv").write_text(
        "pair_id,seed,drawn_on\n20V003000-3,42,2026-10-24\nGONE-1,42,2026-10-24\n",
        encoding="utf-8")
    at = _run(review_project)
    at.checkbox(key="review_only_sample").check().run()
    assert not at.exception, [e.value for e in at.exception]
    caps = " ".join(c.value for c in at.tabs[3].caption)
    assert "Sample: 1 of 2 reviewed" in caps and "1 sampled pair(s) are no longer" in caps
    assert "20V001000-1" not in " ".join(m.value for m in at.tabs[3].markdown)


# ---------- India rows with several families; going back to reviewed items ----------

SIAM_ROW = ("family|siam|BIGCO|252,2,family,siam,BIGCO,252,open,,,,"
            "Bigco | variant: City & Amaze,,,SIAM row produced zero families,"
            "ffffffffffffffff,,,,,,\n")


@pytest.fixture
def siam_project(review_project):
    q = review_project / "data/clean/review_queue.csv"
    q.write_text(q.read_text(encoding="utf-8") + SIAM_ROW, encoding="utf-8")
    (review_project / "data/reference/model_families.csv").write_text(
        "make,family,source\nBIGCO,AMAZE,india_models\nBIGCO,CITY,india_models\n",
        encoding="utf-8")
    # FIT already decided, so the India row is the first item
    (review_project / "data/reference/model_family_overrides.csv").write_text(
        "side,make,key,family,note,fingerprint,reviewed_at\n"
        "nhtsa,BIGCO,FIT,CITY,,aaaaaaaaaaaaaaaa,2026-10-24\n", encoding="utf-8")
    return review_project


def test_india_row_shows_variant_and_saves_several_families(siam_project):
    at = _run(siam_project)
    text = " ".join(m.value for m in at.tabs[3].markdown)
    assert "variant: City &amp; Amaze" in text or "variant: City & Amaze" in text
    at.multiselect(key="review_family_multi").select("CITY").select("AMAZE").run()
    at.button(key="review_confirm").click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = pd.read_csv(siam_project / "data/reference/model_family_overrides.csv", dtype=str)
    assert sorted(df[df.side == "siam"].family) == ["AMAZE", "CITY"]


def test_show_reviewed_and_search_bring_back_a_decided_item(siam_project):
    at = _run(siam_project)
    at.checkbox(key="review_show_reviewed").check().run()
    at.text_input(key="review_search").input("FIT").run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.tabs[3].markdown)
    assert "FIT" in text and "already decided" in text


def test_cannot_judge_button_saves_and_is_left_out_of_the_share(review_project):
    at = _run(review_project)
    at.checkbox(key="review_only_sample").check().run()        # go straight to the pair
    at.button(key="review_cannot_judge").click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = pd.read_csv(review_project / "data/reference/pair_reviews.csv", dtype=str)
    assert df.decision.tolist() == ["cannot_judge"]
    q = review_project / "data/clean/review_queue.csv"
    q.write_text(q.read_text(encoding="utf-8").replace(
        "20V003000-3,reviewed,same,0.8,a,b,,,,cccccccccccccccc,same",
        "20V003000-3,reviewed,same,0.8,a,b,,,,cccccccccccccccc,cannot_judge"), encoding="utf-8")
    assert ds.pair_review_summary() == {"n_llm_same": 2, "n_reviewed": 0, "n_confirmed": 0,
                                        "n_cannot_judge": 1}


def test_explorer_names_cannot_judge(project):
    path = project / "data/clean/echo_explorer.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",same,", ",cannot_judge,"),
                    encoding="utf-8")
    at = _run(project)
    assert not at.exception, [e.value for e in at.exception]
