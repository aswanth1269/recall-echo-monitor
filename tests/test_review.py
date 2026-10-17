"""Tests for src/review.py. The LLM is never called; the database is a small fake."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import review as rv

PAIR_ITEM = {"kind": "pair", "side": "", "make": "HONDA", "key": "20V001000-1",
             "fingerprint": "aaaaaaaaaaaaaaaa"}
FAMILY_ITEM = {"kind": "family", "side": "nhtsa", "make": "HONDA", "key": "FIT",
               "fingerprint": "bbbbbbbbbbbbbbbb"}
VOCAB_ITEM = {"kind": "vocabulary", "side": "", "make": "HONDA", "key": "CITY HYBRID",
              "fingerprint": "cccccccccccccccc"}


@pytest.fixture
def ref(tmp_path: Path) -> Path:
    d = tmp_path / "reference"
    d.mkdir()
    pd.DataFrame({"make": ["HONDA"] * 3, "family": ["AMAZE", "CITY", "CITY HYBRID"],
                  "source": ["india_models"] * 3}).to_csv(d / "model_families.csv", index=False)
    return d


def _files(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# ---------- record / undo / outstanding ----------

def test_record_pair_writes_one_row_with_fingerprint_and_date(ref):
    changes = rv.record(PAIR_ITEM, "same", "checked both texts", ref_dir=ref, today="2026-10-16")
    df = rv.read_decisions("pair", ref)
    assert df.to_dict("records") == [{"pair_id": "20V001000-1", "decision": "same",
                                      "note": "checked both texts",
                                      "fingerprint": "aaaaaaaaaaaaaaaa",
                                      "reviewed_at": "2026-10-16"}]
    assert changes == [{"kind": "pair", "keys": {"pair_id": "20V001000-1"}, "previous": None}]


def test_second_decision_replaces_the_first(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref, today="2026-10-16")
    changes = rv.record(PAIR_ITEM, "different", "on second look", ref_dir=ref, today="2026-10-17")
    df = rv.read_decisions("pair", ref)
    assert df.decision.tolist() == ["different"]
    assert changes[0]["previous"]["decision"] == "same"


@pytest.mark.parametrize("item, decision, message", [
    (PAIR_ITEM, "maybe", "same or different"),
    (FAMILY_ITEM, "CIVIC", "not in the HONDA vocabulary"),
    (VOCAB_ITEM, "merge", "distinct or alias"),
    ({**PAIR_ITEM, "kind": "nonsense"}, "same", "unknown item kind"),
    ({**PAIR_ITEM, "fingerprint": ""}, "same", "no fingerprint"),
])
def test_invalid_decisions_are_refused_and_nothing_is_written(ref, item, decision, message):
    before = _files(ref)
    with pytest.raises(ValueError, match=message):
        rv.record(item, decision, ref_dir=ref)
    assert _files(ref) == before


def test_family_decision_uses_vocabulary_spelling_or_none(ref):
    rv.record(FAMILY_ITEM, "city", ref_dir=ref, today="2026-10-16")
    rv.record({**FAMILY_ITEM, "key": "PILOT"}, "None", ref_dir=ref, today="2026-10-16")
    df = rv.read_decisions("family", ref)
    assert df[["side", "make", "key", "family"]].values.tolist() == [
        ["nhtsa", "HONDA", "FIT", "CITY"], ["nhtsa", "HONDA", "PILOT", "none"]]
    assert (df.fingerprint == "bbbbbbbbbbbbbbbb").all()


def test_alias_needs_a_source_and_writes_both_files(ref):
    with pytest.raises(ValueError, match="source_url"):
        rv.record(VOCAB_ITEM, "alias", from_name="CITY E:HEV", ref_dir=ref)
    with pytest.raises(ValueError, match="retrieved_on"):
        rv.record(VOCAB_ITEM, "alias", from_name="CITY E:HEV",
                  source_url="https://example.com/city", retrieved_on="16/10/2026", ref_dir=ref)
    changes = rv.record(VOCAB_ITEM, "alias", "same car", from_name="CITY E:HEV",
                        source_url="https://example.com/city", retrieved_on="2026-10-16",
                        ref_dir=ref, today="2026-10-16")
    assert [c["kind"] for c in changes] == ["alias", "vocabulary"]
    alias = pd.read_csv(ref / "model_aliases.csv", dtype=str)
    assert alias[["make", "from_name", "family"]].values.tolist() == [
        ["HONDA", "CITY E:HEV", "CITY HYBRID"]]
    assert rv.read_decisions("vocabulary", ref).decision.tolist() == ["alias"]


def test_distinct_writes_only_the_vocabulary_review_file(ref):
    rv.record(VOCAB_ITEM, "distinct", ref_dir=ref, today="2026-10-16")
    assert not (ref / "model_aliases.csv").exists()
    assert rv.read_decisions("vocabulary", ref).decision.tolist() == ["distinct"]


def test_note_is_made_formula_safe(ref):
    rv.record(PAIR_ITEM, "same", '=HYPERLINK("http://x")', ref_dir=ref)
    assert rv.read_decisions("pair", ref).note.iloc[0].startswith("'=")


def test_record_only_writes_the_decision_files(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    rv.record(FAMILY_ITEM, "CITY", ref_dir=ref)
    rv.record(VOCAB_ITEM, "alias", from_name="X", source_url="https://e.com",
              retrieved_on="2026-10-16", ref_dir=ref)
    assert _files(ref) == {"model_families.csv", *rv.FILES.values()}   # no .tmp left behind


def test_record_keeps_rows_added_by_hand_meanwhile(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    path = ref / rv.FILES["pair"]
    path.write_text(path.read_text(encoding="utf-8")
                    + "20V009000-9,different,by hand,,2026-10-16\n", encoding="utf-8")
    rv.record({**PAIR_ITEM, "key": "20V002000-2"}, "same", ref_dir=ref)
    assert set(rv.read_decisions("pair", ref).pair_id) == {
        "20V001000-1", "20V009000-9", "20V002000-2"}


def test_undo_restores_the_previous_state(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref, today="2026-10-16")
    changes = rv.record(PAIR_ITEM, "different", ref_dir=ref, today="2026-10-17")
    rv.undo(changes, ref)
    assert rv.read_decisions("pair", ref).decision.tolist() == ["same"]
    first = rv.record({**PAIR_ITEM, "key": "20V002000-2"}, "same", ref_dir=ref)
    rv.undo(first, ref)
    assert rv.read_decisions("pair", ref).pair_id.tolist() == ["20V001000-1"]


def test_outstanding_drops_items_decided_since_the_queue_was_written(ref):
    q = pd.DataFrame([{**PAIR_ITEM, "status": "open"},
                      {**PAIR_ITEM, "key": "20V002000-2", "status": "open"},
                      {**FAMILY_ITEM, "status": "stale"},
                      {**VOCAB_ITEM, "status": "reviewed"}])
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    out = rv.outstanding(q, ref)
    assert out[["kind", "key"]].values.tolist() == [["pair", "20V002000-2"], ["family", "FIT"]]


# ---------- queue ----------

import duckdb  # noqa: E402

from src import matcher as mt  # noqa: E402


def _db(tmp_path: Path) -> Path:
    """HONDA only. Pairs: 20V001000-1 same, 20V001000-2 different, 20V002000-2 needs_review.
    Families: FIT llm->CITY, PILOT llm->none, CITY rule. SIAM row 3 has zero families."""
    db = tmp_path / "r.duckdb"
    if db.exists():                       # built once per test; later calls reuse it
        return db
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE india_relevant_campaigns AS SELECT * FROM (VALUES
        ('20V001000', 'HONDA', ['AIR BAGS'], 'Driver air bag inflator may rupture.', 'Injury.'),
        ('20V002000', 'HONDA', ['SEATS'], 'Seat may detach from the floor.', 'Injury.'))
        AS t(CAMPNO, make, components, DESC_DEFECT, CONEQUENCE_DEFECT)""")
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'HONDA', 'City', 'Driver airbag inflator may rupture.'),
        (2, 'HONDA', 'Amaze', 'Seat rail may detach.'),
        (3, 'HONDA', 'Mystery Van', 'Brake hose may leak.'))
        AS t(csv_row, make, model, defect_description)""")
    con.execute("""CREATE TABLE siam_row_family AS SELECT * FROM (VALUES
        (1, 'HONDA', 'CITY', 'rule'), (2, 'HONDA', 'AMAZE', 'rule'))
        AS t(csv_row, make, family, method)""")
    con.execute("""CREATE TABLE nhtsa_model_family AS SELECT * FROM (VALUES
        ('HONDA', 'CITY', 'CITY', 'rule'), ('HONDA', 'FIT', 'CITY', 'llm'),
        ('HONDA', 'PILOT', NULL, 'llm')) AS t(make, MODELTXT, family, method)""")
    con.execute("CREATE TABLE nhtsa_focus AS SELECT make, MODELTXT FROM nhtsa_model_family")
    keys = pd.DataFrame({"CAMPNO": ["20V001000", "20V001000", "20V002000"], "csv_row": [1, 2, 2]})
    p = mt.load_pairs(con, keys)
    labels = {"20V001000-1": "same", "20V001000-2": "different", "20V002000-2": "needs_review"}
    m = p[["pair_id", "CAMPNO", "make", "csv_row", "fingerprint"]].assign(
        label=p.pair_id.map(labels), llm_label=p.pair_id.map(labels).replace("needs_review", "same"),
        confidence=0.9, us_evidence="us quote", india_evidence="india quote",
        review_reason=p.pair_id.map({"20V002000-2": "india_evidence not found"}),
        llm_model="fake", prompt_version=mt.PROMPT_VERSION, scope="all")
    con.register("m", m)
    con.execute("CREATE TABLE matches AS SELECT * FROM m")
    con.close()
    return db


def _queue(tmp_path, ref):
    con = duckdb.connect(str(_db(tmp_path)), read_only=True)
    try:
        return rv.queue(con, ref)
    finally:
        con.close()


def test_queue_lists_every_kind_in_order(tmp_path, ref):
    q = _queue(tmp_path, ref)
    assert list(q.columns) == rv.QUEUE_COLUMNS
    assert q[["order", "kind", "side", "key", "status"]].values.tolist() == [
        [1, "family", "nhtsa", "FIT", "open"],
        [2, "family", "siam", "3", "open"],
        [3, "vocabulary", "", "CITY HYBRID", "open"],
        [4, "pair", "", "20V002000-2", "open"],
        [5, "pair", "", "20V001000-1", "open"],
        [6, "family", "nhtsa", "PILOT", "open"]]
    assert q.item_id.is_unique
    pair = q[q.key == "20V001000-1"].iloc[0]
    assert pair.us_text.startswith("Component: AIR BAGS") and pair.llm_answer == "same"
    assert q[q.key == "FIT"].iloc[0].llm_answer == "CITY"


def test_current_decisions_show_as_reviewed(tmp_path, ref):
    q = _queue(tmp_path, ref)
    for item in q.to_dict("records")[:5]:
        decision = {"family": "none", "vocabulary": "distinct", "pair": "same"}[item["kind"]]
        rv.record(item, decision, ref_dir=ref)
    q2 = _queue(tmp_path, ref)
    assert q2.status.tolist()[:5] == ["reviewed"] * 5
    assert q2.previous_decision.tolist()[:5] == ["none", "none", "distinct", "same", "same"]


def test_changed_text_makes_a_decision_stale(tmp_path, ref):
    q = _queue(tmp_path, ref)
    item = q[q.key == "20V001000-1"].iloc[0].to_dict()
    rv.record({**item, "fingerprint": "0" * 16}, "different", ref_dir=ref)
    row = _queue(tmp_path, ref).set_index("key").loc["20V001000-1"]
    assert (row.status, row.previous_decision) == ("stale", "different")


def test_decision_for_a_vanished_item_is_orphaned(tmp_path, ref):
    rv.record({**PAIR_ITEM, "key": "99V999000-9"}, "same", ref_dir=ref)
    rv.record({**FAMILY_ITEM, "key": "GONE MODEL"}, "CITY", ref_dir=ref)
    q = _queue(tmp_path, ref)
    orphans = q[q.status == "orphaned"]
    assert set(orphans.key) == {"99V999000-9", "GONE MODEL"} and set(orphans.order) == {7}


def test_queue_without_matcher_fingerprints_says_what_to_run(tmp_path, ref):
    db = _db(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE matches DROP COLUMN fingerprint")
    con.close()
    con = duckdb.connect(str(db), read_only=True)
    with pytest.raises(ValueError, match="src.matcher --all"):
        rv.queue(con, ref)
    con.close()


# ---------- apply / precision ----------

def _fp(db, pair_id):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT fingerprint FROM matches WHERE pair_id = ?",
                           [pair_id]).fetchone()[0]
    finally:
        con.close()


def test_apply_sets_final_label_only_for_current_decisions(tmp_path, ref):
    db = _db(tmp_path)
    rv.record({**PAIR_ITEM, "key": "20V001000-1", "fingerprint": _fp(db, "20V001000-1")},
              "same", ref_dir=ref)
    rv.record({**PAIR_ITEM, "key": "20V002000-2", "fingerprint": "0" * 16}, "same", ref_dir=ref)
    rv.record({**PAIR_ITEM, "key": "99V999000-9"}, "different", ref_dir=ref)
    counts = rv.apply(db, ref)
    assert counts == {"reviewed": 1, "stale": 1, "unreviewed": 1, "orphaned": 1}
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("""SELECT pair_id, label, human_decision, review_status, final_label
                          FROM matches_reviewed ORDER BY pair_id""").fetchall()
    assert rows == [("20V001000-1", "same", "same", "reviewed", "same"),
                    ("20V001000-2", "different", None, "unreviewed", None),
                    ("20V002000-2", "needs_review", "same", "stale", None)]
    con.close()


def test_apply_refuses_bad_decision_values(tmp_path, ref):
    db = _db(tmp_path)
    (ref / rv.FILES["pair"]).write_text(
        "pair_id,decision,note,fingerprint,reviewed_at\n20V001000-1,yes,,x,2026-10-16\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="same/different"):
        rv.apply(db, ref)


def _reviewed_db(tmp_path, n_same, n_confirmed):
    con = duckdb.connect(str(tmp_path / "p.duckdb"))
    con.execute(f"""CREATE TABLE matches_reviewed AS
        SELECT 'same' AS label, 'reviewed' AS review_status,
               CASE WHEN i < {n_confirmed} THEN 'same' ELSE 'different' END AS final_label
        FROM range({n_same}) AS t(i)""")
    con.execute("INSERT INTO matches_reviewed VALUES ('same', 'unreviewed', NULL), "
                "('different', 'reviewed', 'different')")
    return con


def test_precision_counts_reviewed_llm_same_pairs_only(tmp_path):
    con = _reviewed_db(tmp_path, 12, 9)
    row = rv.precision(con).iloc[0]
    assert (row.n_reviewed, row.n_confirmed, row.precision) == (12, 9, 0.75)


def test_precision_is_suppressed_below_ten(tmp_path):
    con = _reviewed_db(tmp_path, 9, 9)
    row = rv.precision(con).iloc[0]
    assert row.n_reviewed == 9 and pd.isna(row.precision)


# ---------- CLI ----------

def test_write_queue_is_formula_safe_with_blank_inputs(tmp_path, ref):
    db = _db(tmp_path)
    out = tmp_path / "queue.csv"
    rv.write_queue(db, ref, out)
    sheet = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(sheet.columns) == rv.QUEUE_COLUMNS + rv.SHEET_INPUT_COLUMNS
    assert (sheet.decision == "").all() and len(sheet) == 6


def test_import_records_filled_rows_and_reports_bad_ones(tmp_path, ref):
    db = _db(tmp_path)
    out = tmp_path / "queue.csv"
    rv.write_queue(db, ref, out)
    sheet = pd.read_csv(out, dtype=str, keep_default_na=False)
    sheet.loc[sheet.key == "20V001000-1", ["decision", "note"]] = ["same", "ok"]
    sheet.loc[sheet.key == "FIT", "decision"] = "CIVIC"           # not in the vocabulary
    extra = sheet.iloc[[0]].assign(item_id="pair||HONDA|00X000000-1", decision="same")
    pd.concat([sheet, extra]).to_csv(out, index=False)
    recorded, errors = rv.import_sheet(out, db, ref)
    assert recorded == 1 and len(errors) == 2
    assert any("CIVIC" in e for e in errors)
    assert any("not in the current queue" in e for e in errors)
    assert rv.read_decisions("pair", ref).pair_id.tolist() == ["20V001000-1"]


def test_rebuild_runs_steps_in_order():
    ran = []
    rv.rebuild(steps=[(name, lambda n=name: ran.append(n)) for name in ("a", "b", "c")])
    assert ran == ["a", "b", "c"]


def test_main_applies_and_prints_counts(monkeypatch, capsys):
    monkeypatch.setattr(rv, "apply", lambda: {"reviewed": 1, "stale": 0, "unreviewed": 2,
                                              "orphaned": 0})
    monkeypatch.setattr(rv, "write_queue", lambda: pd.DataFrame(
        {"order": [1, 5], "kind": ["family", "pair"], "status": ["open", "reviewed"]}))
    rv.main([])
    out = capsys.readouterr().out
    assert "matches_reviewed: reviewed=1" in out and "family" in out


def test_main_import_then_apply(monkeypatch, capsys, tmp_path):
    seen = []
    monkeypatch.setattr(rv, "import_sheet", lambda p: seen.append(p) or (2, ["x: bad"]))
    monkeypatch.setattr(rv, "apply", lambda: {"reviewed": 2, "stale": 0, "unreviewed": 0,
                                              "orphaned": 0})
    monkeypatch.setattr(rv, "write_queue", lambda: pd.DataFrame(
        {"order": [5], "kind": ["pair"], "status": ["reviewed"]}))
    rv.main(["--import", str(tmp_path / "q.csv")])
    out = capsys.readouterr().out
    assert seen == [tmp_path / "q.csv"] and "recorded 2" in out and "x: bad" in out


def test_apply_with_no_decisions_keeps_text_columns(tmp_path, ref):
    # an empty decisions frame must not make DuckDB type final_label as a number
    db = _db(tmp_path)
    assert rv.apply(db, ref) == {"reviewed": 0, "stale": 0, "unreviewed": 3, "orphaned": 0}
    con = duckdb.connect(str(db), read_only=True)
    types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns "
                             "WHERE table_name = 'matches_reviewed'").fetchall())
    assert {types[c] for c in ("human_decision", "review_status", "final_label")} == {"VARCHAR"}
    assert con.execute("SELECT count(*) FROM matches_reviewed WHERE final_label = 'same'"
                       ).fetchone()[0] == 0
    con.close()
