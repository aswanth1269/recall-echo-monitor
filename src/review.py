"""Human review of LLM decisions: build the queue, record decisions, apply them.

Decision files (data/reference, hand-editable, committed):
  pair_reviews.csv            pair_id, decision (same|different), note, fingerprint, reviewed_at
  model_family_overrides.csv  side, make, key, family, note, fingerprint, reviewed_at
                              (read by src/model_family.py)
  vocabulary_reviews.csv      make, family, decision (distinct|alias), note, fingerprint, reviewed_at
  model_aliases.csv           make, from_name, family, source_url, retrieved_on, note
                              (read by src/model_family.py; an alias needs a source)

Every decision stores a fingerprint of the text the reviewer saw. If that text
changes later the decision is stale: kept, not applied, and the item returns
to the queue with the old decision shown. A decision whose item no longer
exists is orphaned: kept, not applied, logged.

Table matches_reviewed = matches + human decision; final_label is the decision
when it is current, else NULL. KPIs count an echo only when final_label =
'same'. The LLM's own label is never changed.

The dashboard and `--import` both go through record(), so the rules are the same.

Run:  .venv/Scripts/python.exe -m src.review            apply decisions, write the queue
      .venv/Scripts/python.exe -m src.review --import data/clean/review_queue.csv
      .venv/Scripts/python.exe -m src.review --rebuild  rerun the pipeline after family decisions
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src import model_family
from src.csv_safe import excel_safe_value
from src.fingerprint import fingerprint
from src.matcher import load_pairs

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data/reference"
VOCAB_FILE = "model_families.csv"

FILES = {"pair": "pair_reviews.csv",
         "family": "model_family_overrides.csv",
         "vocabulary": "vocabulary_reviews.csv",
         "alias": "model_aliases.csv"}
COLUMNS = {"pair": ["pair_id", "decision", "note", "fingerprint", "reviewed_at"],
           "family": model_family.OVERRIDE_COLUMNS + model_family.OVERRIDE_REVIEW_COLUMNS,
           "vocabulary": ["make", "family", "decision", "note", "fingerprint", "reviewed_at"],
           "alias": model_family.ALIAS_COLUMNS}
KEYS = {"pair": ["pair_id"], "family": ["side", "make", "key"],
        "vocabulary": ["make", "family"], "alias": ["make", "from_name"]}
PAIR_DECISIONS = ("same", "different")
VOCAB_DECISIONS = ("distinct", "alias")

log = logging.getLogger(__name__)


def _s(value: object) -> str:
    """Text or ''; never 'nan'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def read_decisions(kind: str, ref_dir: Path = REF_DIR) -> pd.DataFrame:
    """One decision file as text; an absent file is an empty frame. Optional columns filled."""
    path = ref_dir / FILES[kind]
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS[kind])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in COLUMNS[kind]:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS[kind]]


def _write_atomic(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _mask(df: pd.DataFrame, keys: dict[str, str]) -> pd.Series:
    hit = pd.Series(True, index=df.index)
    for k, v in keys.items():
        hit &= df[k] == v
    return hit


def _upsert(kind: str, row: dict, ref_dir: Path) -> dict:
    """Replace the row with the same key (re-reading the file first); return the change."""
    df = read_decisions(kind, ref_dir)
    keys = {k: row[k] for k in KEYS[kind]}
    hit = _mask(df, keys)
    previous = df[hit].iloc[0].to_dict() if hit.any() else None
    new = pd.DataFrame([row], columns=COLUMNS[kind])
    out = pd.concat([df[~hit], new], ignore_index=True) if len(df) else new
    _write_atomic(ref_dir / FILES[kind], out)
    return {"kind": kind, "keys": keys, "previous": previous}


def _family_value(make: str, decision: str, ref_dir: Path) -> str:
    if decision.lower() == model_family.NONE:
        return model_family.NONE
    allowed = {f.upper(): f for f in model_family.load_vocab(ref_dir / VOCAB_FILE).get(make, [])}
    if decision.upper() not in allowed:
        raise ValueError(f"family {decision!r} is not in the {make} vocabulary "
                         f"(choose one of {sorted(allowed.values())} or 'none')")
    return allowed[decision.upper()]


def record(item: dict, decision: str, note: str = "", *, from_name: str | None = None,
           source_url: str | None = None, retrieved_on: str | None = None,
           ref_dir: Path = REF_DIR, today: str | None = None) -> list[dict]:
    """Validate one decision on a queue item and write it; return the changes (for undo).

    Raises ValueError, writing nothing, when the decision is not allowed for the
    item, a family is outside the vocabulary, or an alias lacks a source.
    """
    kind, make, key = _s(item.get("kind")), _s(item.get("make")), _s(item.get("key"))
    fp, decision = _s(item.get("fingerprint")), _s(decision)
    today = today or date.today().isoformat()
    note = excel_safe_value(_s(note))
    if kind not in ("pair", "family", "vocabulary"):
        raise ValueError(f"unknown item kind {kind!r}")
    if not fp:
        raise ValueError(f"item {key!r} has no fingerprint; rebuild the queue with src.review")

    if kind == "pair":
        if decision not in PAIR_DECISIONS:
            raise ValueError(f"a pair decision must be same or different, not {decision!r}")
        return [_upsert("pair", {"pair_id": key, "decision": decision, "note": note,
                                 "fingerprint": fp, "reviewed_at": today}, ref_dir)]

    if kind == "family":
        family = _family_value(make, decision, ref_dir)
        return [_upsert("family", {"side": _s(item.get("side")), "make": make, "key": key,
                                   "family": family, "note": note, "fingerprint": fp,
                                   "reviewed_at": today}, ref_dir)]

    if decision not in VOCAB_DECISIONS:
        raise ValueError(f"a vocabulary decision must be distinct or alias, not {decision!r}")
    alias_row = None
    if decision == "alias":
        name, url, when = _s(from_name), _s(source_url), _s(retrieved_on)
        if not name:
            raise ValueError("an alias needs from_name (the other spelling)")
        if not url.startswith(("http://", "https://")):
            raise ValueError("an alias needs a source_url (http or https) showing both "
                             "names are one model")
        try:
            date.fromisoformat(when)
        except ValueError:
            raise ValueError(f"an alias needs retrieved_on as YYYY-MM-DD, not {when!r}") from None
        alias_row = {"make": make, "from_name": name, "family": key, "source_url": url,
                     "retrieved_on": when, "note": note}
    changes = [_upsert("alias", alias_row, ref_dir)] if alias_row else []
    changes.append(_upsert("vocabulary", {"make": make, "family": key, "decision": decision,
                                          "note": note, "fingerprint": fp,
                                          "reviewed_at": today}, ref_dir))
    return changes


def undo(changes: list[dict], ref_dir: Path = REF_DIR) -> None:
    """Put back what record() replaced (or remove what it added), newest change first."""
    for change in reversed(changes):
        kind = change["kind"]
        df = read_decisions(kind, ref_dir)
        df = df[~_mask(df, change["keys"])]
        if change["previous"] is not None:
            prev = pd.DataFrame([change["previous"]], columns=COLUMNS[kind])
            df = pd.concat([df, prev], ignore_index=True) if len(df) else prev
        _write_atomic(ref_dir / FILES[kind], df)


def _current_keys(ref_dir: Path) -> set[tuple[str, str, str, str, str]]:
    """(kind, side, make, key, fingerprint) of every decision in the files."""
    out = set()
    for r in read_decisions("pair", ref_dir).itertuples(index=False):
        out.add(("pair", "", "", r.pair_id, r.fingerprint))
    for r in read_decisions("family", ref_dir).itertuples(index=False):
        out.add(("family", r.side, r.make, r.key, r.fingerprint))
    for r in read_decisions("vocabulary", ref_dir).itertuples(index=False):
        out.add(("vocabulary", "", r.make, r.family, r.fingerprint))
    return out


def outstanding(queue_df: pd.DataFrame, ref_dir: Path = REF_DIR) -> pd.DataFrame:
    """Open or stale items with no current decision on file yet."""
    done = _current_keys(ref_dir)

    def decided(r) -> bool:
        side = _s(r.side) if r.kind == "family" else ""
        make = "" if r.kind == "pair" else _s(r.make)
        return (r.kind, side, make, _s(r.key), _s(r.fingerprint)) in done

    todo = queue_df[queue_df.status.isin(["open", "stale"])]
    keep = pd.Series([not decided(r) for r in todo.itertuples(index=False)],
                     index=todo.index, dtype=bool)
    return todo[keep].reset_index(drop=True)


# ---------- the queue ----------

QUEUE_COLUMNS = ["item_id", "order", "kind", "side", "make", "key", "status",
                 "llm_answer", "confidence", "us_text", "india_text", "us_evidence",
                 "india_evidence", "reason", "fingerprint", "previous_decision"]
ORDER = {"family_llm": 1, "family_zero": 2, "vocabulary": 3, "needs_review": 4,
         "same": 5, "family_none": 6, "orphaned": 7}
DEFAULT_ORDERS = (1, 2, 3, 4, 5)


def _item(order: int, kind: str, side: str, make: str, key: str, **kw) -> dict:
    row = {c: None for c in QUEUE_COLUMNS}
    row.update(order=order, kind=kind, side=side, make=make, key=str(key), status="open",
               item_id=f"{kind}|{side}|{make}|{key}")
    row.update(kw)
    return row


def _resolve(items: dict, decisions: list[tuple], lookup) -> list[dict]:
    """Attach decisions to items.

    items: key tuple -> open item. decisions: (key tuple, decision value, fingerprint).
    lookup(key tuple) -> item for an existing thing that is not a default candidate, or None.
    Current decision (blank or equal fingerprint) -> reviewed; different -> stale;
    nothing to attach to -> orphaned.
    """
    items = dict(items)
    for k, value, fp in decisions:
        item = items.get(k) or lookup(k)
        if item is None:
            kind, side, make, key = k
            items[k] = _item(ORDER["orphaned"], kind, side, make, key, status="orphaned",
                             previous_decision=value, reason="decision has no matching item")
            continue
        status = "reviewed" if fp in ("", item["fingerprint"]) else "stale"
        items[k] = {**item, "status": status, "previous_decision": value}
    return list(items.values())


def _family_items(con: duckdb.DuckDBPyConnection, ref_dir: Path) -> list[dict]:
    nhtsa = con.execute("SELECT make, MODELTXT, family, method FROM nhtsa_model_family").df()
    siam = con.execute("SELECT csv_row, make, model FROM siam_recalls").df()
    with_family = {int(r[0]) for r in con.execute(
        "SELECT DISTINCT csv_row FROM siam_row_family").fetchall()}

    def nhtsa_item(r, order):
        llm = r.method == "llm"
        return _item(order, "family", "nhtsa", r.make, r.MODELTXT, us_text=r.MODELTXT,
                     llm_answer=(_s(r.family) or model_family.NONE) if llm else None,
                     reason="an LLM chose this family" if llm else None,
                     fingerprint=model_family.override_fingerprint(
                         "nhtsa", r.make, r.MODELTXT, r.MODELTXT))

    def siam_item(r):
        return _item(ORDER["family_zero"], "family", "siam", r.make, str(r.csv_row),
                     india_text=r.model,
                     reason=("SIAM row produced zero families" if r.csv_row not in with_family
                             else "SIAM row has a family decision"),
                     fingerprint=model_family.override_fingerprint(
                         "siam", r.make, str(r.csv_row), r.model))

    items = {}
    for r in nhtsa[nhtsa.method == "llm"].itertuples(index=False):
        order = ORDER["family_llm"] if _s(r.family) else ORDER["family_none"]
        items[("family", "nhtsa", r.make, r.MODELTXT)] = nhtsa_item(r, order)
    for r in siam[~siam.csv_row.isin(with_family)].itertuples(index=False):
        items[("family", "siam", r.make, str(r.csv_row))] = siam_item(r)

    by_nhtsa = {(r.make, r.MODELTXT): r for r in nhtsa.itertuples(index=False)}
    by_siam = {(r.make, str(r.csv_row)): r for r in siam.itertuples(index=False)}

    def lookup(k):
        _, side, make, key = k
        if side == "nhtsa" and (make, key) in by_nhtsa:
            return nhtsa_item(by_nhtsa[(make, key)], ORDER["family_llm"])
        if side == "siam" and (make, key) in by_siam:
            return siam_item(by_siam[(make, key)])
        return None

    dec = [(("family", r.side, r.make, r.key), r.family, r.fingerprint)
           for r in read_decisions("family", ref_dir).itertuples(index=False)]
    return _resolve(items, dec, lookup)


def _vocabulary_items(con: duckdb.DuckDBPyConnection, ref_dir: Path) -> list[dict]:
    vocab = pd.read_csv(ref_dir / VOCAB_FILE, dtype=str, keep_default_na=False)
    models = con.execute("SELECT DISTINCT make, MODELTXT FROM nhtsa_focus "
                         "WHERE MODELTXT IS NOT NULL").df()
    flags = model_family.vocabulary_review(vocab, models)
    if len(flags):
        flags = flags.groupby(["make", "family"], as_index=False).review_note.agg(" | ".join)
    items = {("vocabulary", "", r.make, r.family): _item(
        ORDER["vocabulary"], "vocabulary", "", r.make, r.family, india_text=r.family,
        reason=r.review_note,
        fingerprint=fingerprint("vocabulary", r.make, r.family, r.review_note))
        for r in flags.itertuples(index=False)}
    dec = [(("vocabulary", "", r.make, r.family), r.decision, r.fingerprint)
           for r in read_decisions("vocabulary", ref_dir).itertuples(index=False)]
    return _resolve(items, dec, lambda k: None)


def _pair_items(con: duckdb.DuckDBPyConnection, ref_dir: Path) -> list[dict]:
    decisions = read_decisions("pair", ref_dir)
    try:
        m = con.execute("""SELECT pair_id, CAMPNO, make, csv_row, label, llm_label, confidence,
                                  us_evidence, india_evidence, review_reason, fingerprint
                           FROM matches""").df()
    except duckdb.BinderException:
        raise ValueError("matches has no fingerprint column; "
                         "run python -m src.matcher --all") from None
    wanted = m[m.label.isin(["same", "needs_review"]) | m.pair_id.isin(decisions.pair_id)]
    texts = load_pairs(con, wanted[["CAMPNO", "csv_row"]]).set_index("pair_id")

    def pair_item(r):
        order = ORDER["same"] if r.label == "same" else ORDER["needs_review"]
        t = texts.loc[r.pair_id] if r.pair_id in texts.index else None
        return _item(order, "pair", "", r.make, r.pair_id,
                     llm_answer="needs_review" if r.label == "needs_review" else r.llm_label,
                     confidence=r.confidence, us_evidence=r.us_evidence,
                     india_evidence=r.india_evidence, reason=r.review_reason,
                     us_text=None if t is None else t.us_text,
                     india_text=None if t is None else t.india_text,
                     fingerprint=r.fingerprint)

    by_id = {r.pair_id: r for r in wanted.itertuples(index=False)}
    items = {("pair", "", r.make, r.pair_id): pair_item(r)
             for r in wanted[wanted.label.isin(["same", "needs_review"])].itertuples(index=False)}

    def lookup(k):
        r = by_id.get(k[3])
        return pair_item(r) if r is not None else None

    dec = [(("pair", "", by_id[r.pair_id].make if r.pair_id in by_id else "", r.pair_id),
            r.decision, r.fingerprint) for r in decisions.itertuples(index=False)]
    return _resolve(items, dec, lookup)


def queue(con: duckdb.DuckDBPyConnection, ref_dir: Path = REF_DIR) -> pd.DataFrame:
    """Every review item with its status, in review order (families first)."""
    rows = _family_items(con, ref_dir) + _vocabulary_items(con, ref_dir) + _pair_items(con, ref_dir)
    q = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    q["side"] = q.side.fillna("")
    return q.sort_values(["order", "make", "key"], kind="stable").reset_index(drop=True)
