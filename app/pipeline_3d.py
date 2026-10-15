"""3D animated explainer of the pipeline, embedded in the Streamlit dashboard.

The scene is a self-contained HTML page (app/assets/pipeline_3d.html) that
uses a vendored copy of three.js (app/assets/three.min.js, MIT licence), so it
works offline and needs no CDN. This module fills it with live counts read
from data/clean/recalls.duckdb and, through data_sources, the KPI files.

A count is None when its table or file does not exist yet, or when the
database is locked by a running pipeline step. The page shows None as a dash;
nothing is estimated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import data_sources as ds

ASSETS = Path(__file__).resolve().parent / "assets"
TEMPLATE = ASSETS / "pipeline_3d.html"
THREE_JS = ASSETS / "three.min.js"

log = logging.getLogger(__name__)

# Each stage count is one SQL query. If the table is missing, the count is None.
STAGE_QUERIES: dict[str, str] = {
    "us_campaigns": "SELECT count(*) FROM nhtsa_campaigns",
    "siam_recalls": "SELECT count(*) FROM siam_recalls",
    "families": ("SELECT count(DISTINCT family) FROM nhtsa_model_family "
                 "WHERE family IS NOT NULL AND lower(family) <> 'none'"),
    # distinct campaigns: 20 campaigns cover both Audi and VW and appear twice
    "relevant": "SELECT count(DISTINCT CAMPNO) FROM india_relevant_campaigns",
    "pairs": "SELECT count(*) FROM candidate_pairs",
    "matched": "SELECT count(DISTINCT CAMPNO) FROM matches WHERE label = 'same'",
}


def db_path() -> Path:
    return ds.root() / "data/clean/recalls.duckdb"


def stage_counts(db: Path | None = None) -> dict[str, int | None]:
    """Row counts for each pipeline stage; None where a table is missing or the DB is busy."""
    db = db or db_path()
    counts: dict[str, int | None] = {k: None for k in STAGE_QUERIES}
    if not db.exists():
        return counts
    import duckdb

    try:
        con = duckdb.connect(str(db), read_only=True)
    except duckdb.Error as exc:      # e.g. a pipeline step holds the write lock
        log.warning("pipeline counts unavailable: %s", str(exc).splitlines()[0])
        return counts
    try:
        for key, sql in STAGE_QUERIES.items():
            try:
                counts[key] = int(con.execute(sql).fetchone()[0])
            except duckdb.Error:
                counts[key] = None
    finally:
        con.close()
    return counts


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _overall(df: pd.DataFrame | None, column: str) -> float | None:
    if df is None:
        return None
    rows = df[df["group_type"] == "overall"]
    return None if rows.empty else _num(rows.iloc[0][column])


def _safe(read):
    """A data_sources table, or None if the file is missing or malformed.

    The dashboard's own panels report a malformed file; the 3D scene just
    shows a dash rather than crash the page.
    """
    try:
        return read()
    except ds.FormatError as exc:
        log.warning("3D explainer: %s", exc)
        return None


def headline_numbers() -> dict[str, float | None]:
    """Overall echo rate, median lag and LLM F1 (None if missing, malformed or suppressed)."""
    out: dict[str, float | None] = {
        "echo_rate": _overall(_safe(ds.echo_rate), "echo_rate"),
        "median_lag_days": _overall(_safe(ds.echo_lag), "median_days"),
        "f1": None,
    }
    ev = _safe(ds.eval_summary)
    if ev is not None:
        llm = ev[ev["matcher"].astype(str).str.lower() == "llm"]
        if not llm.empty:
            out["f1"] = _num(llm.iloc[0]["f1"])
    return out


def build_html(data: dict[str, Any], template: Path = TEMPLATE, three_js: Path = THREE_JS) -> str:
    """Return the page with the data JSON and three.js inlined."""
    page = template.read_text(encoding="utf-8")
    lib = three_js.read_text(encoding="utf-8")
    payload = json.dumps(data).replace("</", "<\\/")   # safe inside <script>
    page = page.replace("/*__DATA__*/null", payload, 1)
    page = page.replace("<script>/*__THREE__*/</script>", "<script>" + lib + "</script>", 1)
    return page


def render(height: int = 600) -> None:
    """Draw the 3D explainer inside the current Streamlit container.

    st.iframe runs the page's JavaScript with same-origin access, so only
    trusted content may go in: the fixed template, vendored three.js and
    numeric counts (no recall text, no LLM output).
    """
    import streamlit as st

    data: dict[str, Any] = {**stage_counts(), **headline_numbers()}
    st.iframe(build_html(data), height=height)
