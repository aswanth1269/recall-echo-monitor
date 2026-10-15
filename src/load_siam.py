"""Load the SIAM recall CSV into DuckDB as table siam_recalls.

Reads data/raw/siam_recalls.csv (never writes to data/raw) and writes table
siam_recalls into data/clean/recalls.duckdb, next to the NHTSA tables:

  * text cells have runs of whitespace collapsed and ends stripped;
  * `make` is the NHTSA spelling (e.g. MERCEDES-BENZ), `make_label` the
    name used in india_makes.csv;
  * the three date columns are parsed into DATE columns only when the cell
    holds exactly one unambiguous date. Everything else - two dates in one
    cell, month-and-year only, 03/04/2021 (day or month first?), impossible
    dates - stays NULL, keeps its raw text, and is written to the table
    siam_date_issues and the log. Nothing is guessed.

Run:  python -m src.load_siam
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.load_nhtsa import DB_PATH, MAKETXT_ALIASES

ROOT = Path(__file__).resolve().parents[1]
SIAM_CSV = ROOT / "data/raw/siam_recalls.csv"

DATE_COLUMNS = {  # raw column -> parsed column
    "voluntary_recall_date": "recall_date",
    "manufactured_from": "manufactured_from_date",
    "manufactured_to": "manufactured_to_date",
}
EARLIEST_PLAUSIBLE_YEAR = 1990  # before any car still being recalled was built

log = logging.getLogger(__name__)

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS |= {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
_MONTHS["sept"] = 9

_NAMED = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?[\s\-]*([A-Za-z]+)[\s\-]*(\d{2}|\d{4})$")
_NUMERIC = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2}|\d{4})$")
_SEVERAL = re.compile(r"[;&,()]|\band\b|\bto\b", re.I)


def clean_text(value: object) -> str | None:
    """Collapse whitespace runs and strip ends; blank becomes None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = " ".join(str(value).split())
    return text or None


def _year(token: str) -> int:
    """'18' -> 2018, '98' -> 1998, '2018' -> 2018 (fixed pivot, not a guess per row)."""
    if len(token) == 4:
        return int(token)
    yy = int(token)
    return 2000 + yy if yy <= 69 else 1900 + yy


def parse_date(value: object) -> tuple[date | None, str | None]:
    """Parse one SIAM date cell.

    Returns (date, None) on success, or (None, reason) when the cell does not
    hold exactly one unambiguous calendar date.
    """
    text = clean_text(value)
    if text is None:
        return None, "blank"
    if _SEVERAL.search(text):
        return None, "several dates or labelled text in one cell"

    if m := _NAMED.match(text):
        day, word, yr = int(m.group(1)), m.group(2).lower(), m.group(3)
        month = _MONTHS.get(word)
        if month is None:
            return None, f"unknown month name {m.group(2)!r}"
    elif m := _NUMERIC.match(text):
        a, b, yr = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12 and b <= 12:
            day, month = a, b
        elif b > 12 and a <= 12:
            day, month = b, a
        elif a == b:
            day = month = a
        elif a <= 12 and b <= 12:
            return None, "ambiguous day/month order"
        else:
            return None, "not a real date"
    else:
        return None, "unrecognised format (e.g. month-year or year only)"

    try:
        return date(_year(yr), month, day), None
    except ValueError:
        return None, "not a real date"


def standardise_make(make: str, aliases: dict[str, list[str]] = MAKETXT_ALIASES) -> str:
    """india_makes.csv name -> primary NHTSA MAKETXT spelling ('Mercedes-Benz' -> 'MERCEDES-BENZ')."""
    key = clean_text(make)
    if key not in aliases:
        raise ValueError(f"no NHTSA spelling recorded for make {make!r}")
    return aliases[key][0]


def clean_siam(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned rows, date issues). Raw date text is kept beside each parsed date."""
    out = df.map(clean_text)
    # The CSV has no row-number column, so rows are identified by their
    # 1-based position in it; that is what siam_date_issues refers to.
    out.insert(0, "csv_row", range(1, len(out) + 1))
    out.insert(1, "make_label", out.pop("make"))
    out.insert(1, "make", out["make_label"].map(standardise_make))
    out["target_recall_vehicles"] = pd.to_numeric(out["target_recall_vehicles"],
                                                  errors="coerce").astype("Int64")

    issues = []
    for raw_col, date_col in DATE_COLUMNS.items():
        parsed = []
        for row_no, raw in zip(out["csv_row"], out[raw_col]):
            value, reason = parse_date(raw)
            parsed.append(value)
            if reason and reason != "blank":
                issues.append((row_no, raw_col, raw, reason))
            elif reason == "blank":
                issues.append((row_no, raw_col, None, "blank in source"))
        out[date_col] = pd.Series(parsed, index=out.index, dtype="object")

    # Plausibility: a manufacturing date before 1990 or after the recall
    # date is not believable; null it and log it rather than keep it.
    for date_col, raw_col in (("manufactured_from_date", "manufactured_from"),
                              ("manufactured_to_date", "manufactured_to")):
        for idx in out.index:
            d, recall = out.at[idx, date_col], out.at[idx, "recall_date"]
            if d is None:
                continue
            if d.year < EARLIEST_PLAUSIBLE_YEAR or (recall is not None and d > recall):
                issues.append((out.at[idx, "csv_row"], raw_col, out.at[idx, raw_col],
                               "parsed but outside 1990..recall date"))
                out.at[idx, date_col] = None

    issue_df = pd.DataFrame(issues, columns=["csv_row", "column_name", "raw_value", "reason"])
    return out, issue_df


def build(db_path: Path = DB_PATH, csv_path: Path = SIAM_CSV) -> tuple[int, int]:
    """Write siam_recalls and siam_date_issues; return (rows, issues)."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8")
    clean, issues = clean_siam(raw)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.register("clean_df", clean)
        con.execute("""
            CREATE OR REPLACE TABLE siam_recalls AS
            SELECT * REPLACE (
                CAST(csv_row AS INTEGER)            AS csv_row,
                CAST(recall_date AS DATE)               AS recall_date,
                CAST(manufactured_from_date AS DATE)    AS manufactured_from_date,
                CAST(manufactured_to_date AS DATE)      AS manufactured_to_date,
                CAST(retrieved_on AS DATE)              AS retrieved_on)
            FROM clean_df ORDER BY csv_row""")
        con.unregister("clean_df")
        con.register("issues_df", issues)
        con.execute("""
            CREATE OR REPLACE TABLE siam_date_issues AS
            SELECT CAST(csv_row AS INTEGER) AS csv_row,
                   CAST(column_name AS VARCHAR) AS column_name,
                   CAST(raw_value AS VARCHAR)   AS raw_value,
                   CAST(reason AS VARCHAR)      AS reason
            FROM issues_df ORDER BY csv_row, column_name""")
        con.unregister("issues_df")
    finally:
        con.close()

    for row in issues.itertuples(index=False):
        log.info("SIAM row %s %s=%r: %s", *row)
    return len(clean), len(issues)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    rows, n_issues = build()
    print(f"wrote siam_recalls ({rows} rows) and siam_date_issues ({n_issues} rows) "
          f"to {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
