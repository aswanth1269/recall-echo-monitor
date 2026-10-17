"""Run the KPI queries in sql/ and export each to data/clean/kpi_<name>.csv.

KPIs (one SQL file each, a one-line comment at the top of each says what it is):
  echo_rate          share of India-relevant campaigns that echo, by make / segment
  echo_lag           days from US report to earliest matched India recall
  echo_by_component  echo rate by component group (n >= 10 only)
  ota                OTA share by make / report year; echo rate OTA vs non-OTA

Row-level export for the dashboard (data/clean/echo_explorer.csv):
  echo_explorer      one row per campaign x make with its best pair or no_candidate

A campaign (campaign x make) "echoes" when a person confirmed at least one of
its pairs as `same` (matches_reviewed.final_label; run src.review first). Every output row carries n; rates are NULL where n < 10.

Before running, the inputs are checked: matches must come from the full run
(scope 'all') and cover every candidate pair, and campaign_ota must exist -
otherwise the KPIs would silently describe a partial picture.

Run:  .venv/Scripts/python.exe -m src.kpis
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.load_nhtsa import DB_PATH

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
OUT_DIR = ROOT / "data/clean"
MAKES_CSV = ROOT / "data/raw/india_makes.csv"
KPIS = ("echo_rate", "echo_lag", "echo_by_component", "ota")
# Row-level exports for the dashboard (no n column; not KPIs): file = <name>.csv
EXPORTS = ("echo_explorer",)


def load_make_segments(makes_csv: Path = MAKES_CSV) -> pd.DataFrame:
    """(make, segment) with make in the NHTSA spelling used by every table."""
    m = pd.read_csv(makes_csv, dtype=str, keep_default_na=False)
    # the file's own `make` column is the display name; use the NHTSA spelling instead
    out = m[["nhtsa_maketxt_expected", "segment"]].rename(
        columns={"nhtsa_maketxt_expected": "make"})
    bad = sorted(set(out.segment) - {"luxury", "mass_market"})
    if bad:
        raise ValueError(f"unexpected segment values in {makes_csv.name}: {bad}")
    return out


def check_inputs(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse to compute KPIs from a partial matcher run or without OTA flags."""
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    for t in ("matches", "matches_reviewed", "india_relevant_campaigns", "siam_recalls",
              "campaign_ota"):
        if t not in tables:
            raise ValueError(f"table {t} is missing - build it first")
    scopes = {r[0] for r in con.execute("SELECT DISTINCT scope FROM matches").fetchall()}
    if scopes != {"all"}:
        raise ValueError(f"matches has scope {sorted(scopes)}; run python -m src.matcher --all")
    if "candidate_pairs" in tables:
        missing = con.execute("""SELECT count(*) FROM candidate_pairs p
            WHERE NOT EXISTS (SELECT 1 FROM matches m
                              WHERE m.CAMPNO = p.CAMPNO AND m.csv_row = p.csv_row)""").fetchone()[0]
        if missing:
            raise ValueError(f"{missing} candidate pairs have no verdict in matches")
    n_m, n_r = con.execute("SELECT (SELECT count(*) FROM matches), "
                           "(SELECT count(*) FROM matches_reviewed)").fetchone()
    if n_m != n_r:
        raise ValueError(f"matches_reviewed has {n_r} rows but matches has {n_m}; "
                         "run python -m src.review")


def run_kpi(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Execute sql/<name>.sql and return its result."""
    return con.execute((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")).df()


def build(db_path: Path = DB_PATH, out_dir: Path = OUT_DIR,
          makes_csv: Path = MAKES_CSV) -> dict[str, pd.DataFrame]:
    """Run every KPI and export, write them to data/clean, return the tables.

    KPIs go to kpi_<name>.csv; row-level exports (EXPORTS) to <name>.csv.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        check_inputs(con)
        con.register("make_segments", load_make_segments(makes_csv))
        results = {name: run_kpi(con, name) for name in KPIS + EXPORTS}
    finally:
        con.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        stem = name if name in EXPORTS else f"kpi_{name}"
        df.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8")
    return results


def main() -> None:
    results = build()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    for name, df in results.items():
        if name in EXPORTS:
            print(f"\n=== {name}.csv ({len(df)} rows; row-level, see the file) ===")
            print(df.label.value_counts().to_string() if "label" in df else "")
            continue
        print(f"\n=== kpi_{name}.csv ({len(df)} rows) ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
