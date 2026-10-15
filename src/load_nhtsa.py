"""Load the NHTSA recall flat file into DuckDB and build the focus tables.

Reads data/raw/FLAT_RCL_POST_2010.zip in place (data/raw is never written),
names its 29 columns from data/raw/RCL.txt, and writes
data/clean/recalls.duckdb with:

  nhtsa_raw        every record, all text, empty fields stored as NULL
  focus_makes      the NHTSA MAKETXT spellings that count as each picked make
  nhtsa_focus      picked makes, report date (RCDATE) 2015-01-01 onwards
  nhtsa_campaigns  one row per campaign number with its models and years
  load_issues      counts of missing / unknown / excluded values

The table SQL lives in sql/. Missing values stay NULL and are counted in
load_issues and the log; nothing is filled in.

Run:  python -m src.load_nhtsa
"""

from __future__ import annotations

import csv
import logging
import re
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data/raw/FLAT_RCL_POST_2010.zip"
RCL_PATH = ROOT / "data/raw/RCL.txt"
MAKES_CSV = ROOT / "data/raw/india_makes.csv"
DB_PATH = ROOT / "data/clean/recalls.duckdb"
SQL_DIR = ROOT / "sql"

log = logging.getLogger(__name__)

# Every MAKETXT spelling (after upper/trim) that means one of the picked makes,
# checked against the file retrieved 2026-10-13. Look-alikes deliberately left
# out: HONDA GENUINE ACCESSORY (accessory line), HYUNDAI TRANSLEAD (trailer
# maker), MERCEDES ROYALE (RV converter), JL AUDIO (audio maker), VOLVO BUS
# (Volvo Group buses, not Volvo Cars), and LEXUS / MINI (separate brands).
# MERCEDES-MAYBACH is kept: SIAM files Maybach under Mercedes Benz India, so
# keeping it makes the two sides comparable. MAKETXT stays in nhtsa_focus, so
# it can be filtered back out.
MAKETXT_ALIASES: dict[str, list[str]] = {
    "Toyota": ["TOYOTA"],
    "Honda": ["HONDA"],
    "Hyundai": ["HYUNDAI"],
    "Kia": ["KIA"],
    "Nissan": ["NISSAN"],
    "Volkswagen": ["VOLKSWAGEN"],
    "Mercedes-Benz": ["MERCEDES-BENZ", "MERCEDES BENZ", "MERCEDES-MAYBACH"],
    "BMW": ["BMW"],
    "Audi": ["AUDI"],
    "Volvo": ["VOLVO"],
}

_FIELD_LINE = re.compile(r"^\s*(\d+)\s+([A-Z][A-Z0-9_]*)\s")


def read_field_names(rcl_path: Path = RCL_PATH) -> list[str]:
    """Column names in file order, parsed from the FIELDS table in RCL.txt."""
    fields = [(int(m.group(1)), m.group(2))
              for line in rcl_path.read_text(encoding="utf-8").splitlines()
              if (m := _FIELD_LINE.match(line))]
    numbers = [n for n, _ in fields]
    if numbers != list(range(1, len(fields) + 1)):
        raise ValueError(f"RCL.txt field numbers are not 1..N in order: {numbers}")
    return [name for _, name in fields]


def read_nhtsa(zip_path: Path = ZIP_PATH, names: list[str] | None = None) -> pd.DataFrame:
    """Read the tab-delimited, header-less, UTF-8 recall file from inside the zip.

    Every column is text. Empty fields become None so they load as NULL.
    """
    names = names or read_field_names()
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if len(members) != 1:
            raise ValueError(f"expected one file in {zip_path.name}, found {members}")
        with zf.open(members[0]) as fh:
            df = pd.read_csv(
                fh, sep="\t", header=None, names=names, dtype=str,
                quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False,
                encoding="utf-8", on_bad_lines="error",
            )
    return df.replace({"": None})


def focus_make_table(makes_csv: Path = MAKES_CSV,
                     aliases: dict[str, list[str]] = MAKETXT_ALIASES) -> pd.DataFrame:
    """One row per (make, NHTSA MAKETXT spelling) for the makes in india_makes.csv.

    nhtsa_make is the primary spelling (first alias), used as `make` downstream.
    """
    with makes_csv.open(newline="", encoding="utf-8") as fh:
        makes = [row["make"] for row in csv.DictReader(fh)]
    unmapped = [m for m in makes if m not in aliases]
    if unmapped:
        raise ValueError(f"no NHTSA MAKETXT spelling recorded for: {unmapped}")
    return pd.DataFrame([(m, a, aliases[m][0]) for m in makes for a in aliases[m]],
                        columns=["make", "maketxt", "nhtsa_make"])


def run_sql(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """Execute sql/<name>.sql against the connection."""
    con.execute((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8"))


def record_issues(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Count missing, unknown and excluded values; store as load_issues and log them."""
    raw_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'nhtsa_raw' ORDER BY ordinal_position").fetchall()]
    null_counts = con.execute(
        "SELECT " + ", ".join(f'count(*) - count("{c}")' for c in raw_cols)
        + " FROM nhtsa_raw").fetchone()

    rows = [("nhtsa_raw", c, "empty in source, stored as NULL", n)
            for c, n in zip(raw_cols, null_counts)]
    rows += [
        ("nhtsa_raw", "RCDATE", "no parseable report date, excluded from nhtsa_focus",
         con.execute("SELECT count(*) FROM nhtsa_raw "
                     "WHERE try_strptime(RCDATE, '%Y%m%d') IS NULL").fetchone()[0]),
        ("nhtsa_focus", "model_year", "YEARTXT 9999 = unknown, stored as NULL",
         con.execute("SELECT count(*) FROM nhtsa_focus "
                     "WHERE YEARTXT = '9999'").fetchone()[0]),
        ("nhtsa_focus", "model_year", "YEARTXT not a year, stored as NULL",
         con.execute("SELECT count(*) FROM nhtsa_focus WHERE model_year IS NULL "
                     "AND YEARTXT IS DISTINCT FROM '9999'").fetchone()[0]),
        ("nhtsa_focus", "RCDATE", "campaign has more than one report date, min() used",
         con.execute("SELECT count(*) FROM (SELECT CAMPNO FROM nhtsa_focus "
                     "GROUP BY CAMPNO HAVING count(DISTINCT RCDATE) > 1)").fetchone()[0]),
    ]
    issues = pd.DataFrame([r for r in rows if r[3]],
                          columns=["table_name", "column_name", "issue", "n_rows"])
    con.register("issues_df", issues)
    con.execute("CREATE OR REPLACE TABLE load_issues AS "
                "SELECT table_name, column_name, issue, CAST(n_rows AS BIGINT) AS n_rows "
                "FROM issues_df")
    con.unregister("issues_df")
    for t, c, issue, n in issues.itertuples(index=False):
        log.info("%s.%s: %s (%d rows)", t, c, issue, n)
    return issues


def build(db_path: Path = DB_PATH, zip_path: Path = ZIP_PATH,
          rcl_path: Path = RCL_PATH, makes_csv: Path = MAKES_CSV) -> dict[str, int]:
    """Build every table in db_path and return their row counts."""
    names = read_field_names(rcl_path)
    raw = read_nhtsa(zip_path, names)
    makes = focus_make_table(makes_csv)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.register("raw_df", raw)
        cols = ", ".join(f'CAST("{n}" AS VARCHAR) AS "{n}"' for n in names)
        con.execute(f"CREATE OR REPLACE TABLE nhtsa_raw AS SELECT {cols} FROM raw_df")
        con.unregister("raw_df")

        con.register("makes_df", makes)
        con.execute("CREATE OR REPLACE TABLE focus_makes AS SELECT * FROM makes_df")
        con.unregister("makes_df")

        for (alias,) in con.execute(
                "SELECT f.maketxt FROM focus_makes f WHERE NOT EXISTS "
                "(SELECT 1 FROM nhtsa_raw r WHERE upper(trim(r.MAKETXT)) = f.maketxt)"
        ).fetchall():
            log.warning("MAKETXT spelling %r not found in the NHTSA file", alias)

        run_sql(con, "nhtsa_focus")
        run_sql(con, "nhtsa_campaigns")
        record_issues(con)

        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("nhtsa_raw", "focus_makes", "nhtsa_focus",
                          "nhtsa_campaigns", "load_issues")}
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = build()
    print(f"\nwrote {DB_PATH.relative_to(ROOT)}")
    for table, n in counts.items():
        print(f"  {table:<16} {n:>8,} rows")


if __name__ == "__main__":
    main()
