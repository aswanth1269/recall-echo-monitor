"""Write a blind labelling sheet of 30 candidate pairs.

Pairs from candidate_pairs are ordered by score and split into three equal
thirds (top, middle, bottom); 10 are drawn from each with a fixed seed, then
the 30 rows are shuffled with the same seed so their order says nothing about
their score.

data/clean/label_sheet.csv holds only what a person needs to judge "is this
the same defect?" - no score, rank or band - plus an empty `label` column.
data/clean/label_sheet_key.csv keeps each pair's score, rank and band at the
time of sampling, for checking the labels afterwards. Do not open it until
labelling is done.

An existing sheet that already has labels is never overwritten unless
--force is given.

--human-labels writes data/reference/human_labels.csv (pair_id, human_label,
note) from the sheet, in the sheet's order and with both columns empty, so a
person can label blind. It reads only the sheet - never ground_truth.csv - and
never overwrites an existing file.

Run:  .venv/Scripts/python.exe -m src.make_label_sheet [--force]
      .venv/Scripts/python.exe -m src.make_label_sheet --human-labels
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.csv_safe import excel_safe
from src.load_nhtsa import DB_PATH

ROOT = Path(__file__).resolve().parents[1]
SHEET_CSV = ROOT / "data/clean/label_sheet.csv"
KEY_CSV = ROOT / "data/clean/label_sheet_key.csv"
HUMAN_LABELS_CSV = ROOT / "data/reference/human_labels.csv"
HUMAN_LABEL_COLUMNS = ["pair_id", "human_label", "note"]
SEED = 20261015
PER_BAND = 10
BANDS = ("top", "middle", "bottom")
SHEET_COLUMNS = ["pair_id", "make", "us_report_date", "us_component", "us_defect_text",
                 "siam_date", "siam_model", "siam_defect_text", "label"]


def load_pairs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every candidate pair with the text a labeller needs, plus score and rank."""
    return con.execute("""
        SELECT p.CAMPNO, p.make, p.csv_row, p.score, p.rank,
               p.report_date                              AS us_report_date,
               array_to_string(c.components, '; ')        AS us_component,
               c.DESC_DEFECT                              AS us_defect_text,
               s.recall_date                              AS siam_date,
               s.model                                    AS siam_model,
               s.defect_description                       AS siam_defect_text
        FROM candidate_pairs AS p
        JOIN india_relevant_campaigns AS c USING (CAMPNO, make)
        JOIN siam_recalls AS s USING (csv_row)""").df()


def assign_thirds(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add band = top / middle / bottom: three equal-size thirds by score.

    Ordered by score, then CAMPNO and csv_row, so tied scores (many are 0)
    still split deterministically.
    """
    ordered = pairs.sort_values(["score", "CAMPNO", "csv_row"],
                                ascending=[False, True, True]).reset_index(drop=True)
    band = np.empty(len(ordered), dtype=object)
    for name, idx in zip(BANDS, np.array_split(np.arange(len(ordered)), 3)):
        band[idx] = name
    return ordered.assign(band=band)


def sample_pairs(pairs: pd.DataFrame, per_band: int = PER_BAND, seed: int = SEED) -> pd.DataFrame:
    """per_band pairs from each third, then shuffled so order hides the band."""
    banded = assign_thirds(pairs)
    picks = []
    for name in BANDS:
        band = banded[banded.band == name]
        if len(band) < per_band:
            raise ValueError(f"only {len(band)} pairs in the {name} third; need {per_band}")
        picks.append(band.sample(n=per_band, random_state=seed))
    return pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)


def pair_id(campno: str, csv_row: int) -> str:
    """Stable id that maps back to candidate_pairs and reveals nothing about score."""
    return f"{campno}-{int(csv_row)}"


def to_sheet(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(label sheet without score/rank/band, key with them)."""
    ids = [pair_id(c, r) for c, r in zip(sample.CAMPNO, sample.csv_row)]
    sheet = sample.assign(pair_id=ids, label="")[SHEET_COLUMNS]
    key = sample.assign(pair_id=ids)[["pair_id", "CAMPNO", "make", "csv_row",
                                      "score", "rank", "band"]]
    return sheet, key


def has_labels(path: Path) -> bool:
    """True if a sheet exists and any label cell is filled in."""
    if not path.exists():
        return False
    labels = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return "label" in labels and labels["label"].str.strip().ne("").any()


def build(db_path: Path = DB_PATH, sheet_csv: Path = SHEET_CSV, key_csv: Path = KEY_CSV,
          seed: int = SEED, force: bool = False) -> pd.DataFrame:
    """Write the sheet and key; return the key. Refuses to replace filled-in labels."""
    if has_labels(sheet_csv) and not force:
        raise FileExistsError(f"{sheet_csv} already has labels - not overwriting "
                              "(use --force to replace it)")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        pairs = load_pairs(con)
    finally:
        con.close()
    sheet, key = to_sheet(sample_pairs(pairs, seed=seed))
    sheet_csv.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows curly quotes and dashes in the defect text correctly;
    # excel_safe so no recall text can run as a formula when the sheet is opened
    excel_safe(sheet).to_csv(sheet_csv, index=False, encoding="utf-8-sig")
    key.to_csv(key_csv, index=False, encoding="utf-8")
    return key


def human_labels_template(sheet: pd.DataFrame) -> pd.DataFrame:
    """pair_id in the sheet's (blind) order, with empty human_label and note."""
    return pd.DataFrame({"pair_id": sheet.pair_id.astype(str).str.strip(),
                         "human_label": "", "note": ""})[HUMAN_LABEL_COLUMNS]


def write_human_labels(sheet_csv: Path = SHEET_CSV, out_csv: Path = HUMAN_LABELS_CSV) -> int:
    """Write the empty human-label file from the sheet; refuse to overwrite. Returns rows."""
    if out_csv.exists():
        raise FileExistsError(f"{out_csv.name} already exists - it may hold your labels, "
                              "so it is never overwritten")
    sheet = pd.read_csv(sheet_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    out = human_labels_template(sheet)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    return len(out)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if "--human-labels" in args:
        try:
            n = write_human_labels()
        except FileExistsError as exc:
            print(exc)
            raise SystemExit(1) from None
        print(f"wrote {n} pairs to data/reference/human_labels.csv - fill human_label with "
              "same or different, reading the texts in data/clean/label_sheet.csv")
        return
    key = build(force="--force" in args)
    print(f"wrote {len(key)} pairs to {SHEET_CSV.relative_to(ROOT)} "
          f"({', '.join(f'{n} {b}' for b, n in key.band.value_counts().reindex(BANDS).items())}; "
          f"order shuffled, no scores)")
    print(f"scores and bands are in {KEY_CSV.relative_to(ROOT)} - open it only after labelling")


if __name__ == "__main__":
    main()
