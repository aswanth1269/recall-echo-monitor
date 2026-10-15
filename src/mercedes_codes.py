"""Build the Mercedes platform-code reference from SIAM's own model text.

SIAM lists Mercedes recalls by internal platform code ("205 (C-Class)",
"GLC (253 Platform)", "Model 206 (C 220 d)") rather than by model name.
This module extracts only the code -> name pairs that SIAM itself prints,
so every pair in data/reference/mercedes_platform_codes.csv is backed by
the SIAM rows it came from (siam_csv_rows, source_url, retrieved_on).

Nothing is inferred: a code with no name beside it, or a 6-digit model
number such as 206087, is not mapped. Those segments go to
data/clean/mercedes_code_issues.csv with a reason.

Run:  .venv/Scripts/python.exe -m src.mercedes_codes
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

from src.load_nhtsa import DB_PATH

ROOT = Path(__file__).resolve().parents[1]
REF_CSV = ROOT / "data/reference/mercedes_platform_codes.csv"
ISSUES_CSV = ROOT / "data/clean/mercedes_code_issues.csv"
MAKE = "MERCEDES-BENZ"

# Names may not contain brackets: a cell with broken brackets such as
# "213 (E-CLASS0, 223 (S-CLASS, 253 (GLC)" must fail, not yield "223 (S-CLASS".
_CODE_FIRST = re.compile(r"^(?:model\s*)?(\d{3})\s*\(([^()]+)\)$", re.I)
_NAME_FIRST = re.compile(
    r"^([^()]+?)\s*\(\s*(?:model\s*|w)?(\d{3})(?:\s*(?:vehicle\s+)?platform)?\s*\)$", re.I)
_COLON = re.compile(r"^model\s*(\d{3})\s*:\s*([^()]+)$", re.I)
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "�": "-"})
_INNER_SPLIT = re.compile(r"\s*(?:,|/|\band\b)\s*", re.I)


def split_segments(text: str | None) -> list[str]:
    """Split a SIAM model cell on ; , & \\ / and 'and' - but never inside brackets."""
    if not text:
        return []
    segments, buf, depth, i = [], [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if depth == 0 and (ch in ";,&\\/"):
            segments.append("".join(buf)); buf = []; i += 1
            continue
        if depth == 0 and text[i:i + 5].lower() == " and ":
            segments.append("".join(buf)); buf = []; i += 5
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def normalise_name(name: str) -> str:
    """Make spelling variants of one name equal ('C – Class', 'C - Class' -> 'C-CLASS').

    Only spacing, case, the source's replacement character and 'X CLASS' are
    touched (en/em dashes, minus signs and U+FFFD become '-'); typos such as
    'E-CLASS0' or 'AMD GLE' are left as SIAM wrote them.
    """
    s = name.translate(_DASHES).upper()
    s = re.sub(r"\s*-\s*", "-", s)
    s = " ".join(s.split())
    s = re.sub(r"\b([A-Z]{1,3}) CLASS\b", r"\1-CLASS", s)
    return s.strip(" -")


def parse_segment(segment: str) -> list[tuple[str, str]]:
    """(code, normalised name) pairs stated in one segment, or [] if none are stated."""
    seg = " ".join(segment.split())
    if m := _CODE_FIRST.match(seg):
        code, names = m.group(1), _INNER_SPLIT.split(m.group(2))
    elif m := _COLON.match(seg):
        code, names = m.group(1), [m.group(2)]
    elif m := _NAME_FIRST.match(seg):
        code, names = m.group(2), [m.group(1)]
    else:
        return []
    return [(code, normalise_name(n)) for n in names
            if re.search(r"[A-Za-z]", n)]


def _issue_reason(segment: str) -> str:
    if re.search(r"\d{4,}", segment):
        return "number longer than 3 digits; not cut down to a platform code"
    if segment.count("(") != segment.count(")"):
        return "unbalanced brackets in source text"
    return "3-digit number with no model name stated beside it"


def extract_pairs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rows needs csv_row, model, source_url, retrieved_on. Returns (pairs, issues)."""
    seen: dict[tuple[str, str], dict] = defaultdict(lambda: {"rows": set()})
    issues = []
    for r in rows.itertuples(index=False):
        for seg in split_segments(r.model):
            pairs = parse_segment(seg)
            if not pairs:
                if re.search(r"\d{3}", seg):
                    issues.append((r.csv_row, seg, _issue_reason(seg)))
                continue
            for code, name in pairs:
                entry = seen[(code, name)]
                entry["rows"].add(int(r.csv_row))
                entry.setdefault("source_text", seg)
                entry.setdefault("source_url", r.source_url)
                entry.setdefault("retrieved_on", str(r.retrieved_on))

    pairs_df = pd.DataFrame(
        [(code, name, len(e["rows"]), ";".join(map(str, sorted(e["rows"]))),
          e["source_text"], e["source_url"], e["retrieved_on"])
         for (code, name), e in seen.items()],
        columns=["platform_code", "model_name", "n_siam_rows", "siam_csv_rows",
                 "source_text", "source_url", "retrieved_on"],
    ).sort_values(["platform_code", "model_name"], ignore_index=True)
    issues_df = pd.DataFrame(issues, columns=["csv_row", "segment", "reason"])
    return pairs_df, issues_df


def build(db_path: Path = DB_PATH, out_csv: Path = REF_CSV,
          issues_csv: Path = ISSUES_CSV) -> tuple[int, int]:
    """Read Mercedes rows from siam_recalls, write both CSVs, return (pairs, issues)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT csv_row, model, source_url, retrieved_on FROM siam_recalls "
            "WHERE make = ? ORDER BY csv_row", [MAKE]).df()
    finally:
        con.close()
    pairs, issues = extract_pairs(rows)
    for path, df in ((out_csv, pairs), (issues_csv, issues)):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    return len(pairs), len(issues)


def main() -> None:
    n_pairs, n_issues = build()
    print(f"wrote {n_pairs} code-name pairs to {REF_CSV.relative_to(ROOT)}")
    print(f"wrote {n_issues} unparsed segments to {ISSUES_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
