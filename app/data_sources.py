"""Read the pipeline's outputs and hand them to the dashboard in the shape it draws.

The pipeline writes (src/kpis.py, src/evaluate.py):
  data/clean/kpi_echo_rate.csv          level (make|segment|all), grp, n, echoing, echo_rate
  data/clean/kpi_echo_lag.csv           level, grp, n, median_days, p25_days, p75_days, india_first
  data/clean/kpi_echo_by_component.csv  component_group, n, echoing, echo_rate
  data/clean/kpi_ota.csv                part ota_share: level, grp, n, n_known_ota, ota_true, ota_share
                                        part echo_by_ota: grp (ota|not_ota|unknown), n, echoing, echo_rate
  data/clean/echo_explorer.csv          one row per campaign x make, best pair or no_candidate
  reports/eval_summary.csv              matcher, n, positives, tp, fp, fn, tn, precision, recall, f1, *_n

Each function returns None when its file does not exist yet, and raises
FormatError (naming the file and the missing columns) when it exists in an
unexpected shape - so the page says what is wrong instead of crashing.
Rates are passed through as written: the pipeline already leaves them empty
where n < 10, and nothing here fills a missing value.

The project root defaults to the folder above app/; RECALL_ECHO_ROOT overrides
it (used by tests).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

LEVELS = {"all": "overall", "make": "make", "segment": "segment", "report_year": "report_year"}
OTA_GROUPS = {"ota": "OTA", "not_ota": "Not OTA", "unknown": "Unknown"}
FILES = {
    "echo_rate": "data/clean/kpi_echo_rate.csv",
    "echo_lag": "data/clean/kpi_echo_lag.csv",
    "components": "data/clean/kpi_echo_by_component.csv",
    "ota": "data/clean/kpi_ota.csv",
    "explorer": "data/clean/echo_explorer.csv",
    "eval_summary": "reports/eval_summary.csv",
}
_PLAIN = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
_FLEX = {"'": "['‘’]", '"': '["“”]', "-": "[-–—]"}


class FormatError(ValueError):
    """A pipeline output exists but does not have the columns the dashboard needs."""


def root() -> Path:
    """Project root, read at call time so tests can point it elsewhere."""
    return Path(os.environ.get("RECALL_ECHO_ROOT", Path(__file__).resolve().parents[1]))


def path(name: str) -> Path:
    return root() / FILES[name]


def stamp() -> tuple:
    """(file, mtime) for every output: changes whenever the pipeline rewrites one."""
    return tuple((name, path(name).stat().st_mtime if path(name).exists() else None)
                 for name in FILES)


def _read(name: str, required: list[str]) -> pd.DataFrame | None:
    p = path(name)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FormatError(f"{FILES[name]} is missing columns {missing}; "
                          "re-run the pipeline step that writes it")
    return df


def _regroup(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline level/grp -> dashboard group_type/group ('all' -> 'overall')."""
    out = df.rename(columns={"grp": "group"})
    level = out.pop("level")
    out.insert(0, "group_type", level.map(LEVELS).fillna(level))
    return out.reset_index(drop=True)


def echo_rate() -> pd.DataFrame | None:
    df = _read("echo_rate", ["level", "grp", "n", "echoing", "echo_rate"])
    return None if df is None else _regroup(df)


def echo_lag() -> pd.DataFrame | None:
    df = _read("echo_lag", ["level", "grp", "n", "median_days", "p25_days", "p75_days",
                            "india_first"])
    return None if df is None else _regroup(df)


def components() -> pd.DataFrame | None:
    return _read("components", ["component_group", "n", "echoing", "echo_rate"])


def ota_share() -> pd.DataFrame | None:
    """group_type, group, n = campaigns with a known OTA flag, ota_n, ota_share, n_all.

    The share is out of campaigns whose flag is known; NULL flags are unknown,
    not "not OTA", so n_all keeps the full count beside it.
    """
    df = _read("ota", ["part", "level", "grp", "n", "n_known_ota", "ota_true", "ota_share"])
    if df is None:
        return None
    share = _regroup(df[df.part == "ota_share"].drop(columns="part"))
    return pd.DataFrame({"group_type": share.group_type, "group": share.group,
                         "n": share.n_known_ota, "ota_n": share.ota_true,
                         "ota_share": share.ota_share, "n_all": share.n})


def ota_echo() -> pd.DataFrame | None:
    """ota (OTA | Not OTA | Unknown), n, echoing, echo_rate."""
    df = _read("ota", ["part", "grp", "n", "echoing", "echo_rate"])
    if df is None:
        return None
    by = df[df.part == "echo_by_ota"]
    return pd.DataFrame({"ota": by.grp.map(OTA_GROUPS).fillna(by.grp), "n": by.n,
                         "echoing": by.echoing, "echo_rate": by.echo_rate}).reset_index(drop=True)


def explorer() -> pd.DataFrame | None:
    return _read("explorer", ["CAMPNO", "make", "segment", "report_date", "component",
                              "us_defect", "label", "india_recall_date", "india_model",
                              "india_defect", "lag_days", "confidence", "us_evidence",
                              "india_evidence"])


def eval_summary() -> pd.DataFrame | None:
    return _read("eval_summary", ["matcher", "n", "positives", "tp", "fp", "fn", "tn",
                                  "precision", "recall", "f1", "f1_n"])


def text(value: object) -> str:
    """Display string for a cell: '' for missing values, never the word 'nan'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def find_evidence(source: object, phrase: object) -> tuple[int, int] | None:
    """(start, end) of phrase in source, or None.

    Matches the way src/matcher.py checks evidence: case-insensitive, any run
    of whitespace, and typographic quotes/dashes equal to plain ones.
    """
    src, ph = text(source), text(phrase)
    words = "".join(_PLAIN.get(ch, ch) for ch in ph).split()
    if not src or not words:
        return None
    pattern = r"\s+".join("".join(_FLEX.get(ch, re.escape(ch)) for ch in w) for w in words)
    m = re.search(pattern, src, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None
