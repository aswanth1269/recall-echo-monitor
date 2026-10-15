"""Make CSV files that people open in Excel safe from formula injection.

A spreadsheet runs any cell that starts with = + - @ (or a tab / carriage
return) as a formula. Review and label files carry NHTSA, SIAM and LLM text,
so a crafted value could run a formula when the file is opened. Following the
OWASP guidance, such cells get a leading apostrophe, which Excel shows as
plain text. Real numbers such as -12.5 are left alone.

Use only on files meant for people, never on files the pipeline reads back:
the apostrophe would change the data it re-reads.
"""

from __future__ import annotations

import re

import pandas as pd

TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")


def excel_safe_value(value: object) -> object:
    """'=SUM(A1)' -> "'=SUM(A1)"; numbers, blanks and other values unchanged."""
    if not isinstance(value, str) or not value or not value.startswith(TRIGGERS):
        return value
    if _NUMBER.fullmatch(value.strip()):
        return value
    return "'" + value


def excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of df with every text cell passed through excel_safe_value."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(excel_safe_value)
    return out
