"""Fetch the over-the-air (OTA) remedy flag for each India-relevant campaign.

The NHTSA flat file has no OTA field, so for every distinct CAMPNO in
india_relevant_campaigns this calls
    https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber=<CAMPNO>
waiting 0.3 s between real requests. Each successful JSON response (including
"Count": 0) is cached in data/interim/nhtsa_api/<CAMPNO>.json, so a rerun
makes no requests for campaigns already fetched. Failed requests are not
cached and are retried on the next run.

The API returns one result per make/model/year, each with a boolean
overTheAirUpdate. A campaign's ota is that value when every result that has
it agrees; it is NULL - never guessed - when the API has no results, no
result carries the field, the results disagree, or the request failed. The
null count and reasons are logged.

The API's ReportReceivedDate is day-first (e.g. 20/04/2022) and is not used:
report_date stays the one from the flat file.

Writes table campaign_ota (CAMPNO, ota).

Run:  .venv/Scripts/python.exe -m src.fetch_ota
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

from src.load_nhtsa import DB_PATH

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data/interim/nhtsa_api"
API_URL = "https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber={}"
PAUSE_SECONDS = 0.3
TIMEOUT_SECONDS = 30

log = logging.getLogger(__name__)


def campaign_numbers(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Distinct CAMPNOs (a campaign covering Audi and VW appears once)."""
    return [r[0] for r in con.execute(
        "SELECT DISTINCT CAMPNO FROM india_relevant_campaigns ORDER BY CAMPNO").fetchall()]


# NHTSA campaign numbers: 2-digit year, a letter, then 6 digits or capital
# letters (e.g. 24V200000, and a handful like 21V00M000). Checked before the
# number becomes a filename, so no value can point outside the cache folder.
_CAMPNO = re.compile(r"[0-9]{2}[A-Z][0-9A-Z]{6}")


def cache_path(campno: str, cache_dir: Path = CACHE_DIR) -> Path:
    if not isinstance(campno, str) or not _CAMPNO.fullmatch(campno):
        raise ValueError(f"not a valid NHTSA campaign number: {campno!r}")
    return cache_dir / f"{campno}.json"


def http_get_json(url: str) -> dict:
    """GET a URL and parse JSON. Raises on HTTP errors, timeouts and bad JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "recall-echo-monitor/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_campaign(campno: str, cache_dir: Path = CACHE_DIR,
                   get: Callable[[str], dict] = http_get_json) -> tuple[dict, bool]:
    """(payload, from_cache). Cache hit makes no request; a fresh payload is cached."""
    path = cache_path(campno, cache_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    payload = get(API_URL.format(urllib.parse.quote(campno)))
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(f"unexpected API response for {campno}: {str(payload)[:200]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, False


def ota_from_response(payload: dict) -> tuple[bool | None, str | None]:
    """(ota, None) or (None, reason). Only real booleans count; nothing is inferred."""
    results = payload.get("results") or []
    if not results:
        return None, "API has no results for this campaign"
    values = {r.get("overTheAirUpdate") for r in results if isinstance(r, dict)}
    flags = {v for v in values if isinstance(v, bool)}
    if not flags:
        return None, "no result carries a true/false overTheAirUpdate"
    if len(flags) > 1:
        return None, "results disagree on overTheAirUpdate"
    return flags.pop(), None


def fetch_all(campnos: list[str], cache_dir: Path = CACHE_DIR,
              get: Callable[[str], dict] = http_get_json,
              sleep: Callable[[float], None] = time.sleep,
              pause: float = PAUSE_SECONDS) -> tuple[pd.DataFrame, int]:
    """(rows of CAMPNO, ota, null_reason; number of real requests made)."""
    rows, requests = [], 0
    for campno in campnos:
        try:
            cached = cache_path(campno, cache_dir).exists()
        except ValueError as exc:                # never used as a filename, never fetched
            rows.append((campno, None, f"invalid campaign number: {exc}"[:200]))
            continue
        if not cached and requests:
            sleep(pause)                         # 0.3 s between real requests only
        try:
            payload, _ = fetch_campaign(campno, cache_dir, get)
            ota, reason = ota_from_response(payload)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            ota, reason = None, f"request failed: {type(exc).__name__}: {exc}"[:200]
        if not cached:
            requests += 1
        rows.append((campno, ota, reason))
    return pd.DataFrame(rows, columns=["CAMPNO", "ota", "null_reason"]), requests


def build(db_path: Path = DB_PATH, cache_dir: Path = CACHE_DIR,
          get: Callable[[str], dict] = http_get_json,
          sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    """Fetch (or read cached) OTA flags and write campaign_ota."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        campnos = campaign_numbers(con)
    finally:
        con.close()
    rows, requests = fetch_all(campnos, cache_dir, get, sleep)

    nulls = rows[rows.ota.isna()]
    reasons = Counter(r.split(":")[0] if r.startswith("request failed") else r
                      for r in nulls.null_reason)
    log.info("campaign_ota: %d campaigns, %d requests made, %d from cache",
             len(rows), requests, len(rows) - requests)
    log.info("ota is NULL for %d of %d campaigns", len(nulls), len(rows))
    for reason, n in reasons.most_common():
        log.info("  %d: %s", n, reason)

    table = rows[["CAMPNO", "ota"]].astype({"ota": "boolean"})
    con = duckdb.connect(str(db_path))
    try:
        con.register("ota_df", table)
        con.execute("CREATE OR REPLACE TABLE campaign_ota AS "
                    "SELECT CAMPNO, CAST(ota AS BOOLEAN) AS ota FROM ota_df ORDER BY CAMPNO")
        con.unregister("ota_df")
    finally:
        con.close()
    return {"campaigns": len(rows), "requests": requests, "nulls": len(nulls),
            "true": int(table.ota.eq(True).sum()), "false": int(table.ota.eq(False).sum()),
            "null_reasons": dict(reasons)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = build()
    print(f"campaign_ota: {s['campaigns']} campaigns | ota true={s['true']} false={s['false']} "
          f"null={s['nulls']} | requests made: {s['requests']}")


if __name__ == "__main__":
    main()
