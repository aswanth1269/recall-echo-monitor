"""Parse the SIAM voluntary-recall table into data/raw/siam_recalls.csv.

The SIAM portal has no API and no CSV export, so the whole table is rendered
server-side into a single <table> on first page load. We keep a byte-for-byte
HTML snapshot in data/raw/ so the CSV is always reproducible from what we saw
on the retrieval date, even after SIAM edits or re-sorts the live page.

Run:  python src/parse_siam.py [--refresh]
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from datetime import date
from pathlib import Path

SOURCE_URL = "https://www.siam.in/siam-voluntary-recall.aspx"
RETRIEVED_ON = "2026-10-13"

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/raw/siam_voluntary_recall_snapshot.html"
OUT_PICKED = ROOT / "data/raw/siam_recalls.csv"
OUT_ALL = ROOT / "data/raw/siam_recalls_all.csv"
MAKES_CSV = ROOT / "data/raw/india_makes.csv"

# Sentinel: this entity covers several brands, so the brand must come from
# the model text rather than the entity name.
FROM_MODEL = "__from_model__"

# SIAM files recalls under the legal entity, not the consumer-facing brand.
# One entity (Skoda Auto Volkswagen India) covers five brands, so those rows
# have to be resolved from the model text instead -- see BRAND_TOKENS.
ENTITY_BRAND = {
    "Toyota Kirloskar Motor Pvt Ltd": "Toyota",
    "Honda Cars India Ltd": "Honda",
    "Hyundai Motor India Ltd": "Hyundai",
    "Kia India Private Ltd": "Kia",
    "Nissan Motor India Pvt Ltd": "Nissan",
    "Mercedes Benz India Pvt Ltd": "Mercedes-Benz",
    "BMW India Pvt Ltd": "BMW",
    "Volvo Auto India Pvt Ltd": "Volvo",
    "SkodaAuto India Pvt Ltd": "Skoda",
    "Skoda Auto Volkswagen India Pvt. Ltd.": FROM_MODEL,
}

# Longest-match-wins model tokens -> brand. Covers the VW-group split and the
# sub-brands that ride along inside another entity's rows (Lexus under Toyota
# Kirloskar, Mini under BMW India, Maybach under Mercedes).
BRAND_TOKENS: dict[str, tuple[str, ...]] = {
    "Audi": ("audi", "etron", "e-tron"),
    "Skoda": ("skoda", "škoda", "octavia", "laura", "superb", "rapid",
              "fabia", "yeti", "kodiaq", "kushaq", "slavia", "kylaq"),
    "Volkswagen": ("volkswagen", "polo", "vento", "jetta", "passat", "pasat",
                   "taigun", "virtus", "tiguan", "touareg", "ameo", "golf",
                   "tayron"),
    "Porsche": ("porsche", "porche", "cayenne", "cayman", "boxster", "macan",
                "panamera", "taycan", "911"),
    "Lamborghini": ("lamborghini", "aventador", "huracan", "urus"),
    "Lexus": ("lexus",),
    "Mini": ("mini",),
    "Maybach": ("maybach",),
}

HEADERS = ["#", "manufacturer_name", "voluntary_recall_date", "model", "variant",
           "status", "target_recall_vehicles", "manufactured_from",
           "manufactured_to", "defect_description"]


def _clean(cell: str) -> str:
    """Strip tags/entities and collapse whitespace in one table cell."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", cell))
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_rows(markup: str) -> list[list[str]]:
    table = re.search(r"<table[^>]*>.*?</table>", markup, re.S)
    if not table:
        raise SystemExit("No <table> found -- SIAM may have changed the page.")
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(0), re.S):
        cells = [_clean(c) for c in
                 re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(cells) == len(HEADERS):
            rows.append(cells)
    if not rows:
        raise SystemExit("Table found but no 10-column rows parsed.")
    return rows[1:]  # drop the header row


def brands_in(model_text: str) -> list[str]:
    low = model_text.lower()
    return sorted({b for b, toks in BRAND_TOKENS.items()
                   if any(re.search(rf"\b{re.escape(t)}", low) for t in toks)})


def resolve(entity: str, model_text: str) -> tuple[str, str, str]:
    """-> (make, brands_detected, how_resolved)"""
    found = brands_in(model_text)
    joined = "|".join(found)
    entity = entity.strip()

    if entity not in ENTITY_BRAND:
        # A SIAM filer outside our 10 picked makes (Maruti, Mahindra, the
        # two-wheeler makers, ...). Expected, not a data problem.
        return "OUT_OF_SCOPE", joined, "out_of_scope"

    entity_brand = ENTITY_BRAND[entity]
    if entity_brand is not FROM_MODEL:
        return entity_brand, joined, "entity"

    # Shared VW-group entity: the model text is the only brand signal.
    if len(found) == 1:
        return found[0], joined, "model_token"
    if len(found) > 1:
        return found[0], joined, "ambiguous_multi_brand"
    return "UNRESOLVED", joined, "unresolved"


def picked_makes() -> list[str]:
    with MAKES_CSV.open(newline="", encoding="utf-8") as fh:
        return [r["make"] for r in csv.DictReader(fh)]


def refresh_snapshot() -> None:
    import urllib.request
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        SNAPSHOT.write_bytes(resp.read())
    print(f"snapshot refreshed -> {SNAPSHOT.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the SIAM page before parsing")
    if ap.parse_args().refresh:
        refresh_snapshot()

    rows = parse_rows(SNAPSHOT.read_text(encoding="utf-8", errors="replace"))
    out_cols = (["make", "brands_detected", "make_resolution"] + HEADERS[1:]
                + ["source_url", "retrieved_on"])

    enriched, counts = [], {}
    for r in rows:
        make, found, how = resolve(r[1], r[3])
        counts[how] = counts.get(how, 0) + 1
        enriched.append([make, found, how] + r[1:] + [SOURCE_URL, RETRIEVED_ON])

    def write(path: Path, data: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
            w.writerow(out_cols)
            w.writerows(data)

    wanted = set(picked_makes())
    kept = [e for e in enriched if e[0] in wanted]
    write(OUT_ALL, enriched)
    write(OUT_PICKED, kept)

    print(f"parsed {len(enriched)} SIAM rows; resolution: {counts}")
    print(f"  {OUT_ALL.relative_to(ROOT)}    all brands   ({len(enriched)} rows)")
    print(f"  {OUT_PICKED.relative_to(ROOT)}  chosen makes ({len(kept)} rows)")
    flagged = [e for e in enriched
               if e[2] in ("unresolved", "ambiguous_multi_brand")]
    if flagged:
        print()
        print(f"needs a human eye ({len(flagged)}):")
        for e in flagged:
            print(f"  {e[2]:<22} {e[4]:>12} | model={e[5][:40]!r}")


if __name__ == "__main__":
    main()
