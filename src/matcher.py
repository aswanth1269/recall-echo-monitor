"""Judge candidate pairs with the LLM, then verify its evidence in Python.

For each (NHTSA campaign, SIAM row) pair the LLM, called through src/llm.py
(10 pairs per call, temperature 0), returns:
    pair_id, label ("same" | "different"), confidence (0-1),
    us_evidence, india_evidence - each an exact phrase copied from that side.

"Same" means the same defect on the same model family; a different model-year
range is fine.

The LLM's answer is not trusted on its own. Python checks that each evidence
phrase really appears in the text the LLM was shown (case-insensitive,
whitespace-normalised, typographic quotes/dashes treated as plain) and is at
least 3 words long. Any failure - missing quote, too-short quote, bad label,
bad confidence, no answer for the pair - sets label = "needs_review" with a
review_reason. Pairs with no defect text on a side are not sent at all.

Writes table `matches`. Default scope is the pairs in
data/reference/ground_truth.csv; --all judges every candidate pair.

Run:  .venv/Scripts/python.exe -m src.matcher [--all]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd

from src import llm
from src.llm import batched, complete_json
from src.load_nhtsa import DB_PATH

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_CSV = ROOT / "data/reference/ground_truth.csv"
BATCH = 10
MIN_EVIDENCE_WORDS = 3
LABELS = ("same", "different")
NEEDS_REVIEW = "needs_review"

log = logging.getLogger(__name__)

SYSTEM = ("You compare vehicle safety recalls from the US (NHTSA) and India (SIAM). "
          "You answer only with JSON, and every evidence phrase you give is copied "
          "exactly, character for character, from the text it comes from.")

# Typographic characters an LLM tends to retype as plain ones; same text either way.
_PLAIN = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-"})


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def us_text(row) -> str:
    """The exact US text shown to the LLM (components, defect with model list, consequence)."""
    comps = row.components
    comps = "; ".join(map(str, comps)) if comps is not None and len(comps) else ""
    parts = [("Component", comps), ("Defect", _clean(row.DESC_DEFECT)),
             ("Consequence", _clean(row.CONEQUENCE_DEFECT))]
    return "\n".join(f"{name}: {text}" for name, text in parts if text)


def india_text(row) -> str:
    """The exact SIAM text shown to the LLM."""
    parts = [("Model", _clean(row.model)), ("Defect", _clean(row.defect_description))]
    return "\n".join(f"{name}: {text}" for name, text in parts if text)


def load_pairs(con: duckdb.DuckDBPyConnection, keys: pd.DataFrame) -> pd.DataFrame:
    """keys has CAMPNO, csv_row. Returns one row per pair with both texts."""
    con.register("pair_keys", keys[["CAMPNO", "csv_row"]])
    try:
        df = con.execute("""
            SELECT k.CAMPNO, s.make, CAST(k.csv_row AS INTEGER) AS csv_row,
                   c.components, c.DESC_DEFECT, c.CONEQUENCE_DEFECT,
                   s.model, s.defect_description
            FROM pair_keys AS k
            JOIN siam_recalls AS s ON s.csv_row = CAST(k.csv_row AS INTEGER)
            JOIN india_relevant_campaigns AS c ON c.CAMPNO = k.CAMPNO AND c.make = s.make
            ORDER BY k.CAMPNO, csv_row""").df()
    finally:
        con.unregister("pair_keys")
    if len(df) < len(keys):
        log.warning("%d requested pairs not found in the database", len(keys) - len(df))
    return pd.DataFrame({
        "pair_id": [f"{c}-{r}" for c, r in zip(df.CAMPNO, df.csv_row)],
        "CAMPNO": df.CAMPNO, "make": df.make, "csv_row": df.csv_row,
        "us_text": [us_text(r) for r in df.itertuples(index=False)],
        "india_text": [india_text(r) for r in df.itertuples(index=False)],
        "us_has_defect": [bool(_clean(d)) for d in df.DESC_DEFECT],
        "india_has_defect": [bool(_clean(d)) for d in df.defect_description],
    })


def build_prompt(batch: list[dict]) -> str:
    """One call's prompt: the rules, then up to BATCH pairs with both texts."""
    head = (
        "For each pair decide whether the US recall and the INDIA recall describe the "
        "SAME defect on the SAME model family. A different model-year range is fine. A "
        "different component, a different failure, or a different model family means "
        "\"different\".\n"
        "The SAME defect needs the same failing component in the same position: a driver "
        "airbag inflator and a passenger airbag inflator are DIFFERENT components even if "
        "both may rupture; front and rear seat belts are different too. A shared symptom "
        "alone (for example 'may catch fire') is not enough.\n\n"
        "Return ONLY JSON: {\"results\": [{\"pair_id\": \"...\", \"label\": \"same\" or "
        "\"different\", \"confidence\": number from 0 to 1, \"us_evidence\": \"...\", "
        "\"india_evidence\": \"...\"}]} with exactly one result per pair.\n"
        "us_evidence: an exact phrase of about 5 to 25 words copied character for character "
        "from that pair's US text, stating the US defect. india_evidence: the same from that "
        "pair's INDIA text. Copy - do not paraphrase, shorten inside the phrase, or fix "
        "spelling. Give evidence for \"different\" pairs too.\n"
        "us_evidence must come from the lines under that pair's \"US:\" heading and "
        "india_evidence from the lines under its \"INDIA:\" heading. The two sides usually "
        "word the same defect differently, so never reuse one side's phrase as the other "
        "side's evidence - before answering, check each quote occurs in its own block.\n"
    )
    blocks = [f"=== PAIR {p['pair_id']} (make {p['make']}) ===\nUS:\n{p['us_text']}\n"
              f"INDIA:\n{p['india_text']}" for p in batch]
    return head + "\n" + "\n\n".join(blocks)


def normalise(text: object) -> str:
    """Case-folded, whitespace-collapsed, typographic quotes/dashes made plain."""
    return " ".join(_clean(text).translate(_PLAIN).casefold().split())


def evidence_found(phrase: object, source: object) -> bool:
    """True only if the normalised phrase is a non-empty substring of the normalised source."""
    p = normalise(phrase)
    return bool(p) and p in normalise(source)


def check_result(result: dict | None, pair: dict) -> dict:
    """Validate one LLM result against its pair; anything unsupported -> needs_review."""
    out = {"pair_id": pair["pair_id"], "CAMPNO": pair["CAMPNO"], "make": pair["make"],
           "csv_row": pair["csv_row"], "llm_label": None, "confidence": None,
           "us_evidence": None, "india_evidence": None,
           "us_evidence_found": False, "india_evidence_found": False}
    reasons = []
    if not isinstance(result, dict):
        reasons.append("LLM returned no result for this pair")
    else:
        label = _clean(result.get("label")).lower()
        out["llm_label"] = label or None
        if label not in LABELS:
            reasons.append(f"invalid label {label!r}")

        conf = result.get("confidence")
        try:
            conf_f = float(conf)
            ok = not isinstance(conf, bool) and 0.0 <= conf_f <= 1.0
        except (TypeError, ValueError):
            conf_f, ok = None, False
        out["confidence"] = conf_f if ok else None
        if not ok:
            reasons.append(f"invalid confidence {conf!r}")

        for side, text_key in (("us", "us_text"), ("india", "india_text")):
            phrase = _clean(result.get(f"{side}_evidence"))
            out[f"{side}_evidence"] = phrase or None
            found = evidence_found(phrase, pair[text_key])
            out[f"{side}_evidence_found"] = found
            if not found:
                reasons.append(f"{side}_evidence not found in {side.upper()} text")
            elif len(normalise(phrase).split()) < MIN_EVIDENCE_WORDS:
                reasons.append(f"{side}_evidence shorter than {MIN_EVIDENCE_WORDS} words")

    out["label"] = NEEDS_REVIEW if reasons else out["llm_label"]
    out["review_reason"] = "; ".join(reasons) or None
    return out


def _results_by_id(answer: object) -> dict[str, dict]:
    results = answer.get("results") if isinstance(answer, dict) else answer
    by_id: dict[str, dict] = {}
    for r in results if isinstance(results, list) else []:
        if isinstance(r, dict) and r.get("pair_id") is not None:
            by_id.setdefault(str(r["pair_id"]).strip(), r)
    return by_id


def judge(pairs: pd.DataFrame, ask: Callable[..., object] = complete_json,
          batch_size: int = BATCH) -> pd.DataFrame:
    """Ask the LLM in batches, then check every answer in Python."""
    rows = []
    records = pairs.to_dict("records")
    sendable = [p for p in records if p["us_has_defect"] and p["india_has_defect"]]
    for p in records:
        if p not in sendable:
            side = "US" if not p["us_has_defect"] else "INDIA"
            row = check_result(None, p)
            row["review_reason"] = f"no {side} defect text; not sent to the LLM"
            rows.append(row)
    for batch in batched(sendable, batch_size):
        answer = ask(build_prompt(batch), system=SYSTEM, temperature=0.0, max_tokens=4096)
        by_id = _results_by_id(answer)
        unknown = set(by_id) - {p["pair_id"] for p in batch}
        if unknown:
            log.warning("LLM returned results for pairs not in this batch: %s", sorted(unknown))
        rows += [check_result(by_id.get(p["pair_id"]), p) for p in batch]
    out = pd.DataFrame(rows)
    out["llm_model"] = llm.MODEL
    return out.sort_values(["CAMPNO", "csv_row"]).reset_index(drop=True)


def _rate(part: int, n: int) -> str:
    return f"{100 * part / n:.1f}% (n={n})" if n >= 10 else f"n={n}<10, rate suppressed"


def compare_ground_truth(matches: pd.DataFrame, gt: pd.DataFrame) -> dict[str, object]:
    """Confusion table of ground-truth label vs final label, and agreement with its n."""
    m = gt[["pair_id", "label"]].rename(columns={"label": "truth"}).merge(
        matches[["pair_id", "label"]], on="pair_id", how="left")
    m["label"] = m["label"].fillna("not judged")
    table = pd.crosstab(m.truth, m.label)
    judged = m[m.label.isin(LABELS)]
    agree = int((judged.truth == judged.label).sum())
    per_class = {t: (int((g.truth == g.label).sum()), len(g))
                 for t, g in judged.groupby("truth")}
    return {"table": table, "judged": len(judged), "agree": agree,
            "needs_review": int((m.label == NEEDS_REVIEW).sum()), "per_class": per_class}


def report(matches: pd.DataFrame, comparison: dict | None) -> str:
    lines = ["Final labels: " + ", ".join(f"{k}={v}" for k, v in
                                          matches.label.value_counts().items())]
    review = matches[matches.label == NEEDS_REVIEW]
    if len(review):
        lines.append("\nneeds_review:")
        lines += [f"  {r.pair_id}: {r.review_reason}" for r in review.itertuples()]
    if comparison:
        lines.append("\nAgainst data/reference/ground_truth.csv (rows = ground truth):")
        lines.append(comparison["table"].to_string())
        lines.append(f"Agreement on judged pairs: {comparison['agree']} of "
                     f"{comparison['judged']} = {_rate(comparison['agree'], comparison['judged'])}"
                     f"; needs_review: {comparison['needs_review']}")
        for truth, (ok, n) in comparison["per_class"].items():
            lines.append(f"  ground truth '{truth}': {ok} of {n} agreed, {_rate(ok, n)}")
    return "\n".join(lines)


def build(db_path: Path = DB_PATH, gt_csv: Path = GROUND_TRUTH_CSV,
          scope: str = "ground_truth", ask: Callable[..., object] = complete_json
          ) -> dict[str, object]:
    """Judge the pairs in scope, write `matches`, compare with ground truth if present."""
    gt = pd.read_csv(gt_csv, dtype=str) if gt_csv.exists() else None
    con = duckdb.connect(str(db_path))
    try:
        if scope == "ground_truth":
            if gt is None:
                raise FileNotFoundError(f"{gt_csv} not found")
            keys = gt.pair_id.str.rsplit("-", n=1, expand=True)
            keys.columns = ["CAMPNO", "csv_row"]
            keys["csv_row"] = keys.csv_row.astype(int)
        elif scope == "all":
            keys = con.execute("SELECT CAMPNO, csv_row FROM candidate_pairs").df()
        else:
            raise ValueError(f"unknown scope {scope!r}")
        matches = judge(load_pairs(con, keys), ask).assign(scope=scope)
        con.register("matches_df", matches)
        con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM matches_df")
        con.unregister("matches_df")
    finally:
        con.close()
    comparison = compare_ground_truth(matches, gt) if gt is not None else None
    return {"matches": matches, "comparison": comparison}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = sys.argv[1:] if argv is None else argv
    out = build(scope="all" if "--all" in args else "ground_truth")
    print(report(out["matches"], out["comparison"]))


if __name__ == "__main__":
    main()
