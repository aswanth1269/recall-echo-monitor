"""Find SIAM rows that could be the same recall as an India-relevant NHTSA campaign.

1. india_relevant_campaigns: NHTSA campaigns with at least one model whose
   family is one of that make's India families (the full vocabulary in
   data/reference/model_families.csv: India lineup + families SIAM names).
   Component, defect, consequence and remedy are the first non-null value
   per campaign (lowest RECORD_ID). llm_only flags campaigns that are
   relevant only through unconfirmed LLM family mappings.
2. candidate_window: SIAM rows with the same make, a shared family, and a
   recall_date from 180 days before to 730 days after the NHTSA report_date.
   SIAM rows with a NULL recall_date are skipped and counted in the log.
3. Each pair is scored by TF-IDF cosine similarity between the US component
   + defect + consequence text and the SIAM defect description. The top 3 per
   campaign are kept as candidate_pairs (score, rank).

A score is a text-similarity hint for a person to check, not a match.

Run:  .venv/Scripts/python.exe -m src.candidates
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.load_nhtsa import DB_PATH, run_sql
from src.model_family import VOCAB_CSV

TOP_K = 3
WINDOW_BEFORE_DAYS, WINDOW_AFTER_DAYS = 180, 730  # documented in sql/candidate_window.sql
EXAMPLES_PER_BAND = 5

log = logging.getLogger(__name__)


def load_india_families(vocab_csv: Path = VOCAB_CSV) -> pd.DataFrame:
    """(make, family) pairs that count as India families."""
    df = pd.read_csv(vocab_csv, dtype=str, keep_default_na=False)
    return df[["make", "family"]].drop_duplicates().reset_index(drop=True)


def build_relevant(con: duckdb.DuckDBPyConnection, india: pd.DataFrame) -> int:
    """Create india_relevant_campaigns; return its row count."""
    con.register("india_families", india)
    try:
        run_sql(con, "india_relevant_campaigns")
    finally:
        con.unregister("india_families")
    return con.execute("SELECT count(*) FROM india_relevant_campaigns").fetchone()[0]


def window_pairs(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, int]:
    """Create temp candidate_window; return its keys and the NULL-date skip count."""
    n_null = con.execute(
        "SELECT count(*) FROM siam_recalls WHERE recall_date IS NULL "
        "AND make IN (SELECT DISTINCT make FROM india_relevant_campaigns)").fetchone()[0]
    log.info("skipped %d SIAM rows with a NULL recall_date", n_null)
    run_sql(con, "candidate_window")
    keys = con.execute("SELECT CAMPNO, csv_row, days_after_report FROM candidate_window "
                       "ORDER BY CAMPNO, csv_row").df()
    return keys, n_null


def _text(*parts: object) -> str:
    out = []
    for p in parts:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        out.extend(map(str, p) if isinstance(p, (list, tuple, np.ndarray)) else [str(p)])
    return " ".join(out).strip()


def score_pairs(pairs: pd.DataFrame, us_text: dict[str, str],
                siam_text: dict[int, str]) -> pd.DataFrame:
    """Add score = TF-IDF cosine similarity of the campaign text and the SIAM text.

    The vectorizer is fitted once on every US and SIAM text, so a pair's score
    does not depend on which other pairs exist. Empty text scores 0.
    """
    if pairs.empty:
        return pairs.assign(score=pd.Series(dtype=float))
    corpus = [t for t in [*us_text.values(), *siam_text.values()] if t]
    vec = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True,
                          ngram_range=(1, 2)).fit(corpus)
    us_docs = [us_text.get(c, "") for c in pairs.CAMPNO]
    in_docs = [siam_text.get(int(r), "") for r in pairs.csv_row]
    n_empty = sum(1 for a, b in zip(us_docs, in_docs) if not a or not b)
    if n_empty:
        log.info("%d pairs have empty US or SIAM text and score 0", n_empty)
    scores = np.asarray(vec.transform(us_docs).multiply(vec.transform(in_docs)).sum(axis=1)).ravel()
    return pairs.assign(score=np.round(scores, 4))


def top_k(scored: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Rank per campaign: score desc, then closer in time, then lower csv_row."""
    if scored.empty:
        return scored.assign(rank=pd.Series(dtype=int))
    ordered = scored.assign(_gap=scored.days_after_report.abs()).sort_values(
        ["CAMPNO", "score", "_gap", "csv_row"], ascending=[True, False, True, True])
    ordered["rank"] = ordered.groupby("CAMPNO").cumcount() + 1
    return ordered[ordered["rank"] <= k].drop(columns="_gap").reset_index(drop=True)


def build(db_path: Path = DB_PATH, vocab_csv: Path = VOCAB_CSV) -> dict[str, object]:
    """Write india_relevant_campaigns and candidate_pairs; return report frames."""
    india = load_india_families(vocab_csv)
    con = duckdb.connect(str(db_path))
    try:
        build_relevant(con, india)
        keys, n_null = window_pairs(con)

        rel = con.execute("SELECT CAMPNO, components, DESC_DEFECT, CONEQUENCE_DEFECT "
                          "FROM india_relevant_campaigns").df()
        us_text = {r.CAMPNO: _text(r.components, r.DESC_DEFECT, r.CONEQUENCE_DEFECT)
                   for r in rel.itertuples(index=False)}
        siam_text = {int(r): _text(t) for r, t in con.execute(
            "SELECT csv_row, defect_description FROM siam_recalls").fetchall()}
        kept = top_k(score_pairs(keys, us_text, siam_text))

        con.register("kept_scores", kept[["CAMPNO", "csv_row", "score", "rank"]]
                     if len(kept) else pd.DataFrame({"CAMPNO": pd.Series(dtype=str),
                                                     "csv_row": pd.Series(dtype=int),
                                                     "score": pd.Series(dtype=float),
                                                     "rank": pd.Series(dtype=int)}))
        run_sql(con, "candidate_pairs")
        con.unregister("kept_scores")

        window_counts = keys.groupby("CAMPNO").size().rename("n").reset_index() \
            if len(keys) else pd.DataFrame({"CAMPNO": pd.Series(dtype=str), "n": pd.Series(dtype=int)})
        con.register("window_counts", window_counts)
        con.execute("""CREATE TEMP TABLE rel_tmp AS
            SELECT c.*, CAST(coalesce(w.n, 0) AS INTEGER) AS n_window_candidates
            FROM india_relevant_campaigns AS c LEFT JOIN window_counts AS w USING (CAMPNO)""")
        con.execute("CREATE OR REPLACE TABLE india_relevant_campaigns AS SELECT * FROM rel_tmp")
        con.execute("DROP TABLE rel_tmp")
        con.unregister("window_counts")
        con.execute("DROP TABLE IF EXISTS candidate_window")

        return {
            "null_date_skipped": n_null,
            "relevant": con.execute("""
                WITH totals AS (SELECT m AS make, count(*) AS all_campaigns
                                FROM (SELECT unnest(makes) AS m FROM nhtsa_campaigns)
                                GROUP BY m)
                SELECT c.make, count(*) AS relevant_campaigns,
                       count(*) FILTER (WHERE c.llm_only) AS llm_only, t.all_campaigns
                FROM india_relevant_campaigns AS c JOIN totals AS t USING (make)
                GROUP BY c.make, t.all_campaigns ORDER BY c.make""").df(),
            "zero": con.execute("""
                SELECT make, count(*) AS zero_candidate_campaigns,
                       list(CAMPNO ORDER BY CAMPNO)[1:10] AS examples
                FROM india_relevant_campaigns WHERE n_window_candidates = 0
                GROUP BY make ORDER BY make""").df(),
            "by_rank": con.execute("SELECT rank, count(*) AS pairs FROM candidate_pairs "
                                   "GROUP BY rank ORDER BY rank").df(),
            "pairs": con.execute("""
                SELECT p.CAMPNO, p.csv_row, p.make, p.shared_families, p.score, p.rank,
                       p.days_after_report, p.llm_only,
                       concat_ws(' | ', c.COMPNAME, c.DESC_DEFECT) AS us_text,
                       s.defect_description AS siam_text
                FROM candidate_pairs AS p
                JOIN india_relevant_campaigns AS c USING (CAMPNO)
                JOIN siam_recalls AS s USING (csv_row)
                ORDER BY p.score DESC, p.CAMPNO, p.csv_row""").df(),
        }
    finally:
        con.close()


def _pct(part: int, n: int) -> str:
    return f"{100 * part / n:.1f}%" if n >= 10 else "n<10"


def report(summary: dict[str, object]) -> str:
    """Printable summary: relevance, zero-candidate campaigns, pair totals, examples."""
    lines = ["India-relevant campaigns per make (rates suppressed where n < 10):"]
    for r in summary["relevant"].itertuples(index=False):
        lines.append(f"  {r.make:<14} {r.relevant_campaigns:>4} of {r.all_campaigns:>4} campaigns "
                     f"({_pct(r.relevant_campaigns, r.all_campaigns)})"
                     f"  llm_only={r.llm_only}")
    total_rel = int(summary["relevant"].relevant_campaigns.sum())
    lines.append(f"  {'total':<14} {total_rel:>4}")

    zero = summary["zero"]
    n_zero = int(zero.zero_candidate_campaigns.sum()) if len(zero) else 0
    lines.append(f"\nCampaigns with zero candidates: {n_zero} of {total_rel} "
                 f"({_pct(n_zero, total_rel)})")
    for r in zero.itertuples(index=False):
        lines.append(f"  {r.make:<14} {r.zero_candidate_campaigns:>4}  e.g. "
                     f"{', '.join(list(r.examples)[:10])}")

    pairs = summary["pairs"]
    lines.append(f"\nTotal candidate pairs: {len(pairs)}  "
                 + "  ".join(f"rank {r.rank}: {r.pairs}" for r in summary["by_rank"].itertuples()))
    lines.append(f"SIAM rows skipped for a NULL recall_date: {summary['null_date_skipped']}")

    if len(pairs):
        mid = len(pairs) // 2
        bands = {"HIGH": pairs.head(EXAMPLES_PER_BAND),
                 "MIDDLE": pairs.iloc[max(mid - 2, 0): mid + 3],
                 "LOW": pairs.tail(EXAMPLES_PER_BAND)}
        for name, band in bands.items():
            lines.append(f"\n{name} scores:")
            for r in band.itertuples(index=False):
                lines.append(f"  {r.score:.3f}  {r.CAMPNO} <-> SIAM row {r.csv_row}  {r.make} "
                             f"{'/'.join(r.shared_families)}  ({r.days_after_report:+d} days"
                             f"{', llm_only' if r.llm_only else ''})")
                lines.append(f"         US:   {(_text(r.us_text) or '(no text)')[:90]}")
                lines.append(f"         SIAM: {(_text(r.siam_text) or '(no SIAM defect text)')[:90]}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(report(build()))


if __name__ == "__main__":
    main()
