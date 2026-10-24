"""Evaluate the LLM matcher and the TF-IDF baseline against the ground-truth labels.

Positive class = "same". Two matchers are compared on the pairs in
data/reference/ground_truth.csv:

  * LLM matcher: the final `label` in table `matches` (run src.matcher first);
    "needs_review" counts as "different".
  * TF-IDF baseline: the pair's `score` in candidate_pairs, predicting "same"
    when score >= threshold. The threshold is chosen to maximise F1 on these
    SAME 30 pairs, which flatters the baseline - it is tuned on its own test
    set, the LLM matcher is not.

For each: confusion matrix, precision, recall, F1. Following CLAUDE.md, every
rate shows its n and the percentage is suppressed where n < 10 (the fraction
is still shown). Every pair where either matcher disagrees with the ground
truth is listed with both texts. Output: reports/eval.md.

Also reports the matcher's precision from the human review of LLM `same`
pairs (table matches_reviewed). A person's blind labels in
data/reference/human_labels.csv (pair_id, human_label, note; made by
`python -m src.make_label_sheet --human-labels`) are the truth wherever
human_label is filled; other pairs keep Claude's label from
ground_truth.csv. The report says who labelled what.

The ground-truth and human-label files are only ever read.

Run:  .venv/Scripts/python.exe -m src.evaluate
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pandas as pd

from src.load_nhtsa import DB_PATH
from src.matcher import NEEDS_REVIEW, load_pairs

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_CSV = ROOT / "data/reference/ground_truth.csv"
REPORT_MD = ROOT / "reports/eval.md"
POSITIVE = "same"
MIN_N = 10


HUMAN_LABELS_NAME = "human_labels.csv"   # lives next to ground_truth.csv


def human_labels_path(gt_path: Path) -> Path:
    return gt_path.parent / HUMAN_LABELS_NAME


def load_human_labels(path: Path, pair_ids: set[str]) -> pd.DataFrame:
    """pair_id, human_label, note for filled rows only; a missing file means none yet.

    Refuses unknown pair_ids, values other than same/different/blank, and duplicates.
    """
    cols = ["pair_id", "human_label", "note"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    hl = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(cols[:2]) - set(hl.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    if "note" not in hl:
        hl["note"] = ""
    hl = hl[cols].apply(lambda c: c.str.strip())
    hl["human_label"] = hl.human_label.str.lower()
    bad = hl[~hl.human_label.isin(["", "same", "different"])]
    if len(bad):
        raise ValueError(f"{path.name}: human_label must be same, different or blank; "
                         f"got {dict(zip(bad.pair_id, bad.human_label))}")
    unknown = sorted(set(hl.pair_id) - pair_ids)
    if unknown:
        raise ValueError(f"{path.name}: pair_ids not in the ground truth: {unknown}")
    dup = sorted(hl.pair_id[hl.pair_id.duplicated()].unique())
    if dup:
        raise ValueError(f"{path.name}: pair_ids listed twice: {dup}")
    return hl[hl.human_label != ""].reset_index(drop=True)


def load_ground_truth(path: Path = GROUND_TRUTH_CSV) -> pd.DataFrame:
    """Read the labels (read-only); a person's label in human_labels.csv wins where filled.

    Adds label_source ('human' | 'claude') and CAMPNO / csv_row parsed from pair_id.
    For a human-labelled pair, reason is the person's note and confidence is 'human'.
    """
    gt = pd.read_csv(path, dtype=str, keep_default_na=False)
    bad = sorted(set(gt.label) - {"same", "different"})
    if bad:
        raise ValueError(f"ground truth has labels other than same/different: {bad}")
    hl = load_human_labels(human_labels_path(path), set(gt.pair_id)).set_index("pair_id")
    human = gt.pair_id.isin(hl.index)
    gt = gt.assign(label_source=human.map({True: "human", False: "claude"}))
    if human.any():
        ids = gt.loc[human, "pair_id"]
        gt.loc[human, "label"] = ids.map(hl.human_label).values
        gt.loc[human, "reason"] = ids.map(hl.note).values
        gt.loc[human, "confidence"] = "human"
    keys = gt.pair_id.str.rsplit("-", n=1, expand=True)
    return gt.assign(CAMPNO=keys[0], csv_row=keys[1].astype(int))


def human_verified(gt: pd.DataFrame) -> bool:
    """True only when a person labelled every ground-truth pair."""
    return bool(len(gt)) and bool((gt.label_source == "human").all())


def labelled_by_text(gt: pd.DataFrame) -> str:
    """Who labelled the truth, e.g. 'a person for 12 of 30 pairs (...); Claude ... for the other 18'."""
    n, k = len(gt), int((gt.label_source == "human").sum())
    claude = "; ".join(sorted(set(gt.loc[gt.label_source == "claude", "labelled_by"])
                              if "labelled_by" in gt else {"Claude"})) or "Claude"
    if k == n:
        return f"a person for all {n} pairs ({HUMAN_LABELS_NAME})"
    if k == 0:
        return f"{claude} - all {n} pairs; no human labels yet in {HUMAN_LABELS_NAME}"
    return (f"a person for {k} of {n} pairs ({HUMAN_LABELS_NAME}); "
            f"{claude} for the other {n - k}")


def reviewed_precision(con: duckdb.DuckDBPyConnection,
                       ref_dir: Path | None = None) -> dict | None:
    """Matcher precision from the human review of LLM `same` pairs, or None before review.

    With ref_dir, "sample" holds the same figure for the saved random sample
    (review_sample.csv there), or None while no sample has been saved.
    """
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    if "matches_reviewed" not in tables:
        return None
    from src.review import precision, sample_precision   # here: review imports matcher
    row = precision(con).iloc[0]
    return {"n_reviewed": int(row.n_reviewed), "n_confirmed": int(row.n_confirmed),
            "precision": None if pd.isna(row.precision) else float(row.precision),
            "sample": sample_precision(con, ref_dir) if ref_dir is not None else None}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def llm_predictions(con: duckdb.DuckDBPyConnection, gt: pd.DataFrame) -> pd.DataFrame:
    """pair_id, llm_raw (as stored), llm_pred (needs_review -> different), confidence, evidence."""
    m = con.execute("""SELECT pair_id, label AS llm_raw, confidence AS llm_confidence,
                              us_evidence, india_evidence, review_reason
                       FROM matches""").df()
    missing = sorted(set(gt.pair_id) - set(m.pair_id))
    if missing:
        raise ValueError(f"{len(missing)} ground-truth pairs have no LLM result in `matches` "
                         f"(run python -m src.matcher first): {missing[:5]}")
    m["llm_pred"] = m.llm_raw.where(m.llm_raw == POSITIVE, "different")
    return m[m.pair_id.isin(gt.pair_id)]


def baseline_scores(con: duckdb.DuckDBPyConnection, gt: pd.DataFrame) -> pd.DataFrame:
    """pair_id, tfidf score from candidate_pairs."""
    con.register("gt_keys", gt[["pair_id", "CAMPNO", "csv_row"]])
    try:
        s = con.execute("""SELECT g.pair_id, p.score FROM gt_keys AS g
                           JOIN candidate_pairs AS p
                             ON p.CAMPNO = g.CAMPNO AND p.csv_row = g.csv_row""").df()
    finally:
        con.unregister("gt_keys")
    missing = sorted(set(gt.pair_id) - set(s.pair_id))
    if missing:
        raise ValueError(f"ground-truth pairs missing from candidate_pairs: {missing[:5]}")
    return s


def confusion(truth: pd.Series, pred: pd.Series) -> dict[str, int]:
    t, p = truth.eq(POSITIVE).to_numpy(), pred.eq(POSITIVE).to_numpy()
    return {"tp": int((t & p).sum()), "fp": int((~t & p).sum()),
            "fn": int((t & ~p).sum()), "tn": int((~t & ~p).sum())}


def metrics(c: dict[str, int]) -> dict[str, tuple[float, int, int]]:
    """Each metric as (value, numerator, denominator); value is NaN when denominator is 0."""
    def ratio(num, den):
        return (num / den if den else float("nan"), num, den)
    return {"precision": ratio(c["tp"], c["tp"] + c["fp"]),
            "recall": ratio(c["tp"], c["tp"] + c["fn"]),
            "f1": ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])}


def fmt_rate(value: float, num: int, den: int) -> str:
    """'3 / 4 (n=4 < 10, rate suppressed)' or '27 / 29 = 0.931 (n=29)'."""
    if den == 0:
        return "undefined (n=0)"
    if den < MIN_N:
        return f"{num} / {den} (n={den} < {MIN_N}, rate suppressed)"
    return f"{num} / {den} = {value:.3f} (n={den})"


def choose_threshold(scores: pd.Series, truth: pd.Series) -> tuple[float, float]:
    """Threshold maximising F1 on these pairs; ties go to the higher threshold."""
    best_t, best_f1 = float("inf"), -1.0
    for t in sorted(scores.unique()):
        f1 = metrics(confusion(truth, scores.ge(t).map({True: POSITIVE, False: "different"})))["f1"][0]
        f1 = 0.0 if f1 != f1 else f1          # NaN -> 0
        if f1 > best_f1 or (f1 == best_f1 and t > best_t):
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def evaluate(gt: pd.DataFrame, llm: pd.DataFrame, base: pd.DataFrame) -> dict[str, object]:
    df = gt.merge(llm, on="pair_id").merge(base, on="pair_id")
    threshold, _ = choose_threshold(df.score, df.label)
    df["base_pred"] = df.score.ge(threshold).map({True: POSITIVE, False: "different"})
    out = {"threshold": threshold, "pairs": df, "n": len(df)}
    for name, col in (("llm", "llm_pred"), ("baseline", "base_pred")):
        c = confusion(df.label, df[col])
        out[name] = {"confusion": c, "metrics": metrics(c)}
    return out


def _cm_table(c: dict[str, int]) -> list[str]:
    return ["| | predicted same | predicted different |", "|---|---:|---:|",
            f"| **truth same** | {c['tp']} | {c['fn']} |",
            f"| **truth different** | {c['fp']} | {c['tn']} |"]


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in str(text).splitlines()]


def _revision_lines(df: pd.DataFrame) -> list[str]:
    """One bullet per ground-truth label revised after seeing matcher output."""
    if "revision_note" not in df.columns:
        return []
    revised = df[df.revision_note.fillna("").str.strip() != ""]
    if revised.empty:
        return []
    return ["- **Labels revised after seeing matcher output** "
            f"({len(revised)} note(s) in the ground-truth file):",
            *[f"  - `{r.pair_id}` (now `{r.label}`): {r.revision_note}"
              for r in revised.itertuples(index=False)]]


def render_markdown(result: dict[str, object], texts: pd.DataFrame, labelled_by: str,
                    gt_path: Path, reviewed: dict | None = None) -> str:
    df = result["pairs"]
    n_pos = int(df.label.eq(POSITIVE).sum())
    try:
        shown_path = gt_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:                       # file outside the repo (e.g. a test copy)
        shown_path = gt_path.as_posix()
    lines = [
        "# Matcher evaluation", "",
        f"Ground truth: `{shown_path}` - {result['n']} pairs, "
        f"{n_pos} labelled `same`, {result['n'] - n_pos} `different`. Positive class: `same`.", "",
        f"**Who labelled it:** {labelled_by}. "
        + ("These are a person's blind labels." if result.get("all_human") else
           "Until a person has labelled every pair, the numbers below partly measure "
           "agreement with Claude's labels, not accuracy."), "",
        "## Read this first", "",
        f"- **The TF-IDF baseline is flattered.** Its threshold (score >= "
        f"**{result['threshold']:.4f}**) was chosen to maximise F1 on these same "
        f"{result['n']} pairs, so it is tuned on its own test set. A fair comparison needs "
        "a threshold picked on separate pairs.",
        "- **The LLM matcher is flattered too.** Its prompt was sharpened after reviewing its "
        "disagreements on these same pairs (2026-10-15, round 1: same component and position; "
        "each quote from its own side), so it has also been tuned on its own test set.",
        *_revision_lines(df[df.label_source == "claude"] if "label_source" in df else df),
        "- **`needs_review` counts as `different`** for the LLM matcher "
        f"({int(df.llm_raw.eq(NEEDS_REVIEW).sum())} pair(s) here).",
        f"- **Small sample.** With only {n_pos} positive pairs, recall has n={n_pos}. Per "
        f"CLAUDE.md every rate shows its n and the percentage is suppressed when n < {MIN_N}; "
        "the fraction is always given.", "",
        "## Results", "",
        "| metric | LLM matcher | TF-IDF baseline |", "|---|---|---|",
    ]
    for m in ("precision", "recall", "f1"):
        lines.append(f"| {m} | {fmt_rate(*result['llm']['metrics'][m])} | "
                     f"{fmt_rate(*result['baseline']['metrics'][m])} |")
    lines += ["", "F1 is shown as 2·TP / (2·TP + FP + FN).", ""]
    for name, title in (("llm", "LLM matcher"),
                        ("baseline", f"TF-IDF baseline (score >= {result['threshold']:.4f})")):
        lines += [f"### Confusion matrix - {title}", "", *_cm_table(result[name]["confusion"]), ""]

    lines += ["## Matcher precision (human-reviewed)", ""]
    if reviewed is None or reviewed["n_reviewed"] == 0:
        lines += ["No LLM `same` pairs have been reviewed yet (run `python -m src.review`).", ""]
    else:
        n, ok = reviewed["n_reviewed"], reviewed["n_confirmed"]
        lines += [f"A person confirmed {ok} of {n} reviewed LLM `same` pairs: precision "
                  f"{fmt_rate(ok / n, ok, n)}.",
                  "This is precision only. It says nothing about echoes the matcher missed; "
                  "recall comes from the ground-truth pairs above.", ""]
    sample = (reviewed or {}).get("sample")
    if sample is not None:
        n, ok = sample["n_reviewed"], sample["n_confirmed"]
        lines += [f"**Random sample** ({sample['n_sample']} LLM `same` pairs, fixed seed, "
                  "listed in `data/reference/review_sample.csv`): "
                  + (f"a person confirmed {ok} of {n}: precision {fmt_rate(ok / n, ok, n)}."
                     if n else "none reviewed yet.")
                  + f" Not yet reviewed: {sample['n_sample'] - n - sample['n_cannot_judge'] - sample['n_not_in_matches']};"
                  f" cannot judge: {sample['n_cannot_judge']};"
                  f" no longer a pair: {sample['n_not_in_matches']}.",
                  "This is the unbiased figure: the sample was drawn before review, so hard "
                  "pairs were not skipped.", ""]

    dis = df[(df.llm_pred != df.label) | (df.base_pred != df.label)].sort_values("pair_id")
    lines += ["## Disagreements", "",
              f"{len(dis)} pair(s) where at least one matcher differs from the ground truth.", ""]
    t = texts.set_index("pair_id")
    for r in dis.itertuples(index=False):
        who = [n for n, col in (("LLM", "llm_pred"), ("baseline", "base_pred"))
               if getattr(r, col) != r.label]
        lines += [
            f"### {r.pair_id} ({r.make if hasattr(r, 'make') else ''}) - wrong: {', '.join(who)}", "",
            (f"- **Ground truth:** `{r.label}` (labelled by a person)"
             + (f" - note: {r.reason}" if r.reason else "")
             if getattr(r, "label_source", "claude") == "human" else
             f"- **Ground truth:** `{r.label}` (labelled by Claude, {r.confidence} confidence) "
             f"- {r.reason}"),
            f"- **LLM:** `{r.llm_raw}`"
            + (f" (confidence {r.llm_confidence:.2f})" if pd.notna(r.llm_confidence) else "")
            + (f" - review reason: {r.review_reason}" if isinstance(r.review_reason, str) else ""),
            f"  - US evidence: \"{r.us_evidence}\"" if isinstance(r.us_evidence, str) else "  - US evidence: none",
            f"  - India evidence: \"{r.india_evidence}\"" if isinstance(r.india_evidence, str) else "  - India evidence: none",
            f"- **Baseline:** score {r.score:.4f} -> `{r.base_pred}`", "",
            "**US text**", "", *_quote(t.loc[r.pair_id, "us_text"]), "",
            "**India text**", "", *_quote(t.loc[r.pair_id, "india_text"]), "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def summary_frame(result: dict[str, object], reviewed: dict | None = None,
                  labelled_by: str = "Claude, not human-verified") -> pd.DataFrame:
    """One row per matcher: counts, and each rate with its own n.

    A rate is left empty where its n < MIN_N (CLAUDE.md), so a reader of the
    CSV cannot show a suppressed rate by accident. `n` is the number of pairs.
    """
    rows = []
    for name in ("llm", "baseline"):
        c, m = result[name]["confusion"], result[name]["metrics"]
        row = {"matcher": name, "n": result["n"], "positives": c["tp"] + c["fn"], **c}
        for k in ("precision", "recall", "f1"):
            value, _num, den = m[k]
            row[k] = round(value, 4) if den >= MIN_N else None
            row[f"{k}_n"] = den
        row["threshold"] = result["threshold"] if name == "baseline" else None
        row["note"] = ("prompt tuned on these pairs" if name == "llm" else
                       "threshold chosen on these same pairs (flatters the baseline)")
        row["labelled_by"] = labelled_by
        rows.append(row)
    if reviewed and reviewed["n_reviewed"]:
        n, ok = reviewed["n_reviewed"], reviewed["n_confirmed"]
        rows.append({"matcher": "llm_reviewed", "n": n, "positives": None, "tp": ok,
                     "fp": n - ok, "fn": None, "tn": None,
                     "precision": round(ok / n, 4) if n >= MIN_N else None, "precision_n": n,
                     "recall": None, "recall_n": None, "f1": None, "f1_n": None,
                     "threshold": None,
                     "note": "precision from human review of LLM same pairs; not recall",
                     "labelled_by": "a person (review queue)"})
    sample = (reviewed or {}).get("sample")
    if sample and sample["n_reviewed"]:
        n, ok = sample["n_reviewed"], sample["n_confirmed"]
        rows.append({"matcher": "llm_reviewed_sample", "n": n, "positives": None, "tp": ok,
                     "fp": n - ok, "fn": None, "tn": None,
                     "precision": round(ok / n, 4) if n >= MIN_N else None, "precision_n": n,
                     "recall": None, "recall_n": None, "f1": None, "f1_n": None,
                     "threshold": None,
                     "note": (f"precision on the fixed random sample of {sample['n_sample']} "
                              "LLM same pairs; not recall"),
                     "labelled_by": "a person (review queue)"})
    return pd.DataFrame(rows)


def build(db_path: Path = DB_PATH, gt_path: Path = GROUND_TRUTH_CSV,
          out_md: Path = REPORT_MD) -> dict[str, object]:
    """Compute both evaluations and write the markdown report. Never writes gt_path."""
    if out_md.resolve() == gt_path.resolve():
        raise ValueError("refusing to write the report over the ground-truth file")
    before = file_digest(gt_path)
    hl_path = human_labels_path(gt_path)
    human_before = file_digest(hl_path) if hl_path.exists() else None
    gt = load_ground_truth(gt_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        llm = llm_predictions(con, gt)
        base = baseline_scores(con, gt)
        texts = load_pairs(con, gt[["CAMPNO", "csv_row"]])
        reviewed = reviewed_precision(con, gt_path.parent)
    finally:
        con.close()
    result = evaluate(gt.drop(columns=["CAMPNO", "csv_row"]).merge(
        texts[["pair_id", "make"]], on="pair_id"), llm, base)
    result["all_human"] = human_verified(gt)
    labelled_by = labelled_by_text(gt)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(result, texts, labelled_by, gt_path, reviewed), encoding="utf-8")
    summary_frame(result, reviewed, labelled_by).to_csv(out_md.parent / "eval_summary.csv", index=False,
                                 encoding="utf-8")
    if file_digest(gt_path) != before:          # belt and braces: labels must be untouched
        raise RuntimeError("ground-truth file changed during evaluation")
    if human_before is not None and file_digest(hl_path) != human_before:
        raise RuntimeError("human_labels.csv changed during evaluation")
    return result


def main() -> None:
    r = build()
    print(f"wrote {REPORT_MD.relative_to(ROOT)} - {r['n']} pairs, baseline threshold "
          f"{r['threshold']:.4f} (tuned on these pairs: flatters the baseline)")
    for name in ("llm", "baseline"):
        c, m = r[name]["confusion"], r[name]["metrics"]
        print(f"  {name:<9} TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} | "
              + " | ".join(f"{k} {fmt_rate(*m[k])}" for k in ("precision", "recall", "f1")))


if __name__ == "__main__":
    main()
