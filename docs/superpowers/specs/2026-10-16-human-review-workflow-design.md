# Human review workflow — design

Date: 2026-10-16. Status: approved in brainstorming, awaiting spec review.

## Goal

No number in the KPIs rests on an unchecked LLM call. A person confirms every
LLM "same" verdict and every flagged item; the LLM's own answer is kept beside
the human decision so the matcher's precision can be measured.

Decisions taken in brainstorming:

| Question | Decision |
|---|---|
| Review bar | Every echo candidate (LLM `same`) plus every flagged item |
| Surface | Both: dashboard Review tab and hand-editable CSV, same rules |
| Text changes after review | Keep the decision, mark it stale, re-queue with the old decision shown |
| Evaluation | Reviews give matcher precision; the 30-pair ground truth stays separate for recall |
| Approach | A: decision files in `data/reference/` applied as a layer after the matcher |

## 1. Data model

All decision files live in `data/reference/` (inputs, hand-editable, committed).
Nothing in `data/raw` is touched.

### 1.1 Pair decisions — `pair_reviews.csv` (new)

| column | meaning |
|---|---|
| `pair_id` | as in `matches`, e.g. `16V899000-228` |
| `decision` | `same` or `different`; nothing else is accepted |
| `note` | optional free text, stored formula-safe |
| `fingerprint` | first 16 hex chars of SHA-256 over `us_text + "\x1f" + india_text`, built by `matcher.us_text` / `matcher.india_text` |
| `reviewed_at` | ISO date `YYYY-MM-DD` |

One row per `pair_id`; a new decision replaces the old row.

### 1.2 Family decisions — `model_family_overrides.csv` (existing seam)

`model_family.py` already reads this file and applies it last
(`method = 'override'`). It does not exist yet. Two columns are added to
`OVERRIDE_COLUMNS`: `fingerprint` (hash of `side|make|key|text`) and
`reviewed_at`. `load_overrides` accepts files without them (blank = never
stale). Confirming the LLM was right is an override with the same family.
Families must be in the vocabulary or `none` (existing check).

### 1.3 Vocabulary decisions — `vocabulary_reviews.csv` (new) and `model_aliases.csv` (existing)

The 18 rows of `data/clean/vocabulary_review.csv` ask whether two names are the
same model (e.g. HYUNDAI XCENT vs XCIENT). Two outcomes:

- `distinct`: recorded in `vocabulary_reviews.csv` (`make, family, decision, note, reviewed_at`); no pipeline effect, it only leaves the queue.
- `alias`: written as a row of the existing `model_aliases.csv`, which already requires `source_url` and `retrieved_on`. No source, no alias — `record()` refuses it. This keeps the "never guess" rule.

### 1.4 Applying decisions — table `matches_reviewed` (new)

All `matches` columns, plus:

| column | meaning |
|---|---|
| `fingerprint` | current fingerprint of the pair's texts |
| `human_decision` | the decision from `pair_reviews.csv`, or NULL |
| `review_status` | `unreviewed`, `reviewed` or `stale` |
| `final_label` | `human_decision` when `review_status = 'reviewed'`, else NULL |

The LLM's `label` and `llm_label` are never changed.

- **Stale**: decision exists but its fingerprint differs from the current one. Kept in the CSV, not applied, re-queued with the old decision shown.
- **Orphaned**: decision whose `pair_id` is no longer in `matches` (e.g. a family change removed the candidate pair). Kept, not applied, logged with a count.

## 2. Queue and `src/review.py`

### 2.1 Queue (default: 339 items, in this order)

| order | kind | count today | reason |
|---|---|---|---|
| 1 | LLM family mappings (`nhtsa_model_family.method = 'llm'` with a family) | 42 (2,254 records) | an LLM chose the family |
| 2 | SIAM rows with zero families | 26 | they can never match |
| 3 | vocabulary flags | 18 | possible same-model spellings |
| 4 | `needs_review` pairs | 15 | evidence check failed or instruction-like text |
| 5 | LLM `same` pairs | 238 | every echo is confirmed |

Family items come first because they change which candidate pairs exist.
Not queued by default but reachable through a filter: the 359 LLM "none"
family answers. Not queued at all: the 1,367 `different` pairs (out of scope).

### 2.2 Interface

The dashboard and the CLI call only these:

- `queue(con) -> DataFrame` — open, stale and orphaned items with `kind`, `key`, both texts, the LLM answer and evidence, and the previous decision for stale items.
- `record(kind, key, decision, note="", source_url=None, retrieved_on=None)` — validates (key exists, decision allowed for the kind, family in vocabulary, alias has a source), makes text formula-safe with `csv_safe.excel_safe_value`, stamps the current fingerprint and date, and writes one row to the file for that kind, replacing any earlier row for the key. Writes are atomic (temp file, then rename) and re-read the file first so concurrent hand edits survive. Raises `ValueError` with a plain message on anything invalid; writes nothing then.
- `apply(db_path) -> dict` — builds `matches_reviewed`; returns counts `reviewed`, `stale`, `orphaned`, `unreviewed`; logs stale and orphaned keys.
- `precision(con) -> DataFrame` — LLM `same` vs human decision over reviewed pairs: `n_reviewed`, `n_confirmed`, `precision`; `precision` is NULL when `n_reviewed < 10`.

### 2.3 CLI

`.venv/Scripts/python.exe -m src.review`

- default: run `apply`, print queue counts by kind and status, write `data/clean/review_queue.csv` (formula-safe, blank `decision`/`note` columns).
- `--import data/clean/review_queue.csv`: every row with a filled `decision` goes through `record()`; rejected rows are listed with the reason and the rest are kept.
- `--rebuild`: run `model_family`, `candidates`, `matcher --all`, `review`, `kpis` in order, for use after family or vocabulary decisions.

### 2.4 Matcher reuse (change to `src/matcher.py`)

The LLM cache key is the whole batch prompt (`src/llm.py:76`), so adding one
candidate pair shifts every later batch and defeats the cache. The matcher
therefore stores `fingerprint` and `prompt_version` (hash of the system prompt
and instruction text) in `matches`, and on a rerun reuses the existing row for
any pair whose `fingerprint`, `prompt_version` and `llm_model` are unchanged.
Only new or changed pairs are sent to the LLM. The run logs how many were
reused and how many sent.

## 3. Dashboard Review tab (`app/dashboard.py`)

- Top: tiles for families reviewed, pairs reviewed, stale items (each "x of n"), and matcher precision once n ≥ 10.
- Filters: kind, make, status (open, stale, orphaned, LLM "none").
- One item at a time:
  - pair: US and India text side by side, evidence highlighted with `data_sources.find_evidence`, LLM label and confidence; buttons Same / Different.
  - family: model text, LLM choice, dropdown limited to vocabulary families plus `none`; buttons Confirm / Change.
  - vocabulary: the two names; Distinct, or Alias with required source URL and date.
  - stale items also show the old decision.
  - Note box; Skip; Undo last (restores the previous row of that file).
- Buttons call `review.record()`; a `ValueError` is shown and nothing is written.
- After a family or vocabulary decision, a banner says to run `python -m src.review --rebuild`. The dashboard never runs the pipeline.
- When any decision file is newer than `matches_reviewed`, a banner says the KPI tabs are behind until `src.review` is rerun.
- A locked database or missing file shows a message, as the other tabs do.
- Still localhost-only with XSRF on.

## 4. KPIs, evaluation, tests

### 4.1 KPIs

`echo_rate`, `echo_lag`, `echo_by_component`, `ota` and `echo_explorer` read
`matches_reviewed`. An echo counts only when `final_label = 'same'`. Each KPI
row gains `n_reviewed`; n < 10 suppression is unchanged. Before any review the
echo count is 0 by design; the dashboard states "based on x of 238 reviewed
echo candidates". `echo_explorer` gains `review_status`.

### 4.2 Evaluation (`src/evaluate.py`)

- New section "Matcher precision (human-reviewed)" from `review.precision`, labelled as precision only (says nothing about missed echoes). Rows added to `reports/eval_summary.csv`.
- `ground_truth.csv` gains a blank `human_label` column. Evaluation uses `human_label` where filled, else `label`, and keeps the "not human-verified" warning until all 30 have a `human_label`. The existing labels are never overwritten.

### 4.3 Tests

New `tests/test_review.py`, plus updates to matcher, KPI, evaluate and
dashboard tests. The LLM is always faked.

- fingerprint: same texts → same hash; one changed character → different hash.
- `record`: rejects unknown key, bad decision, family outside vocabulary, alias without source; makes notes formula-safe; replaces an earlier decision; atomic write; can only write `pair_reviews.csv`, `model_family_overrides.csv`, `vocabulary_reviews.csv`, `model_aliases.csv`.
- `apply`: reviewed, stale, orphaned, unreviewed each get the right `final_label`; `label` unchanged.
- `queue`: order and counts; stale items carry the old decision.
- matcher reuse: unchanged pair → no LLM call; changed text or `prompt_version` → re-sent.
- `model_family.load_overrides`: accepts files with and without the new columns.
- KPIs on a fake `matches_reviewed`: unreviewed `same` pairs do not count; `n_reviewed` correct.
- `--import` round-trip, including a rejected row.
- `precision`: NULL below n = 10.
- dashboard Review tab renders on fake data; a locked database does not crash it.

`pytest -q` stays green.

## Out of scope

- Multiple reviewers or reviewer identity.
- Reviewing the 1,367 `different` pairs (a recall sample is a separate decision).
- Running the pipeline from the dashboard.
- Reviewing the 359 LLM "none" family answers by default (reachable by filter).
