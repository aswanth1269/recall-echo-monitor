# Human Review Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-executing-plans to implement this plan task-by-task (the user asked for inline work: no sub-agents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A person confirms every LLM "same" verdict and every flagged item; KPIs count an echo only when a person confirmed it, and the matcher's precision is measured from those confirmations.

**Architecture:** Decisions live in hand-editable CSVs in `data/reference/`. A new module `src/review.py` builds the review queue, records decisions (one validation path for the dashboard and the CSV import), and writes table `matches_reviewed` (LLM verdict + human decision + `final_label`). Every decision carries a fingerprint of the text the reviewer saw; changed text makes it stale (kept, not applied, re-queued). The KPI SQL reads `matches_reviewed.final_label`.

**Tech Stack:** Python 3.11, pandas, duckdb, Streamlit (AppTest for dashboard tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-human-review-workflow-design.md`

## Global Constraints

- Python 3.11, pandas, duckdb. SQL lives in `sql/`, one file per query/KPI; every SQL file starts with `-- <name>: <one line>`.
- Never edit `data/raw`. Pipeline outputs go to `data/clean`. Decision files go to `data/reference`.
- Every function in `src/` gets a pytest test in `tests/`. Run `.venv/Scripts/python.exe -m pytest -q` after each change.
- Never invent data or fill gaps with guesses: missing stays NULL and is logged. An alias needs `source_url` and `retrieved_on`.
- Every rate shows its n; rates are NULL/suppressed where n < 10.
- LLM calls only through `src/llm.py`, batched. Tests always fake the LLM.
- Read the API key from `.env`; never print or commit it.
- Files meant for people are written through `src/csv_safe.py`; files the pipeline reads back are not (only the free-text `note` column of decision files is made formula-safe, because nothing reads it back as data).
- After writing a new module, explain it in 5 plain lines.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Refinements to the spec found while planning

1. **`record()` takes a queue item, not `(kind, key)`.** The item carries `kind`, `side`, `make`, `key` and the current `fingerprint`, so `record()` needs no database and the dashboard never opens DuckDB for review (it reads `data/clean/review_queue.csv`, like every other tab reads `data/clean`). Signature: `record(item, decision, note="", *, from_name=None, source_url=None, retrieved_on=None, ref_dir=REF_DIR, today=None) -> list[dict]`.
2. **The queue keeps reviewed items** (status `reviewed`) so the dashboard can show "x of n". Statuses: `open`, `stale`, `reviewed`, `orphaned`.
3. **`fingerprint()` gets its own tiny module** (`src/fingerprint.py`) because `matcher`, `model_family` and `review` all need it and `review` imports the other two.
4. **Explorer label `awaiting_review`** replaces `needs_review` in `echo_explorer`: an LLM `same` or `needs_review` pair that nobody has confirmed. Explorer `same` then always equals a confirmed echo, so explorer and `echo_rate` agree.
5. **`echo_lag` gets no `n_reviewed`:** it only contains confirmed echoes, so its `n` already is the reviewed count. `echo_rate`, `echo_by_component` and `ota` get `n_reviewed`; `echo_rate`'s `needs_review_not_echoing` becomes `awaiting_review_not_echoing`.
6. **Undo:** `record()` returns the list of changes (file, key, previous row); `undo(changes, ref_dir)` restores them.

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/fingerprint.py` | create | `fingerprint(*parts) -> str` |
| `src/matcher.py` | modify | store `fingerprint` and `prompt_version` in `matches`; reuse unchanged verdicts |
| `src/model_family.py` | modify | overrides gain `fingerprint`, `reviewed_at`; stale overrides skipped |
| `src/review.py` | create | decision files, `record`, `undo`, `outstanding`, `queue`, `apply`, `precision`, CLI |
| `sql/matches_reviewed.sql` | create | builds `matches_reviewed` |
| `sql/review_precision.sql` | create | matcher precision from reviewed pairs |
| `sql/echo_rate.sql`, `echo_lag.sql`, `echo_by_component.sql`, `ota.sql`, `echo_explorer.sql` | modify | read `matches_reviewed.final_label` |
| `src/kpis.py` | modify | `check_inputs` requires an up-to-date `matches_reviewed` |
| `src/evaluate.py` | modify | precision section; `human_label` column preferred |
| `data/reference/ground_truth.csv` | modify | add blank `human_label` column (only structural edit; labels untouched) |
| `app/data_sources.py` | modify | `review_queue()`, decision-file stamps |
| `app/dashboard.py` | modify | Review tab; explorer labels |
| `app/pipeline_3d.py`, `app/assets/pipeline_3d.html` | modify | "matched" = confirmed echoes |
| `tests/test_fingerprint.py`, `tests/test_review.py` | create | |
| `tests/test_matcher.py`, `test_model_family.py`, `test_kpis.py`, `test_evaluate.py`, `test_dashboard_data.py`, `test_pipeline_3d.py` | modify | |

---

### Task 1: `src/fingerprint.py`

**Files:**
- Create: `src/fingerprint.py`
- Test: `tests/test_fingerprint.py`

**Interfaces:**
- Produces: `fingerprint(*parts: object) -> str` — 16 lowercase hex chars; `None` counts as `""`; parts are joined with `"\x1f"` so `("ab", "c")` and `("a", "bc")` differ.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for src/fingerprint.py."""

from src.fingerprint import fingerprint


def test_same_parts_same_hash_and_shape():
    a = fingerprint("US text", "India text")
    assert a == fingerprint("US text", "India text")
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_one_changed_character_changes_the_hash():
    assert fingerprint("US text", "India text") != fingerprint("US text", "India texT")


def test_part_boundaries_matter_and_none_is_empty():
    assert fingerprint("ab", "c") != fingerprint("a", "bc")
    assert fingerprint(None, "x") == fingerprint("", "x")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fingerprint.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.fingerprint'`

- [ ] **Step 3: Implement**

```python
"""A short, stable hash of the text a person or the LLM looked at.

Review decisions and matcher verdicts store it; when the underlying text
changes, the hash changes, so an old decision is known to be stale instead of
being silently applied to text nobody reviewed.
"""

from __future__ import annotations

import hashlib

SEP = "\x1f"  # ASCII unit separator: never appears in recall text


def fingerprint(*parts: object) -> str:
    """16 hex chars of SHA-256 over the parts joined by SEP (None counts as "")."""
    blob = SEP.join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fingerprint.py -q` → 3 passed. Then `.venv/Scripts/python.exe -m pytest -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/fingerprint.py tests/test_fingerprint.py
git commit -m "Add fingerprint helper for review decisions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Matcher stores fingerprints and reuses unchanged verdicts

**Files:**
- Modify: `src/matcher.py` (`build_prompt` head → constant; `load_pairs`, `check_result`, `judge`, `build`, `main`)
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: `fingerprint(*parts)` (Task 1).
- Produces:
  - `matcher.PROMPT_VERSION: str` — `fingerprint(SYSTEM, PROMPT_RULES)`.
  - `load_pairs(...)` output gains column `fingerprint` = `fingerprint(us_text, india_text)`.
  - `matches` table gains columns `fingerprint`, `prompt_version`.
  - `judge(pairs, ask=complete_json, batch_size=BATCH, previous: pd.DataFrame | None = None) -> pd.DataFrame`; result has `.attrs["reused"]` and `.attrs["sent"]` (ints).
  - `build(...)` return dict gains `"reused"` and `"sent"`.

Why: the LLM cache key is the whole batch prompt (`src/llm.py:76`); one added pair shifts every later batch of 10 and defeats the cache. Reusing verdicts by fingerprint makes a rebuild cost only the new or changed pairs.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_matcher.py`)

```python
# ---------- fingerprints and reuse ----------

from src.fingerprint import fingerprint  # noqa: E402


def test_prompt_rules_are_unchanged_text():
    # the rules moved into a constant; the prompt must be byte-identical so the LLM cache still hits
    assert mt.build_prompt([PAIR]).startswith(mt.PROMPT_RULES + "\n")
    assert mt.PROMPT_VERSION == fingerprint(mt.SYSTEM, mt.PROMPT_RULES)


def test_check_result_carries_the_pair_fingerprint():
    out = mt.check_result(_result(), {**PAIR, "fingerprint": "abc"})
    assert out["fingerprint"] == "abc"


def _fp_pairs(n):
    p = _pairs(n)
    p["fingerprint"] = [fingerprint(u, i) for u, i in zip(p.us_text, p.india_text)]
    return p


def _echo_fake(calls):
    def fake(prompt, **kw):
        ids = [line.split()[2] for line in prompt.splitlines() if line.startswith("=== PAIR")]
        calls.append(ids)
        return {"results": [_result(pair_id=i) for i in ids]}
    return fake


def test_unchanged_pairs_are_reused_without_a_call():
    first = mt.judge(_fp_pairs(3), ask=_echo_fake([]))
    calls = []
    again = mt.judge(_fp_pairs(3), ask=_echo_fake(calls), previous=first)
    assert calls == [] and again.attrs == {"reused": 3, "sent": 0}
    assert again.label.tolist() == first.label.tolist()


def test_changed_text_or_prompt_version_is_sent_again():
    first = mt.judge(_fp_pairs(3), ask=_echo_fake([]))
    changed = _fp_pairs(3)
    changed.loc[0, "fingerprint"] = "different"
    calls = []
    mt.judge(changed, ask=_echo_fake(calls), previous=first)
    assert calls == [["C0-0"]]
    old_prompt = first.assign(prompt_version="old")
    calls = []
    mt.judge(_fp_pairs(3), ask=_echo_fake(calls), previous=old_prompt)
    assert calls == [["C0-0", "C1-1", "C2-2"]]


def test_previous_without_fingerprints_reuses_nothing():
    calls = []
    out = mt.judge(_fp_pairs(2), ask=_echo_fake(calls),
                   previous=pd.DataFrame({"pair_id": ["C0-0"], "label": ["same"]}))
    assert len(calls) == 1 and out.attrs == {"reused": 0, "sent": 2}
```

Also extend the existing `test_build_writes_matches_table` with:

```python
    cols = [c[0] for c in duckdb.connect(str(db), read_only=True)
            .execute("DESCRIBE matches").fetchall()]
    assert {"fingerprint", "prompt_version"} <= set(cols)
```

(Read the existing test first; use its `db` variable name.)

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_matcher.py -q`
Expected: FAIL — `AttributeError: module 'src.matcher' has no attribute 'PROMPT_RULES'`, `KeyError: 'fingerprint'`, `TypeError: judge() got an unexpected keyword argument 'previous'`.

- [ ] **Step 3: Implement**

3a. Import: `from src.fingerprint import fingerprint`.

3b. Move the `head = (...)` string literal out of `build_prompt` **unchanged, character for character** into a module constant placed just above `build_prompt`, and define the version:

```python
PROMPT_RULES = (
    "For each pair decide whether the US recall and the INDIA recall describe the "
    # ... the existing lines exactly as they are today ...
    "side's evidence - before answering, check each quote occurs in its own block.\n"
)
# Changes whenever the instructions change, so old verdicts are not reused.
PROMPT_VERSION = fingerprint(SYSTEM, PROMPT_RULES)


def build_prompt(batch: list[dict]) -> str:
    """One call's prompt: the rules, then up to BATCH pairs with both texts."""
    blocks = [f"=== PAIR {p['pair_id']} (make {p['make']}) ===\nUS:\n{p['us_text']}\n"
              f"INDIA:\n{p['india_text']}" for p in batch]
    return PROMPT_RULES + "\n" + "\n\n".join(blocks)
```

3c. `load_pairs`: build the texts once and add the fingerprint column:

```python
    us = [us_text(r) for r in df.itertuples(index=False)]
    india = [india_text(r) for r in df.itertuples(index=False)]
    return pd.DataFrame({
        "pair_id": [f"{c}-{r}" for c, r in zip(df.CAMPNO, df.csv_row)],
        "CAMPNO": df.CAMPNO, "make": df.make, "csv_row": df.csv_row,
        "us_text": us, "india_text": india,
        "fingerprint": [fingerprint(u, i) for u, i in zip(us, india)],
        "us_has_defect": [bool(_clean(d)) for d in df.DESC_DEFECT],
        "india_has_defect": [bool(_clean(d)) for d in df.defect_description],
    })
```

3d. `check_result`: add `"fingerprint": pair.get("fingerprint"),` to the initial `out` dict (after `"csv_row"`).

3e. `judge`: add the `previous` parameter and reuse step. Replace the function with:

```python
def _reusable(previous: pd.DataFrame | None) -> dict[str, dict]:
    """pair_id -> earlier verdict judged with this prompt and model, keyed for reuse."""
    need = {"pair_id", "fingerprint", "prompt_version", "llm_model"}
    if previous is None or not need <= set(previous.columns):
        return {}
    ok = previous[(previous.prompt_version == PROMPT_VERSION)
                  & (previous.llm_model == llm.MODEL)]
    return {r["pair_id"]: r for r in ok.to_dict("records")
            if isinstance(r.get("fingerprint"), str) and r["fingerprint"]}


def judge(pairs: pd.DataFrame, ask: Callable[..., object] = complete_json,
          batch_size: int = BATCH, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ask the LLM in batches, then check every answer in Python.

    A pair whose texts (fingerprint), prompt and model match a row in
    `previous` keeps that verdict and is not sent again.
    """
    reuse = _reusable(previous)
    rows, sendable, reused = [], [], 0
    for p in pairs.to_dict("records"):
        reason = None
        if not (p["us_has_defect"] and p["india_has_defect"]):
            side = "US" if not p["us_has_defect"] else "INDIA"
            reason = f"no {side} defect text; not sent to the LLM"
        else:
            for side, key in (("US", "us_text"), ("INDIA", "india_text")):
                phrase = instruction_like(p[key])
                if phrase:
                    reason = (f"{side} text looks addressed to a model ({phrase!r}); "
                              "not sent to the LLM (possible prompt injection)")
                    break
        if reason:
            row = check_result(None, p)
            row["review_reason"] = reason
            rows.append(row)
            continue
        old = reuse.get(p["pair_id"])
        if old is not None and old["fingerprint"] == p.get("fingerprint"):
            rows.append({k: v for k, v in old.items()
                         if k not in ("scope", "llm_model", "prompt_version")})
            reused += 1
        else:
            sendable.append(p)
    for batch in batched(sendable, batch_size):
        answer = ask(build_prompt(batch), system=SYSTEM, temperature=0.0, max_tokens=4096)
        by_id = _results_by_id(answer)
        unknown = set(by_id) - {p["pair_id"] for p in batch}
        if unknown:
            log.warning("LLM returned results for pairs not in this batch: %s", sorted(unknown))
        rows += [check_result(by_id.get(p["pair_id"]), p) for p in batch]
    out = pd.DataFrame(rows)
    out["llm_model"] = llm.MODEL
    out["prompt_version"] = PROMPT_VERSION
    out = out.sort_values(["CAMPNO", "csv_row"]).reset_index(drop=True)
    out.attrs = {"reused": reused, "sent": len(sendable)}
    log.info("matcher: %d verdicts reused, %d pairs sent to the LLM", reused, len(sendable))
    return out
```

3f. `build`: read the previous table before judging and return the counts:

```python
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        previous = con.execute("SELECT * FROM matches").df() if "matches" in tables else None
        judged = judge(load_pairs(con, keys), ask, previous=previous)
        matches = judged.assign(scope=scope)
```

and `return {"matches": matches, "comparison": comparison, "reused": judged.attrs["reused"], "sent": judged.attrs["sent"]}`.

3g. `main`: after `print(report(...))` add
`print(f"verdicts reused: {out['reused']}, pairs sent to the LLM: {out['sent']}")`.
Update `test_main_runs_ground_truth_scope_by_default` if its fake `build` return lacks these keys (add `"reused": 0, "sent": 0`).

3h. Module docstring: add one paragraph: "Each row stores a fingerprint of both texts and the prompt version. On a rerun, a pair whose fingerprint, prompt version and model are unchanged keeps its earlier verdict and is not sent again; only new or changed pairs cost LLM calls."

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_matcher.py tests/test_security.py -q` → all pass. Then the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/matcher.py tests/test_matcher.py
git commit -m "Matcher stores fingerprints and reuses unchanged verdicts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Model-family overrides carry a fingerprint; stale ones are skipped

**Files:**
- Modify: `src/model_family.py` (`OVERRIDE_REVIEW_COLUMNS`, `override_fingerprint`, `load_overrides`, `apply_overrides`, `build`)
- Test: `tests/test_model_family.py`

**Interfaces:**
- Consumes: `fingerprint` (Task 1).
- Produces:
  - `OVERRIDE_REVIEW_COLUMNS = ["fingerprint", "reviewed_at"]` (optional columns; `OVERRIDE_COLUMNS` stays the required five).
  - `override_fingerprint(side: str, make: str, key: str, text: str) -> str` = `fingerprint(side, make, key, text)`. NHTSA text is the MODELTXT (= key); SIAM text is `siam_recalls.model`.
  - `load_overrides(path)` always returns `OVERRIDE_COLUMNS + OVERRIDE_REVIEW_COLUMNS` (missing optional columns filled with `""`).
  - `apply_overrides(nhtsa, siam, overrides, vocab, siam_models: dict[int, str] | None = None)`: an override with a non-blank fingerprint that differs from the current text's is logged as stale and not applied.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model_family.py`)

```python
def test_load_overrides_accepts_files_without_review_columns(tmp_path: Path):
    path = tmp_path / "o.csv"
    pd.DataFrame([["nhtsa", "NISSAN", "ROGUE", "X-Trail", ""]],
                 columns=fam.OVERRIDE_COLUMNS).to_csv(path, index=False)
    df = fam.load_overrides(path)
    assert list(df.columns) == fam.OVERRIDE_COLUMNS + fam.OVERRIDE_REVIEW_COLUMNS
    assert df.fingerprint.tolist() == [""]


def test_stale_override_is_logged_and_not_applied(tmp_path: Path, caplog):
    path = tmp_path / "o.csv"
    good = fam.override_fingerprint("siam", "AUDI", "252", "A6 45 TFSI")
    pd.DataFrame([["siam", "AUDI", "252", "A6", "", good, "2026-10-16"],
                  ["siam", "AUDI", "253", "A6", "", "0000000000000000", "2026-10-16"],
                  ["nhtsa", "NISSAN", "ROGUE", "X-Trail", "",
                   fam.override_fingerprint("nhtsa", "NISSAN", "ROGUE", "ROGUE"), "2026-10-16"]],
                 columns=fam.OVERRIDE_COLUMNS + fam.OVERRIDE_REVIEW_COLUMNS).to_csv(path, index=False)
    nhtsa = pd.DataFrame({"make": ["NISSAN"], "MODELTXT": ["ROGUE"],
                          "family": [None], "method": ["llm"]})
    siam = pd.DataFrame({"csv_row": [253], "make": ["AUDI"], "family": ["A4"], "method": ["rule"]})
    with caplog.at_level("WARNING"):
        n, s = fam.apply_overrides(nhtsa, siam, fam.load_overrides(path),
                                   {"NISSAN": ["X-TRAIL"], "AUDI": ["A4", "A6"]},
                                   siam_models={252: "A6 45 TFSI", 253: "A4 changed text"})
    assert n.family.tolist() == ["X-TRAIL"]
    assert sorted(map(tuple, s.values.tolist())) == [(252, "AUDI", "A6", "override"),
                                                     (253, "AUDI", "A4", "rule")]
    assert "stale override" in caplog.text and "253" in caplog.text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_family.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'OVERRIDE_REVIEW_COLUMNS'`.

- [ ] **Step 3: Implement**

Below `OVERRIDE_COLUMNS = [...]` add:

```python
# Written by src/review.py; optional so hand-written override files still load.
OVERRIDE_REVIEW_COLUMNS = ["fingerprint", "reviewed_at"]
```

Import `from src.fingerprint import fingerprint` and add:

```python
def override_fingerprint(side: str, make: str, key: str, text: object) -> str:
    """Hash of what the reviewer saw: NHTSA model name, or the SIAM row's model text."""
    return fingerprint(side, make, key, text)
```

In `load_overrides`, change the empty-file return to
`pd.DataFrame(columns=OVERRIDE_COLUMNS + OVERRIDE_REVIEW_COLUMNS)` and, after the `bad` side check, add:

```python
    for col in OVERRIDE_REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OVERRIDE_COLUMNS + OVERRIDE_REVIEW_COLUMNS]
```

Replace `apply_overrides` with:

```python
def apply_overrides(nhtsa: pd.DataFrame, siam: pd.DataFrame, overrides: pd.DataFrame,
                    vocab: dict[str, list[str]],
                    siam_models: dict[int, str] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply overrides last. Blank or 'none' family means 'no family'.

    An override whose fingerprint no longer matches the current text (the
    model name or SIAM row changed since review) is stale: logged, not applied.
    SIAM fingerprints are only checked when siam_models (csv_row -> model) is given.
    """
    def target(r) -> str | None:
        fam = r.family.strip()
        if fam == "" or fam.lower() == NONE:
            return None
        if fam.upper() not in {f.upper() for f in vocab.get(r.make, [])}:
            raise ValueError(f"override family {fam!r} for {r.make} is not in the vocabulary")
        return next(f for f in vocab[r.make] if f.upper() == fam.upper())

    def stale(r, text) -> bool:
        fp = str(getattr(r, "fingerprint", "") or "")
        if fp and fp != override_fingerprint(r.side, r.make, r.key, text):
            log.warning("stale override %s %s %r: text changed since review; not applied",
                        r.side, r.make, r.key)
            return True
        return False

    nhtsa, siam = nhtsa.copy(), siam.copy()
    for r in overrides[overrides.side == "nhtsa"].itertuples(index=False):
        if stale(r, r.key):
            continue
        hit = (nhtsa.make == r.make) & (nhtsa.MODELTXT == r.key)
        if not hit.any():
            log.warning("override for %s %r matches no NHTSA model", r.make, r.key)
            continue
        nhtsa.loc[hit, ["family", "method"]] = [target(r), "override"]

    siam_rows = overrides[overrides.side == "siam"]
    if siam_models is not None:
        keep = [not stale(r, siam_models.get(int(r.key)))
                for r in siam_rows.itertuples(index=False)]
        siam_rows = siam_rows[keep]
    for key, group in siam_rows.groupby("key"):
        row = int(key)
        make = group.make.iloc[0]
        siam = siam[siam.csv_row != row]
        new = [(row, make, fam, "override") for fam in map(target, group.itertuples(index=False))
               if fam is not None]
        siam = pd.concat([siam, pd.DataFrame(new, columns=siam.columns)], ignore_index=True)
    return nhtsa, siam
```

In `build`, pass the SIAM model texts:

```python
        nhtsa, siam_fam = apply_overrides(
            nhtsa, siam_fam, load_overrides(overrides_csv), vocab,
            siam_models=dict(zip(siam.csv_row.astype(int), siam.model)))
```

Docstring item 3 ("Overrides") gets one sentence: "Overrides written by src/review.py carry a fingerprint of the reviewed text; if that text has changed, the override is stale and is skipped with a warning."

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_family.py -q` → pass (including the two existing override tests). Full suite.

- [ ] **Step 5: Commit**

```bash
git add src/model_family.py tests/test_model_family.py
git commit -m "Family overrides carry a fingerprint; stale ones are skipped

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `src/review.py` — decision files, `record`, `undo`, `outstanding`

**Files:**
- Create: `src/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `model_family.OVERRIDE_COLUMNS`, `OVERRIDE_REVIEW_COLUMNS`, `ALIAS_COLUMNS`, `load_vocab(path)`; `csv_safe.excel_safe_value`.
- Produces:
  - `REF_DIR: Path`, `FILES: dict[str, str]` (kind → file name), `COLUMNS`, `KEYS`.
  - `read_decisions(kind: str, ref_dir: Path = REF_DIR) -> pd.DataFrame` (all `str`, missing file → empty frame with `COLUMNS[kind]`).
  - `record(item: dict, decision: str, note: str = "", *, from_name=None, source_url=None, retrieved_on=None, ref_dir=REF_DIR, today=None) -> list[dict]`. Each change dict: `{"kind": str, "keys": dict[str, str], "previous": dict | None}`. Raises `ValueError` and writes nothing on invalid input.
  - `undo(changes: list[dict], ref_dir: Path = REF_DIR) -> None`.
  - `outstanding(queue_df: pd.DataFrame, ref_dir: Path = REF_DIR) -> pd.DataFrame` — rows of `queue_df` with status `open`/`stale` that have no current decision in the files (used by the dashboard between `src.review` runs).
- Queue item dict keys used here: `kind` (`pair` | `family` | `vocabulary`), `side` (`nhtsa` | `siam` | `""`), `make`, `key`, `fingerprint`.
- Decision values: pair → `same` | `different`; family → a vocabulary family of that make, or `none`; vocabulary → `distinct` | `alias`.

- [ ] **Step 1: Write the failing tests** (`tests/test_review.py`)

```python
"""Tests for src/review.py. The LLM is never called; the database is a small fake."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import review as rv

PAIR_ITEM = {"kind": "pair", "side": "", "make": "HONDA", "key": "20V001000-1",
             "fingerprint": "aaaaaaaaaaaaaaaa"}
FAMILY_ITEM = {"kind": "family", "side": "nhtsa", "make": "HONDA", "key": "FIT",
               "fingerprint": "bbbbbbbbbbbbbbbb"}
VOCAB_ITEM = {"kind": "vocabulary", "side": "", "make": "HONDA", "key": "CITY HYBRID",
              "fingerprint": "cccccccccccccccc"}


@pytest.fixture
def ref(tmp_path: Path) -> Path:
    d = tmp_path / "reference"
    d.mkdir()
    pd.DataFrame({"make": ["HONDA"] * 3, "family": ["AMAZE", "CITY", "CITY HYBRID"],
                  "source": ["india_models"] * 3}).to_csv(d / "model_families.csv", index=False)
    return d


def _files(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def test_record_pair_writes_one_row_with_fingerprint_and_date(ref):
    changes = rv.record(PAIR_ITEM, "same", "checked both texts", ref_dir=ref, today="2026-10-16")
    df = rv.read_decisions("pair", ref)
    assert df.to_dict("records") == [{"pair_id": "20V001000-1", "decision": "same",
                                      "note": "checked both texts",
                                      "fingerprint": "aaaaaaaaaaaaaaaa",
                                      "reviewed_at": "2026-10-16"}]
    assert changes == [{"kind": "pair", "keys": {"pair_id": "20V001000-1"}, "previous": None}]


def test_second_decision_replaces_the_first(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref, today="2026-10-16")
    changes = rv.record(PAIR_ITEM, "different", "on second look", ref_dir=ref, today="2026-10-17")
    df = rv.read_decisions("pair", ref)
    assert df.decision.tolist() == ["different"]
    assert changes[0]["previous"]["decision"] == "same"


@pytest.mark.parametrize("item, decision, message", [
    (PAIR_ITEM, "maybe", "same or different"),
    (FAMILY_ITEM, "CIVIC", "not in the HONDA vocabulary"),
    (VOCAB_ITEM, "merge", "distinct or alias"),
    ({**PAIR_ITEM, "kind": "nonsense"}, "same", "unknown item kind"),
    ({**PAIR_ITEM, "fingerprint": ""}, "same", "no fingerprint"),
])
def test_invalid_decisions_are_refused_and_nothing_is_written(ref, item, decision, message):
    before = _files(ref)
    with pytest.raises(ValueError, match=message):
        rv.record(item, decision, ref_dir=ref)
    assert _files(ref) == before


def test_family_decision_uses_vocabulary_spelling_or_none(ref):
    rv.record(FAMILY_ITEM, "city", ref_dir=ref, today="2026-10-16")
    rv.record({**FAMILY_ITEM, "key": "PILOT"}, "None", ref_dir=ref, today="2026-10-16")
    df = rv.read_decisions("family", ref)
    assert df[["side", "make", "key", "family"]].values.tolist() == [
        ["nhtsa", "HONDA", "FIT", "CITY"], ["nhtsa", "HONDA", "PILOT", "none"]]
    assert (df.fingerprint == "bbbbbbbbbbbbbbbb").all()


def test_alias_needs_a_source_and_writes_both_files(ref):
    with pytest.raises(ValueError, match="source_url"):
        rv.record(VOCAB_ITEM, "alias", from_name="CITY E:HEV", ref_dir=ref)
    with pytest.raises(ValueError, match="retrieved_on"):
        rv.record(VOCAB_ITEM, "alias", from_name="CITY E:HEV",
                  source_url="https://example.com/city", retrieved_on="16/10/2026", ref_dir=ref)
    changes = rv.record(VOCAB_ITEM, "alias", "same car", from_name="CITY E:HEV",
                        source_url="https://example.com/city", retrieved_on="2026-10-16",
                        ref_dir=ref, today="2026-10-16")
    assert [c["kind"] for c in changes] == ["alias", "vocabulary"]
    alias = pd.read_csv(ref / "model_aliases.csv", dtype=str)
    assert alias[["make", "from_name", "family"]].values.tolist() == [
        ["HONDA", "CITY E:HEV", "CITY HYBRID"]]
    assert rv.read_decisions("vocabulary", ref).decision.tolist() == ["alias"]


def test_distinct_writes_only_the_vocabulary_review_file(ref):
    rv.record(VOCAB_ITEM, "distinct", ref_dir=ref, today="2026-10-16")
    assert not (ref / "model_aliases.csv").exists()
    assert rv.read_decisions("vocabulary", ref).decision.tolist() == ["distinct"]


def test_note_is_made_formula_safe(ref):
    rv.record(PAIR_ITEM, "same", '=HYPERLINK("http://x")', ref_dir=ref)
    assert rv.read_decisions("pair", ref).note.iloc[0].startswith("'=")


def test_record_only_writes_the_decision_files(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    rv.record(FAMILY_ITEM, "CITY", ref_dir=ref)
    rv.record(VOCAB_ITEM, "alias", from_name="X", source_url="https://e.com",
              retrieved_on="2026-10-16", ref_dir=ref)
    assert _files(ref) == {"model_families.csv", *rv.FILES.values()}   # no .tmp left behind


def test_record_keeps_rows_added_by_hand_meanwhile(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    path = ref / rv.FILES["pair"]
    path.write_text(path.read_text(encoding="utf-8")
                    + "20V009000-9,different,by hand,,2026-10-16\n", encoding="utf-8")
    rv.record({**PAIR_ITEM, "key": "20V002000-2"}, "same", ref_dir=ref)
    assert set(rv.read_decisions("pair", ref).pair_id) == {
        "20V001000-1", "20V009000-9", "20V002000-2"}


def test_undo_restores_the_previous_state(ref):
    rv.record(PAIR_ITEM, "same", ref_dir=ref, today="2026-10-16")
    changes = rv.record(PAIR_ITEM, "different", ref_dir=ref, today="2026-10-17")
    rv.undo(changes, ref)
    assert rv.read_decisions("pair", ref).decision.tolist() == ["same"]
    first = rv.record({**PAIR_ITEM, "key": "20V002000-2"}, "same", ref_dir=ref)
    rv.undo(first, ref)
    assert rv.read_decisions("pair", ref).pair_id.tolist() == ["20V001000-1"]


def test_outstanding_drops_items_decided_since_the_queue_was_written(ref):
    q = pd.DataFrame([{**PAIR_ITEM, "status": "open"},
                      {**PAIR_ITEM, "key": "20V002000-2", "status": "open"},
                      {**FAMILY_ITEM, "status": "stale"},
                      {**VOCAB_ITEM, "status": "reviewed"}])
    rv.record(PAIR_ITEM, "same", ref_dir=ref)
    out = rv.outstanding(q, ref)
    assert out[["kind", "key"]].values.tolist() == [["pair", "20V002000-2"], ["family", "FIT"]]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q`
Expected: FAIL with `ImportError: cannot import name 'review' from 'src'`.

- [ ] **Step 3: Implement** (`src/review.py`, first part; Tasks 5–7 add to it)

```python
"""Human review of LLM decisions: build the queue, record decisions, apply them.

Decision files (data/reference, hand-editable, committed):
  pair_reviews.csv            pair_id, decision (same|different), note, fingerprint, reviewed_at
  model_family_overrides.csv  side, make, key, family, note, fingerprint, reviewed_at
                              (read by src/model_family.py)
  vocabulary_reviews.csv      make, family, decision (distinct|alias), note, fingerprint, reviewed_at
  model_aliases.csv           make, from_name, family, source_url, retrieved_on, note
                              (read by src/model_family.py; an alias needs a source)

Every decision stores a fingerprint of the text the reviewer saw. If that text
changes later the decision is stale: kept, not applied, and the item returns
to the queue with the old decision shown. A decision whose item no longer
exists is orphaned: kept, not applied, logged.

Table matches_reviewed = matches + human decision; final_label is the decision
when it is current, else NULL. KPIs count an echo only when final_label =
'same'. The LLM's own label is never changed.

The dashboard and `--import` both go through record(), so the rules are the same.

Run:  .venv/Scripts/python.exe -m src.review            apply decisions, write the queue
      .venv/Scripts/python.exe -m src.review --import data/clean/review_queue.csv
      .venv/Scripts/python.exe -m src.review --rebuild  rerun the pipeline after family decisions
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from src import model_family
from src.csv_safe import excel_safe_value

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data/reference"
VOCAB_FILE = "model_families.csv"

FILES = {"pair": "pair_reviews.csv",
         "family": "model_family_overrides.csv",
         "vocabulary": "vocabulary_reviews.csv",
         "alias": "model_aliases.csv"}
COLUMNS = {"pair": ["pair_id", "decision", "note", "fingerprint", "reviewed_at"],
           "family": model_family.OVERRIDE_COLUMNS + model_family.OVERRIDE_REVIEW_COLUMNS,
           "vocabulary": ["make", "family", "decision", "note", "fingerprint", "reviewed_at"],
           "alias": model_family.ALIAS_COLUMNS}
KEYS = {"pair": ["pair_id"], "family": ["side", "make", "key"],
        "vocabulary": ["make", "family"], "alias": ["make", "from_name"]}
PAIR_DECISIONS = ("same", "different")
VOCAB_DECISIONS = ("distinct", "alias")

log = logging.getLogger(__name__)


def _s(value: object) -> str:
    """Text or ''; never 'nan'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def read_decisions(kind: str, ref_dir: Path = REF_DIR) -> pd.DataFrame:
    """One decision file as text; an absent file is an empty frame. Optional columns filled."""
    path = ref_dir / FILES[kind]
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS[kind])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in COLUMNS[kind]:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS[kind]]


def _write_atomic(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _mask(df: pd.DataFrame, keys: dict[str, str]) -> pd.Series:
    hit = pd.Series(True, index=df.index)
    for k, v in keys.items():
        hit &= df[k] == v
    return hit


def _upsert(kind: str, row: dict, ref_dir: Path) -> dict:
    """Replace the row with the same key (re-reading the file first); return the change."""
    df = read_decisions(kind, ref_dir)
    keys = {k: row[k] for k in KEYS[kind]}
    hit = _mask(df, keys)
    previous = df[hit].iloc[0].to_dict() if hit.any() else None
    out = pd.concat([df[~hit], pd.DataFrame([row], columns=COLUMNS[kind])], ignore_index=True)
    _write_atomic(ref_dir / FILES[kind], out)
    return {"kind": kind, "keys": keys, "previous": previous}


def _family_value(make: str, decision: str, ref_dir: Path) -> str:
    if decision.lower() == model_family.NONE:
        return model_family.NONE
    allowed = {f.upper(): f for f in model_family.load_vocab(ref_dir / VOCAB_FILE).get(make, [])}
    if decision.upper() not in allowed:
        raise ValueError(f"family {decision!r} is not in the {make} vocabulary "
                         f"(choose one of {sorted(allowed.values())} or 'none')")
    return allowed[decision.upper()]


def record(item: dict, decision: str, note: str = "", *, from_name: str | None = None,
           source_url: str | None = None, retrieved_on: str | None = None,
           ref_dir: Path = REF_DIR, today: str | None = None) -> list[dict]:
    """Validate one decision on a queue item and write it; return the changes (for undo).

    Raises ValueError, writing nothing, when the decision is not allowed for the
    item, a family is outside the vocabulary, or an alias lacks a source.
    """
    kind, make, key = _s(item.get("kind")), _s(item.get("make")), _s(item.get("key"))
    fp, decision = _s(item.get("fingerprint")), _s(decision)
    today = today or date.today().isoformat()
    note = excel_safe_value(_s(note))
    if kind not in ("pair", "family", "vocabulary"):
        raise ValueError(f"unknown item kind {kind!r}")
    if not fp:
        raise ValueError(f"item {key!r} has no fingerprint; rebuild the queue with src.review")

    if kind == "pair":
        if decision not in PAIR_DECISIONS:
            raise ValueError(f"a pair decision must be same or different, not {decision!r}")
        return [_upsert("pair", {"pair_id": key, "decision": decision, "note": note,
                                 "fingerprint": fp, "reviewed_at": today}, ref_dir)]

    if kind == "family":
        family = _family_value(make, decision, ref_dir)
        return [_upsert("family", {"side": _s(item.get("side")), "make": make, "key": key,
                                   "family": family, "note": note, "fingerprint": fp,
                                   "reviewed_at": today}, ref_dir)]

    if decision not in VOCAB_DECISIONS:
        raise ValueError(f"a vocabulary decision must be distinct or alias, not {decision!r}")
    alias_row = None
    if decision == "alias":
        name, url, when = _s(from_name), _s(source_url), _s(retrieved_on)
        if not name:
            raise ValueError("an alias needs from_name (the other spelling)")
        if not url.startswith(("http://", "https://")):
            raise ValueError("an alias needs a source_url (http or https) showing both names are one model")
        try:
            date.fromisoformat(when)
        except ValueError:
            raise ValueError(f"an alias needs retrieved_on as YYYY-MM-DD, not {when!r}") from None
        alias_row = {"make": make, "from_name": name, "family": key, "source_url": url,
                     "retrieved_on": when, "note": note}
    changes = [_upsert("alias", alias_row, ref_dir)] if alias_row else []
    changes.append(_upsert("vocabulary", {"make": make, "family": key, "decision": decision,
                                          "note": note, "fingerprint": fp,
                                          "reviewed_at": today}, ref_dir))
    return changes


def undo(changes: list[dict], ref_dir: Path = REF_DIR) -> None:
    """Put back what record() replaced (or remove what it added), newest change first."""
    for change in reversed(changes):
        kind = change["kind"]
        df = read_decisions(kind, ref_dir)
        df = df[~_mask(df, change["keys"])]
        if change["previous"] is not None:
            df = pd.concat([df, pd.DataFrame([change["previous"]], columns=COLUMNS[kind])],
                           ignore_index=True)
        _write_atomic(ref_dir / FILES[kind], df)


def _current_keys(ref_dir: Path) -> set[tuple[str, str, str, str, str]]:
    """(kind, side, make, key, fingerprint) of every decision in the files."""
    out = set()
    for r in read_decisions("pair", ref_dir).itertuples(index=False):
        out.add(("pair", "", "", r.pair_id, r.fingerprint))
    for r in read_decisions("family", ref_dir).itertuples(index=False):
        out.add(("family", r.side, r.make, r.key, r.fingerprint))
    for r in read_decisions("vocabulary", ref_dir).itertuples(index=False):
        out.add(("vocabulary", "", r.make, r.family, r.fingerprint))
    return out


def outstanding(queue_df: pd.DataFrame, ref_dir: Path = REF_DIR) -> pd.DataFrame:
    """Open or stale items with no current decision on file yet."""
    done = _current_keys(ref_dir)

    def decided(r) -> bool:
        side = _s(r.side) if r.kind == "family" else ""
        make = "" if r.kind == "pair" else _s(r.make)
        return (r.kind, side, make, _s(r.key), _s(r.fingerprint)) in done

    todo = queue_df[queue_df.status.isin(["open", "stale"])]
    keep = [not decided(r) for r in todo.itertuples(index=False)]
    return todo[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q` → all pass. Full suite.

- [ ] **Step 5: Commit**

```bash
git add src/review.py tests/test_review.py
git commit -m "Add review decision files with validated record and undo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `review.queue()` — what needs a person

**Files:**
- Modify: `src/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `matcher.load_pairs(con, keys)` (with `fingerprint`, Task 2); `matches.fingerprint` (Task 2); `model_family.override_fingerprint`, `vocabulary_review(vocab_df, models)` (Task 3); `read_decisions` (Task 4); `fingerprint` (Task 1).
- Produces:
  - `QUEUE_COLUMNS` (below); `ORDER` (below); `DEFAULT_ORDERS = (1, 2, 3, 4, 5)`.
  - `queue(con, ref_dir=REF_DIR) -> pd.DataFrame` with `QUEUE_COLUMNS`, sorted by `order, make, key`. Status `open` | `stale` | `reviewed` | `orphaned`; `previous_decision` filled for stale/reviewed/orphaned.
  - `item_id` = `f"{kind}|{side}|{make}|{key}"` (unique; used by `--import`).
  - Vocabulary fingerprint = `fingerprint("vocabulary", make, family, review_note)`.

Orders: 1 LLM family mapping (NHTSA, method `llm`, family set); 2 SIAM row with zero families; 3 vocabulary flag; 4 `needs_review` pair; 5 LLM `same` pair; 6 LLM "none" family answer (filter only); 7 orphaned decision.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_review.py`)

```python
# ---------- queue ----------

import duckdb  # noqa: E402

from src import matcher as mt  # noqa: E402
from src import model_family as fam  # noqa: E402


def _db(tmp_path: Path) -> Path:
    """HONDA only. Pairs: 20V001000-1 same, 20V001000-2 different, 20V002000-2 needs_review.
    Families: FIT llm->CITY, PILOT llm->none, CITY rule. SIAM row 3 has zero families."""
    db = tmp_path / "r.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE india_relevant_campaigns AS SELECT * FROM (VALUES
        ('20V001000', 'HONDA', ['AIR BAGS'], 'Driver air bag inflator may rupture.', 'Injury.'),
        ('20V002000', 'HONDA', ['SEATS'], 'Seat may detach from the floor.', 'Injury.'))
        AS t(CAMPNO, make, components, DESC_DEFECT, CONEQUENCE_DEFECT)""")
    con.execute("""CREATE TABLE siam_recalls AS SELECT * FROM (VALUES
        (1, 'HONDA', 'City', 'Driver airbag inflator may rupture.'),
        (2, 'HONDA', 'Amaze', 'Seat rail may detach.'),
        (3, 'HONDA', 'Mystery Van', 'Brake hose may leak.'))
        AS t(csv_row, make, model, defect_description)""")
    con.execute("""CREATE TABLE siam_row_family AS SELECT * FROM (VALUES
        (1, 'HONDA', 'CITY', 'rule'), (2, 'HONDA', 'AMAZE', 'rule'))
        AS t(csv_row, make, family, method)""")
    con.execute("""CREATE TABLE nhtsa_model_family AS SELECT * FROM (VALUES
        ('HONDA', 'CITY', 'CITY', 'rule'), ('HONDA', 'FIT', 'CITY', 'llm'),
        ('HONDA', 'PILOT', NULL, 'llm')) AS t(make, MODELTXT, family, method)""")
    con.execute("CREATE TABLE nhtsa_focus AS SELECT make, MODELTXT FROM nhtsa_model_family")
    keys = pd.DataFrame({"CAMPNO": ["20V001000", "20V001000", "20V002000"], "csv_row": [1, 2, 2]})
    p = mt.load_pairs(con, keys)
    labels = {"20V001000-1": "same", "20V001000-2": "different", "20V002000-2": "needs_review"}
    m = p[["pair_id", "CAMPNO", "make", "csv_row", "fingerprint"]].assign(
        label=p.pair_id.map(labels), llm_label=p.pair_id.map(labels).replace("needs_review", "same"),
        confidence=0.9, us_evidence="us quote", india_evidence="india quote",
        review_reason=p.pair_id.map({"20V002000-2": "india_evidence not found"}),
        llm_model="fake", prompt_version=mt.PROMPT_VERSION, scope="all")
    con.register("m", m)
    con.execute("CREATE TABLE matches AS SELECT * FROM m")
    con.close()
    return db


def _queue(tmp_path, ref):
    con = duckdb.connect(str(_db(tmp_path)), read_only=True)
    try:
        return rv.queue(con, ref)
    finally:
        con.close()


def test_queue_lists_every_kind_in_order(tmp_path, ref):
    q = _queue(tmp_path, ref)
    assert list(q.columns) == rv.QUEUE_COLUMNS
    assert q[["order", "kind", "side", "key", "status"]].values.tolist() == [
        [1, "family", "nhtsa", "FIT", "open"],
        [2, "family", "siam", "3", "open"],
        [3, "vocabulary", "", "CITY HYBRID", "open"],
        [4, "pair", "", "20V002000-2", "open"],
        [5, "pair", "", "20V001000-1", "open"],
        [6, "family", "nhtsa", "PILOT", "open"]]
    assert q.item_id.is_unique
    pair = q[q.key == "20V001000-1"].iloc[0]
    assert pair.us_text.startswith("Component: AIR BAGS") and pair.llm_answer == "same"
    assert q[q.key == "FIT"].iloc[0].llm_answer == "CITY"


def test_current_decisions_show_as_reviewed(tmp_path, ref):
    q = _queue(tmp_path, ref)
    for item in q.to_dict("records")[:5]:
        decision = {"family": "none", "vocabulary": "distinct", "pair": "same"}[item["kind"]]
        rv.record(item, decision, ref_dir=ref)
    q2 = _queue(tmp_path, ref)
    assert q2.status.tolist()[:5] == ["reviewed"] * 5
    assert q2.previous_decision.tolist()[:5] == ["none", "none", "distinct", "same", "same"]


def test_changed_text_makes_a_decision_stale(tmp_path, ref):
    q = _queue(tmp_path, ref)
    item = q[q.key == "20V001000-1"].iloc[0].to_dict()
    rv.record({**item, "fingerprint": "0" * 16}, "different", ref_dir=ref)
    row = _queue(tmp_path, ref).set_index("key").loc["20V001000-1"]
    assert (row.status, row.previous_decision) == ("stale", "different")


def test_decision_for_a_vanished_item_is_orphaned(tmp_path, ref):
    rv.record({**PAIR_ITEM, "key": "99V999000-9"}, "same", ref_dir=ref)
    rv.record({**FAMILY_ITEM, "key": "GONE MODEL"}, "CITY", ref_dir=ref)
    q = _queue(tmp_path, ref)
    orphans = q[q.status == "orphaned"]
    assert set(orphans.key) == {"99V999000-9", "GONE MODEL"} and set(orphans.order) == {7}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q`
Expected: FAIL with `AttributeError: module 'src.review' has no attribute 'queue'`.

- [ ] **Step 3: Implement** (add to `src/review.py`; extend imports with `import duckdb`, `from src.fingerprint import fingerprint`, `from src.matcher import load_pairs`)

```python
QUEUE_COLUMNS = ["item_id", "order", "kind", "side", "make", "key", "status",
                 "llm_answer", "confidence", "us_text", "india_text", "us_evidence",
                 "india_evidence", "reason", "fingerprint", "previous_decision"]
ORDER = {"family_llm": 1, "family_zero": 2, "vocabulary": 3, "needs_review": 4,
         "same": 5, "family_none": 6, "orphaned": 7}
DEFAULT_ORDERS = (1, 2, 3, 4, 5)


def _item(order: int, kind: str, side: str, make: str, key: str, **kw) -> dict:
    row = {c: None for c in QUEUE_COLUMNS}
    row.update(order=order, kind=kind, side=side, make=make, key=str(key), status="open",
               item_id=f"{kind}|{side}|{make}|{key}", **kw)
    return row


def _resolve(items: dict, decisions: list[tuple], lookup) -> list[dict]:
    """Attach decisions to items.

    items: key tuple -> open item. decisions: (key tuple, decision value, fingerprint).
    lookup(key tuple) -> item for an existing thing that is not a default candidate, or None.
    Current decision (blank or equal fingerprint) -> reviewed; different -> stale;
    nothing to attach to -> orphaned.
    """
    items = dict(items)
    for k, value, fp in decisions:
        item = items.get(k) or lookup(k)
        if item is None:
            kind, side, make, key = k
            items[k] = _item(ORDER["orphaned"], kind, side, make, key, status="orphaned",
                             previous_decision=value, reason="decision has no matching item")
            continue
        status = "reviewed" if fp in ("", item["fingerprint"]) else "stale"
        items[k] = {**item, "status": status, "previous_decision": value}
    return list(items.values())


def _family_items(con: duckdb.DuckDBPyConnection, ref_dir: Path) -> list[dict]:
    nhtsa = con.execute("SELECT make, MODELTXT, family, method FROM nhtsa_model_family").df()
    siam = con.execute("SELECT csv_row, make, model FROM siam_recalls").df()
    with_family = {int(r[0]) for r in con.execute(
        "SELECT DISTINCT csv_row FROM siam_row_family").fetchall()}

    def nhtsa_item(r, order):
        fp = model_family.override_fingerprint("nhtsa", r.make, r.MODELTXT, r.MODELTXT)
        return _item(order, "family", "nhtsa", r.make, r.MODELTXT, us_text=r.MODELTXT,
                     llm_answer=r.family if r.method == "llm" else None,
                     reason="an LLM chose this family" if r.method == "llm" else None,
                     fingerprint=fp)

    def siam_item(r):
        fp = model_family.override_fingerprint("siam", r.make, str(r.csv_row), r.model)
        return _item(ORDER["family_zero"], "family", "siam", r.make, str(r.csv_row),
                     india_text=r.model, fingerprint=fp,
                     reason="SIAM row produced zero families" if r.csv_row not in with_family
                     else "SIAM row has a family decision")

    items = {}
    for r in nhtsa[nhtsa.method == "llm"].itertuples(index=False):
        order = ORDER["family_llm"] if isinstance(r.family, str) and r.family else ORDER["family_none"]
        it = nhtsa_item(r, order)
        it["llm_answer"] = it["llm_answer"] or "none"
        items[("family", "nhtsa", r.make, r.MODELTXT)] = it
    for r in siam[~siam.csv_row.isin(with_family)].itertuples(index=False):
        items[("family", "siam", r.make, str(r.csv_row))] = siam_item(r)

    by_nhtsa = {(r.make, r.MODELTXT): r for r in nhtsa.itertuples(index=False)}
    by_siam = {(r.make, str(r.csv_row)): r for r in siam.itertuples(index=False)}

    def lookup(k):
        _, side, make, key = k
        if side == "nhtsa" and (make, key) in by_nhtsa:
            return nhtsa_item(by_nhtsa[(make, key)], ORDER["family_llm"])
        if side == "siam" and (make, key) in by_siam:
            return siam_item(by_siam[(make, key)])
        return None

    dec = [(("family", r.side, r.make, r.key), r.family, r.fingerprint)
           for r in read_decisions("family", ref_dir).itertuples(index=False)]
    return _resolve(items, dec, lookup)


def _vocabulary_items(con: duckdb.DuckDBPyConnection, ref_dir: Path) -> list[dict]:
    vocab = pd.read_csv(ref_dir / VOCAB_FILE, dtype=str, keep_default_na=False)
    models = con.execute("SELECT DISTINCT make, MODELTXT FROM nhtsa_focus "
                         "WHERE MODELTXT IS NOT NULL").df()
    flags = (model_family.vocabulary_review(vocab, models)
             .groupby(["make", "family"], as_index=False).review_note.agg(" | ".join))
    items = {("vocabulary", "", r.make, r.family): _item(
        ORDER["vocabulary"], "vocabulary", "", r.make, r.family, india_text=r.family,
        reason=r.review_note, fingerprint=fingerprint("vocabulary", r.make, r.family, r.review_note))
        for r in flags.itertuples(index=False)}
    dec = [(("vocabulary", "", r.make, r.family), r.decision, r.fingerprint)
           for r in read_decisions("vocabulary", ref_dir).itertuples(index=False)]
    return _resolve(items, dec, lambda k: None)


def _pair_items(con: duckdb.DuckDBPyConnection, ref_dir: Path) -> list[dict]:
    decisions = read_decisions("pair", ref_dir)
    m = con.execute("""SELECT pair_id, CAMPNO, make, csv_row, label, llm_label, confidence,
                              us_evidence, india_evidence, review_reason, fingerprint
                       FROM matches""").df()
    wanted = m[m.label.isin(["same", "needs_review"]) | m.pair_id.isin(decisions.pair_id)]
    texts = load_pairs(con, wanted[["CAMPNO", "csv_row"]]).set_index("pair_id")

    def pair_item(r):
        order = ORDER["same"] if r.label == "same" else ORDER["needs_review"]
        t = texts.loc[r.pair_id] if r.pair_id in texts.index else None
        return _item(order, "pair", "", r.make, r.pair_id,
                     llm_answer=r.llm_label if r.label != "needs_review" else "needs_review",
                     confidence=r.confidence, us_evidence=r.us_evidence,
                     india_evidence=r.india_evidence, reason=r.review_reason,
                     us_text=None if t is None else t.us_text,
                     india_text=None if t is None else t.india_text,
                     fingerprint=r.fingerprint)

    by_id = {r.pair_id: r for r in wanted.itertuples(index=False)}
    items = {("pair", "", r.make, r.pair_id): pair_item(r)
             for r in wanted[wanted.label.isin(["same", "needs_review"])].itertuples(index=False)}
    make_of = {pid: r.make for pid, r in by_id.items()}

    def lookup(k):
        r = by_id.get(k[3])
        return pair_item(r) if r is not None else None

    dec = [(("pair", "", make_of.get(r.pair_id, ""), r.pair_id), r.decision, r.fingerprint)
           for r in decisions.itertuples(index=False)]
    return _resolve(items, dec, lookup)


def queue(con: duckdb.DuckDBPyConnection, ref_dir: Path = REF_DIR) -> pd.DataFrame:
    """Every review item with its status, in review order (families first)."""
    rows = _family_items(con, ref_dir) + _vocabulary_items(con, ref_dir) + _pair_items(con, ref_dir)
    q = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    q["side"] = q.side.fillna("")
    return q.sort_values(["order", "make", "key"], kind="stable").reset_index(drop=True)
```

Note: if `matches` lacks a `fingerprint` column, `_pair_items` fails with a DuckDB binder error. Catch it in `queue` and re-raise `ValueError("matches has no fingerprint column; run python -m src.matcher --all")` (wrap the `_pair_items` call in `try/except duckdb.BinderException`).

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q` → pass. Full suite.

- [ ] **Step 5: Commit**

```bash
git add src/review.py tests/test_review.py
git commit -m "Build the review queue with open, stale, reviewed and orphaned items

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `matches_reviewed`, `apply()` and `precision()`

**Files:**
- Create: `sql/matches_reviewed.sql`, `sql/review_precision.sql`
- Modify: `src/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `matches` with `fingerprint` (Task 2); `read_decisions("pair")` (Task 4).
- Produces:
  - Table `matches_reviewed`: all `matches` columns + `human_decision`, `review_status` (`unreviewed` | `reviewed` | `stale`), `final_label`.
  - `apply(db_path=DB_PATH, ref_dir=REF_DIR) -> dict[str, int]` with keys `reviewed`, `stale`, `unreviewed`, `orphaned`.
  - `precision(con) -> pd.DataFrame` one row: `n_reviewed`, `n_confirmed`, `precision` (NULL when `n_reviewed < 10`).

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---------- apply / precision ----------

def _fp(db, pair_id):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT fingerprint FROM matches WHERE pair_id = ?", [pair_id]).fetchone()[0]
    finally:
        con.close()


def test_apply_sets_final_label_only_for_current_decisions(tmp_path, ref):
    db = _db(tmp_path)
    rv.record({**PAIR_ITEM, "key": "20V001000-1", "fingerprint": _fp(db, "20V001000-1")},
              "same", ref_dir=ref)
    rv.record({**PAIR_ITEM, "key": "20V002000-2", "fingerprint": "0" * 16}, "same", ref_dir=ref)
    rv.record({**PAIR_ITEM, "key": "99V999000-9"}, "different", ref_dir=ref)
    counts = rv.apply(db, ref)
    assert counts == {"reviewed": 1, "stale": 1, "unreviewed": 1, "orphaned": 1}
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("""SELECT pair_id, label, human_decision, review_status, final_label
                          FROM matches_reviewed ORDER BY pair_id""").fetchall()
    assert rows == [("20V001000-1", "same", "same", "reviewed", "same"),
                    ("20V001000-2", "different", None, "unreviewed", None),
                    ("20V002000-2", "needs_review", "same", "stale", None)]
    con.close()


def test_apply_refuses_bad_decision_values(tmp_path, ref):
    db = _db(tmp_path)
    (ref / rv.FILES["pair"]).write_text(
        "pair_id,decision,note,fingerprint,reviewed_at\n20V001000-1,yes,,x,2026-10-16\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="same/different"):
        rv.apply(db, ref)


def _reviewed_db(tmp_path, n_same, n_confirmed):
    db = tmp_path / "p.duckdb"
    con = duckdb.connect(str(db))
    con.execute(f"""CREATE TABLE matches_reviewed AS
        SELECT 'same' AS label, 'reviewed' AS review_status,
               CASE WHEN i < {n_confirmed} THEN 'same' ELSE 'different' END AS final_label
        FROM range({n_same}) AS t(i)""")
    con.execute("INSERT INTO matches_reviewed VALUES ('same', 'unreviewed', NULL), "
                "('different', 'reviewed', 'different')")
    return con


def test_precision_counts_reviewed_llm_same_pairs_only(tmp_path):
    con = _reviewed_db(tmp_path, 12, 9)
    row = rv.precision(con).iloc[0]
    assert (row.n_reviewed, row.n_confirmed, row.precision) == (12, 9, 0.75)


def test_precision_is_suppressed_below_ten(tmp_path):
    con = _reviewed_db(tmp_path, 9, 9)
    row = rv.precision(con).iloc[0]
    assert row.n_reviewed == 9 and pd.isna(row.precision)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q`
Expected: FAIL with `AttributeError: ... no attribute 'apply'`.

- [ ] **Step 3: Implement**

`sql/matches_reviewed.sql`:

```sql
-- matches_reviewed: matches plus the human decision from pair_reviews.csv; final_label is that decision when its fingerprint matches the current texts, else NULL.
--
-- Input: matches, pair_reviews_df (registered from data/reference/pair_reviews.csv).
-- review_status: unreviewed (no decision), reviewed (fingerprint matches),
-- stale (texts changed since review: decision kept but not applied).
-- The LLM's label and llm_label are copied unchanged.
CREATE OR REPLACE TABLE matches_reviewed AS
SELECT m.*,
       r.decision                                                AS human_decision,
       CASE WHEN r.pair_id IS NULL THEN 'unreviewed'
            WHEN r.fingerprint = m.fingerprint THEN 'reviewed'
            ELSE 'stale' END                                     AS review_status,
       CASE WHEN r.fingerprint = m.fingerprint THEN r.decision END AS final_label
FROM matches AS m
LEFT JOIN pair_reviews_df AS r ON r.pair_id = m.pair_id
ORDER BY m.CAMPNO, m.csv_row;
```

`sql/review_precision.sql`:

```sql
-- review_precision: of the pairs the LLM matcher labelled same and a person reviewed, how many the person confirmed; precision is NULL where n_reviewed < 10.
--
-- Input: matches_reviewed. This is precision only: it says nothing about echoes
-- the matcher missed (LLM `different` pairs are not reviewed).
SELECT count(*)                                              AS n_reviewed,
       count(*) FILTER (WHERE final_label = 'same')          AS n_confirmed,
       CASE WHEN count(*) >= 10
            THEN round(count(*) FILTER (WHERE final_label = 'same') / count(*), 4) END AS precision
FROM matches_reviewed
WHERE label = 'same' AND review_status = 'reviewed';
```

In `src/review.py` (add `from src.load_nhtsa import DB_PATH`, `SQL_DIR = ROOT / "sql"`):

```python
def _sql(name: str) -> str:
    return (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")


def apply(db_path: Path = DB_PATH, ref_dir: Path = REF_DIR) -> dict[str, int]:
    """Write matches_reviewed from matches and pair_reviews.csv; return status counts."""
    pairs = read_decisions("pair", ref_dir)
    bad = sorted(set(pairs.decision) - set(PAIR_DECISIONS))
    if bad:
        raise ValueError(f"{FILES['pair']} has decisions other than same/different: {bad}")
    con = duckdb.connect(str(db_path))
    try:
        cols = {r[0] for r in con.execute("DESCRIBE matches").fetchall()}
        if "fingerprint" not in cols:
            raise ValueError("matches has no fingerprint column; run python -m src.matcher --all")
        con.register("pair_reviews_df", pairs)
        con.execute(_sql("matches_reviewed"))
        con.unregister("pair_reviews_df")
        counts = dict(con.execute("SELECT review_status, count(*) FROM matches_reviewed "
                                  "GROUP BY 1").fetchall())
        known = {r[0] for r in con.execute("SELECT pair_id FROM matches").fetchall()}
    finally:
        con.close()
    orphans = sorted(set(pairs.pair_id) - known)
    if orphans:
        log.warning("%d pair decisions match no pair in matches (kept, not applied): %s",
                    len(orphans), ", ".join(orphans[:20]))
    out = {s: int(counts.get(s, 0)) for s in ("reviewed", "stale", "unreviewed")}
    out["orphaned"] = len(orphans)
    if out["stale"]:
        log.warning("%d pair decisions are stale (texts changed); re-queued", out["stale"])
    return out


def precision(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Matcher precision from human-reviewed LLM `same` pairs (NULL where n < 10)."""
    return con.execute(_sql("review_precision")).df()
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q` → pass. Full suite.

- [ ] **Step 5: Commit**

```bash
git add sql/matches_reviewed.sql sql/review_precision.sql src/review.py tests/test_review.py
git commit -m "Apply review decisions into matches_reviewed; add review precision

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Review CLI — queue sheet, import, rebuild

**Files:**
- Modify: `src/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `queue`, `record`, `apply` (Tasks 4–6); `model_family.build`, `candidates.build`, `matcher.build(scope="all")`, `kpis.build`; `csv_safe.excel_safe`.
- Produces:
  - `QUEUE_CSV = ROOT / "data/clean/review_queue.csv"`; `SHEET_INPUT_COLUMNS = ["decision", "note", "from_name", "source_url", "retrieved_on"]`.
  - `write_queue(db_path=DB_PATH, ref_dir=REF_DIR, out_csv=QUEUE_CSV) -> pd.DataFrame` — formula-safe sheet: `QUEUE_COLUMNS + SHEET_INPUT_COLUMNS` (inputs blank).
  - `import_sheet(path, db_path=DB_PATH, ref_dir=REF_DIR) -> tuple[int, list[str]]` — (recorded, error lines).
  - `rebuild(steps=None) -> None` — runs the pipeline steps in order.
  - `main(argv=None)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---------- CLI ----------

def test_write_queue_is_formula_safe_with_blank_inputs(tmp_path, ref):
    db = _db(tmp_path)
    out = tmp_path / "queue.csv"
    rv.write_queue(db, ref, out)
    sheet = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(sheet.columns) == rv.QUEUE_COLUMNS + rv.SHEET_INPUT_COLUMNS
    assert (sheet.decision == "").all() and len(sheet) == 6


def test_import_records_filled_rows_and_reports_bad_ones(tmp_path, ref):
    db = _db(tmp_path)
    out = tmp_path / "queue.csv"
    rv.write_queue(db, ref, out)
    sheet = pd.read_csv(out, dtype=str, keep_default_na=False)
    sheet.loc[sheet.key == "20V001000-1", ["decision", "note"]] = ["same", "ok"]
    sheet.loc[sheet.key == "FIT", "decision"] = "CIVIC"           # not in vocabulary
    extra = sheet.iloc[[0]].assign(item_id="pair||HONDA|00X000000-1", decision="same")
    pd.concat([sheet, extra]).to_csv(out, index=False)
    recorded, errors = rv.import_sheet(out, db, ref)
    assert recorded == 1
    assert len(errors) == 2
    assert any("CIVIC" in e for e in errors) and any("not in the current queue" in e for e in errors)
    assert rv.read_decisions("pair", ref).pair_id.tolist() == ["20V001000-1"]


def test_rebuild_runs_steps_in_order():
    ran = []
    rv.rebuild(steps=[(name, lambda n=name: ran.append(n)) for name in ("a", "b", "c")])
    assert ran == ["a", "b", "c"]


def test_main_applies_and_prints_counts(monkeypatch, capsys):
    monkeypatch.setattr(rv, "apply", lambda: {"reviewed": 1, "stale": 0, "unreviewed": 2,
                                              "orphaned": 0})
    monkeypatch.setattr(rv, "write_queue", lambda: pd.DataFrame(
        {"order": [1, 5], "kind": ["family", "pair"], "status": ["open", "reviewed"]}))
    rv.main([])
    out = capsys.readouterr().out
    assert "matches_reviewed: reviewed=1" in out and "family" in out


def test_main_import_then_apply(monkeypatch, capsys, tmp_path):
    seen = []
    monkeypatch.setattr(rv, "import_sheet", lambda p: seen.append(p) or (2, ["x: bad"]))
    monkeypatch.setattr(rv, "apply", lambda: {"reviewed": 2, "stale": 0, "unreviewed": 0,
                                              "orphaned": 0})
    monkeypatch.setattr(rv, "write_queue", lambda: pd.DataFrame(
        {"order": [5], "kind": ["pair"], "status": ["reviewed"]}))
    rv.main(["--import", str(tmp_path / "q.csv")])
    out = capsys.readouterr().out
    assert seen == [tmp_path / "q.csv"] and "recorded 2" in out and "x: bad" in out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q`
Expected: FAIL with `AttributeError: ... no attribute 'write_queue'`.

- [ ] **Step 3: Implement** (add `import sys` and `from src.csv_safe import excel_safe`)

```python
QUEUE_CSV = ROOT / "data/clean/review_queue.csv"
SHEET_INPUT_COLUMNS = ["decision", "note", "from_name", "source_url", "retrieved_on"]


def write_queue(db_path: Path = DB_PATH, ref_dir: Path = REF_DIR,
                out_csv: Path = QUEUE_CSV) -> pd.DataFrame:
    """Write the queue as a formula-safe sheet with blank decision columns; return the queue."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        q = queue(con, ref_dir)
    finally:
        con.close()
    sheet = q.assign(**{c: "" for c in SHEET_INPUT_COLUMNS})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    excel_safe(sheet).to_csv(out_csv, index=False, encoding="utf-8")
    return q


def import_sheet(path: Path, db_path: Path = DB_PATH,
                 ref_dir: Path = REF_DIR) -> tuple[int, list[str]]:
    """Record every row of a filled-in queue sheet that has a decision; collect refusals."""
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        current = queue(con, ref_dir).set_index("item_id")
    finally:
        con.close()
    recorded, errors = 0, []
    for r in sheet[sheet.decision.str.strip() != ""].to_dict("records"):
        item_id = r["item_id"]
        if item_id not in current.index:
            errors.append(f"{item_id}: not in the current queue (rerun src.review and refill)")
            continue
        item = {"item_id": item_id, **current.loc[item_id].to_dict()}
        try:
            record(item, r["decision"], r.get("note", ""), from_name=r.get("from_name"),
                   source_url=r.get("source_url"), retrieved_on=r.get("retrieved_on"),
                   ref_dir=ref_dir)
            recorded += 1
        except ValueError as exc:
            errors.append(f"{item_id}: {exc}")
    return recorded, errors


def _pipeline_steps() -> list[tuple[str, object]]:
    from src import candidates, kpis, matcher  # imported here: heavy, and only for --rebuild
    return [("model_family", model_family.build),
            ("candidates", candidates.build),
            ("matcher --all", lambda: matcher.build(scope="all")),
            ("review", apply),
            ("kpis", kpis.build)]


def rebuild(steps: list[tuple[str, object]] | None = None) -> None:
    """Rerun the pipeline after family or vocabulary decisions, in dependency order."""
    for name, step in steps or _pipeline_steps():
        log.info("rebuild: %s", name)
        step()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if "--rebuild" in args:
        rebuild()
    if "--import" in args:
        recorded, errors = import_sheet(Path(args[args.index("--import") + 1]))
        print(f"recorded {recorded} decisions; refused {len(errors)}")
        for e in errors:
            print(f"  {e}")
    counts = apply()
    print("matches_reviewed: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    q = write_queue()
    default = q[q.order.isin(DEFAULT_ORDERS)]
    print(f"queue ({QUEUE_CSV.relative_to(ROOT).as_posix()}):")
    print(default.groupby(["kind", "status"]).size().to_string())


if __name__ == "__main__":
    main()
```

(`main` prints `QUEUE_CSV.relative_to(ROOT)`; the test monkeypatches `write_queue` and leaves `QUEUE_CSV` as the repo path, so this is safe.)

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_review.py -q` → pass. Full suite.

- [ ] **Step 5: Commit**

```bash
git add src/review.py tests/test_review.py
git commit -m "Add review CLI: queue sheet, import and rebuild

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: KPIs count only confirmed echoes

**Files:**
- Modify: `sql/echo_rate.sql`, `sql/echo_lag.sql`, `sql/echo_by_component.sql`, `sql/ota.sql`, `sql/echo_explorer.sql`, `src/kpis.py`, `app/pipeline_3d.py`, `app/assets/pipeline_3d.html`
- Test: `tests/test_kpis.py`, `tests/test_pipeline_3d.py`

**Interfaces:**
- Consumes: table `matches_reviewed` (Task 6).
- Produces:
  - `echo_rate` columns: `level, grp, n, echoing, echo_rate, n_reviewed, awaiting_review_not_echoing`.
  - `echo_by_component` gains `n_reviewed`; `ota` gains `n_reviewed` (NULL on `ota_share` rows).
  - `echo_explorer` labels: `same` (confirmed), `awaiting_review`, `different`, `no_candidate`; new last column `review_status`.
  - A campaign is *settled* when none of its pairs is an LLM `same`/`needs_review` still lacking a current decision; `n_reviewed` = settled campaigns.

- [ ] **Step 1: Update the fake and write the failing tests** in `tests/test_kpis.py`

In `_fake`, after the `CREATE TABLE candidate_pairs` statement add:

```python
    # every LLM `same` pair confirmed by a person; needs_review / different unreviewed
    con.execute("""CREATE TABLE matches_reviewed AS SELECT *,
        CASE WHEN label = 'same' THEN 'same' END                         AS human_decision,
        CASE WHEN label = 'same' THEN 'reviewed' ELSE 'unreviewed' END   AS review_status,
        CASE WHEN label = 'same' THEN 'same' END                         AS final_label
        FROM matches""")
```

Add to the module docstring: "matches_reviewed: every `same` pair confirmed; the rest unreviewed."

Change the first assertion of `test_echo_rate_by_make_segment_and_all` to:

```python
    assert (a.n, a.echoing, a.echo_rate, a.n_reviewed, a.awaiting_review_not_echoing) == (
        12, 3, 0.25, 11, 1)                                 # A03 awaits review
```

In `test_echo_explorer_one_row_per_campaign_with_best_pair` change the A03 line and the column list:

```python
    assert by.loc["A03", "label"] == "awaiting_review" and pd.isna(by.loc["A03", "lag_days"])
    ...
                                 "india_defect", "lag_days", "confidence", "us_evidence",
                                 "india_evidence", "review_status"]
```

Append new tests:

```python
def _unconfirm(db, campno):
    con = duckdb.connect(str(db))
    con.execute("UPDATE matches_reviewed SET review_status = 'unreviewed', final_label = NULL, "
                "human_decision = NULL WHERE CAMPNO = ?", [campno])
    con.close()


def test_unconfirmed_llm_echo_does_not_count(tmp_path):
    db, makes = _fake(tmp_path)
    _unconfirm(db, "A02")
    r = kpis.build(db_path=db, out_dir=tmp_path / "o", makes_csv=makes)
    a = _row(r["echo_rate"], level="make", grp="A")
    assert (a.echoing, a.n_reviewed, a.awaiting_review_not_echoing) == (2, 10, 2)
    assert _row(r["echo_lag"], level="make", grp="A").n == 2
    assert r["echo_by_component"].iloc[0].echoing == 2
    assert r["echo_explorer"].set_index("CAMPNO").loc["A02", "label"] == "awaiting_review"


def test_person_rejecting_an_llm_echo_shows_different(tmp_path):
    db, makes = _fake(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("UPDATE matches_reviewed SET human_decision = 'different', "
                "final_label = 'different' WHERE CAMPNO = 'A00'")
    con.close()
    r = kpis.build(db_path=db, out_dir=tmp_path / "o", makes_csv=makes)
    assert _row(r["echo_rate"], level="make", grp="A").echoing == 2
    assert r["echo_explorer"].set_index("CAMPNO").loc["A00", "label"] == "different"


def test_check_inputs_requires_current_matches_reviewed(tmp_path):
    db, makes = _fake(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("DELETE FROM matches_reviewed WHERE CAMPNO = 'B1'")
    con.close()
    with pytest.raises(ValueError, match="src.review"):
        kpis.build(db_path=db, out_dir=tmp_path / "o", makes_csv=makes)


def test_kpi_rows_carry_n_reviewed(results):
    assert "n_reviewed" in results["echo_by_component"].columns
    by = results["ota"][results["ota"].part == "echo_by_ota"]
    assert by.n_reviewed.notna().all()
```

In `tests/test_pipeline_3d.py` (line ~24) replace the fake `matches` table with:

```python
    con.execute("CREATE TABLE matches_reviewed (CAMPNO VARCHAR, final_label VARCHAR)")
    con.execute("INSERT INTO matches_reviewed VALUES ('A','same'),('A','same'),"
                "('B','different'),('C','same'),('D',NULL)")
```

and the comment on the `matched` assertion to `# distinct campaigns a person confirmed`.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kpis.py tests/test_pipeline_3d.py -q`
Expected: FAIL (`AttributeError: 'Series' object has no attribute 'n_reviewed'`, explorer label `needs_review`, etc.).

- [ ] **Step 3: Implement**

`sql/echo_rate.sql` — replace the whole file:

```sql
-- echo_rate: share of India-relevant campaigns (campaign x make) with at least one pair a person confirmed as `same`, by make, by segment and overall.
--
-- Inputs: india_relevant_campaigns, matches_reviewed, make_segments (make,
-- segment from india_makes.csv). An echo needs final_label = 'same' (a current
-- human decision); an LLM `same` nobody confirmed is not an echo. A campaign is
-- settled when none of its LLM `same` / needs_review pairs still awaits a
-- current decision; n_reviewed counts settled campaigns.
-- echo_rate is NULL where n < 10 (CLAUDE.md).
WITH camp AS (
    SELECT c.CAMPNO, c.make, s.segment,
           coalesce(bool_or(m.final_label = 'same'), FALSE)                     AS echoes,
           coalesce(bool_or(m.label IN ('same', 'needs_review')
                            AND m.review_status <> 'reviewed'), FALSE)          AS awaiting
    FROM india_relevant_campaigns AS c
    JOIN make_segments AS s USING (make)
    LEFT JOIN matches_reviewed AS m ON m.CAMPNO = c.CAMPNO AND m.make = c.make
    GROUP BY c.CAMPNO, c.make, s.segment
)
SELECT CASE WHEN grouping(make) = 0 THEN 'make'
            WHEN grouping(segment) = 0 THEN 'segment' ELSE 'all' END   AS level,
       coalesce(make, segment, 'ALL')                                  AS grp,
       count(*)                                                        AS n,
       count(*) FILTER (WHERE echoes)                                  AS echoing,
       CASE WHEN count(*) >= 10
            THEN round(count(*) FILTER (WHERE echoes) / count(*), 4) END AS echo_rate,
       count(*) FILTER (WHERE NOT awaiting)                            AS n_reviewed,
       count(*) FILTER (WHERE NOT echoes AND awaiting)                 AS awaiting_review_not_echoing
FROM camp
GROUP BY GROUPING SETS ((make), (segment), ())
ORDER BY level, grp;
```

`sql/echo_lag.sql` — in the first comment line replace "among pairs labelled `same`" with "among pairs a person confirmed as `same`"; in the Inputs line replace `matches` with `matches_reviewed`; add the comment line `-- n counts confirmed echoes only, so it is already the reviewed count.`; and replace the join:

```sql
    JOIN matches_reviewed AS m
      ON m.CAMPNO = c.CAMPNO AND m.make = c.make AND m.final_label = 'same'
```

`sql/echo_by_component.sql` — first comment line: "echo rate (echo = confirmed by a person) by component group ..."; Inputs `matches_reviewed`; `camp` CTE becomes:

```sql
WITH camp AS (
    SELECT c.CAMPNO, c.make,
           nullif(trim(split_part(c.COMPNAME, ':', 1)), '')      AS component_group,
           coalesce(bool_or(m.final_label = 'same'), FALSE)      AS echoes,
           coalesce(bool_or(m.label IN ('same', 'needs_review')
                            AND m.review_status <> 'reviewed'), FALSE) AS awaiting
    FROM india_relevant_campaigns AS c
    LEFT JOIN matches_reviewed AS m ON m.CAMPNO = c.CAMPNO AND m.make = c.make
    GROUP BY c.CAMPNO, c.make, c.COMPNAME
)
```

and add `count(*) FILTER (WHERE NOT awaiting) AS n_reviewed,` after the `echo_rate` select item.

`sql/ota.sql` — Inputs `matches_reviewed`; in `camp` use `coalesce(bool_or(m.final_label = 'same'), FALSE) AS echoes`, add the same `awaiting` expression, join `matches_reviewed AS m`; add `CAST(NULL AS BIGINT) AS n_reviewed` as the last column of `share` and `count(*) FILTER (WHERE NOT awaiting) AS n_reviewed` as the last column of `by_ota`. Add `o.ota` stays in GROUP BY unchanged.

`sql/echo_explorer.sql` — replace the whole file:

```sql
-- echo_explorer: one row per India-relevant campaign x make with its best pair (confirmed same > awaiting review > different), or label no_candidate when it had no candidate pair.
--
-- Inputs: india_relevant_campaigns, matches_reviewed, candidate_pairs,
-- siam_recalls, make_segments. label: a person's decision where it is current
-- (same / different); awaiting_review for an LLM `same` or needs_review pair
-- nobody has confirmed; otherwise the LLM's `different`. "Best pair": a
-- confirmed `same` with the earliest India recall_date (the date echo_lag
-- uses), else an awaiting_review pair, else the different pair with the best
-- TF-IDF rank. lag_days only for `same`. Missing values stay NULL.
WITH shown AS (
    SELECT m.*,
           CASE WHEN m.review_status = 'reviewed' THEN m.final_label
                WHEN m.label IN ('same', 'needs_review') THEN 'awaiting_review'
                ELSE m.label END AS shown_label
    FROM matches_reviewed AS m
),
ranked AS (
    SELECT m.CAMPNO, m.make, m.csv_row, m.shown_label AS label, m.review_status,
           m.confidence, m.us_evidence, m.india_evidence,
           s.recall_date, s.model, s.defect_description,
           row_number() OVER (
               PARTITION BY m.CAMPNO, m.make
               ORDER BY CASE m.shown_label WHEN 'same' THEN 0 WHEN 'awaiting_review' THEN 1 ELSE 2 END,
                        CASE WHEN m.shown_label = 'same' THEN s.recall_date END NULLS LAST,
                        p.rank, m.csv_row) AS pick
    FROM shown AS m
    JOIN candidate_pairs AS p
      ON p.CAMPNO = m.CAMPNO AND p.make = m.make AND p.csv_row = m.csv_row
    JOIN siam_recalls AS s ON s.csv_row = m.csv_row
)
SELECT c.CAMPNO, c.make, ms.segment, c.report_date,
       c.COMPNAME                                     AS component,
       c.DESC_DEFECT                                  AS us_defect,
       coalesce(b.label, 'no_candidate')              AS label,
       b.recall_date                                  AS india_recall_date,
       b.model                                        AS india_model,
       b.defect_description                           AS india_defect,
       CASE WHEN b.label = 'same'
            THEN CAST(b.recall_date - c.report_date AS INTEGER) END AS lag_days,
       b.confidence, b.us_evidence, b.india_evidence,
       b.review_status
FROM india_relevant_campaigns AS c
JOIN make_segments AS ms USING (make)
LEFT JOIN ranked AS b ON b.CAMPNO = c.CAMPNO AND b.make = c.make AND b.pick = 1
ORDER BY c.report_date DESC, c.CAMPNO, c.make;
```

`src/kpis.py`:
- Docstring: replace "A campaign (campaign x make) "echoes" when it has at least one pair labelled `same` in matches." with "A campaign (campaign x make) "echoes" when a person confirmed at least one of its pairs as `same` (matches_reviewed.final_label; run src.review first)."
- In `check_inputs`, add `"matches_reviewed"` to the required-tables tuple and, after the candidate-pairs check:

```python
    n_m, n_r = con.execute("SELECT (SELECT count(*) FROM matches), "
                           "(SELECT count(*) FROM matches_reviewed)").fetchone()
    if n_m != n_r:
        raise ValueError(f"matches_reviewed has {n_r} rows but matches has {n_m}; "
                         "run python -m src.review")
```

- In `main`, the explorer summary line stays (`label` counts).

`app/pipeline_3d.py` line 39:

```python
    "matched": "SELECT count(DISTINCT CAMPNO) FROM matches_reviewed WHERE final_label = 'same'",
```

`app/assets/pipeline_3d.html`: line 164 `" echoes found"` → `" echoes confirmed"`; line 186 replace `" US campaigns where the LLM found an India echo, quotes checked by code"` with `" US campaigns where a person confirmed the LLM's India echo"`.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kpis.py tests/test_pipeline_3d.py tests/test_dashboard_data.py -q` → pass. Full suite.

- [ ] **Step 5: Commit**

```bash
git add sql/echo_rate.sql sql/echo_lag.sql sql/echo_by_component.sql sql/ota.sql sql/echo_explorer.sql src/kpis.py app/pipeline_3d.py app/assets/pipeline_3d.html tests/test_kpis.py tests/test_pipeline_3d.py
git commit -m "KPIs count only echoes a person confirmed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Evaluation — reviewed precision and `human_label`

**Files:**
- Modify: `src/evaluate.py`, `data/reference/ground_truth.csv` (add empty column only)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `review.precision(con)` (Task 6).
- Produces:
  - `load_ground_truth` returns `label` = `human_label` where filled, else the original label; adds `label_source` (`human` | `claude`). Accepts files without `human_label`.
  - `human_verified(gt) -> bool` — True only when every row has a `human_label`.
  - `reviewed_precision(con) -> dict | None` — `{"n_reviewed", "n_confirmed", "precision"}`, None when `matches_reviewed` is missing.
  - Report section "## Matcher precision (human-reviewed)"; `eval_summary.csv` gains a row `matcher = "llm_reviewed"` with `n`, `tp` (= confirmed), `fp` (= rejected), `precision`, `precision_n`; other metric cells empty.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_evaluate.py`; read the file first and reuse its fake-database helper — named below `_fake_db(tmp_path) -> (db_path, gt_path)`; if its name differs, use the real one)

```python
def test_human_label_overrides_claude_label(tmp_path):
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"pair_id": ["20V001000-1", "20V002000-2"], "label": ["same", "different"],
                  "human_label": ["different", ""], "reason": ["", ""]}).to_csv(gt, index=False)
    df = ev.load_ground_truth(gt)
    assert df.label.tolist() == ["different", "different"]
    assert df.label_source.tolist() == ["human", "claude"]
    assert not ev.human_verified(df)


def test_ground_truth_without_human_label_column_still_loads(tmp_path):
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"pair_id": ["20V001000-1"], "label": ["same"]}).to_csv(gt, index=False)
    assert ev.load_ground_truth(gt).label_source.tolist() == ["claude"]


def test_bad_human_label_is_refused(tmp_path):
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"pair_id": ["20V001000-1"], "label": ["same"],
                  "human_label": ["maybe"]}).to_csv(gt, index=False)
    with pytest.raises(ValueError, match="human_label"):
        ev.load_ground_truth(gt)


def test_reviewed_precision_none_without_table(tmp_path):
    con = duckdb.connect(str(tmp_path / "x.duckdb"))
    assert ev.reviewed_precision(con) is None


def test_report_and_summary_include_reviewed_precision(tmp_path):
    db, gt = _fake_db(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE matches_reviewed AS
        SELECT 'same' AS label, 'reviewed' AS review_status,
               CASE WHEN i < 11 THEN 'same' ELSE 'different' END AS final_label
        FROM range(12) AS t(i)""")
    con.close()
    out = tmp_path / "eval.md"
    ev.build(db_path=db, gt_path=gt, out_md=out)
    text = out.read_text(encoding="utf-8")
    assert "## Matcher precision (human-reviewed)" in text and "11 of 12" in text
    summary = pd.read_csv(tmp_path / "eval_summary.csv")
    row = summary[summary.matcher == "llm_reviewed"].iloc[0]
    assert (row.n, row.tp, row.fp, row.precision, row.precision_n) == (12, 11, 1, 0.9167, 12)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluate.py -q`
Expected: FAIL (`AttributeError: ... 'label_source'`, `no attribute 'human_verified'`).

- [ ] **Step 3: Implement**

Replace `load_ground_truth`:

```python
def load_ground_truth(path: Path = GROUND_TRUTH_CSV) -> pd.DataFrame:
    """Read the labels (read-only). A filled human_label wins over the original label.

    Adds CAMPNO / csv_row parsed from pair_id and label_source ('human' | 'claude').
    """
    gt = pd.read_csv(path, dtype=str, keep_default_na=False)
    bad = sorted(set(gt.label) - {"same", "different"})
    if bad:
        raise ValueError(f"ground truth has labels other than same/different: {bad}")
    human = gt["human_label"].str.strip() if "human_label" in gt else pd.Series("", index=gt.index)
    bad = sorted(set(human) - {"", "same", "different"})
    if bad:
        raise ValueError(f"ground truth human_label must be same, different or blank: {bad}")
    gt = gt.assign(label=human.where(human != "", gt.label),
                   label_source=["human" if h else "claude" for h in human])
    keys = gt.pair_id.str.rsplit("-", n=1, expand=True)
    return gt.assign(CAMPNO=keys[0], csv_row=keys[1].astype(int))


def human_verified(gt: pd.DataFrame) -> bool:
    """True only when a person labelled every ground-truth pair."""
    return bool(len(gt)) and bool((gt.label_source == "human").all())


def reviewed_precision(con: duckdb.DuckDBPyConnection) -> dict | None:
    """Matcher precision from the human review of LLM `same` pairs, or None before review."""
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    if "matches_reviewed" not in tables:
        return None
    from src.review import precision
    row = precision(con).iloc[0]
    return {"n_reviewed": int(row.n_reviewed), "n_confirmed": int(row.n_confirmed),
            "precision": None if pd.isna(row.precision) else float(row.precision)}
```

In `build`, compute it while the connection is open: `reviewed = reviewed_precision(con)` and pass it on: `render_markdown(result, texts, labelled_by, gt_path, reviewed)` and `summary_frame(result, reviewed)`. Also when `human_verified(gt)` is True, set `labelled_by = "a person (human_label, all pairs)"`; otherwise keep the current value and append `f"; human_label filled for {int((gt.label_source == 'human').sum())} of {len(gt)}"`.

`render_markdown(..., reviewed: dict | None = None)`: before `"## Disagreements"` insert:

```python
    lines += ["## Matcher precision (human-reviewed)", ""]
    if reviewed is None or reviewed["n_reviewed"] == 0:
        lines += ["No LLM `same` pairs have been reviewed yet (run `python -m src.review`).", ""]
    else:
        n, ok = reviewed["n_reviewed"], reviewed["n_confirmed"]
        lines += [f"A person confirmed {ok} of {n} reviewed LLM `same` pairs: precision "
                  f"{fmt_rate(ok / n, ok, n)}.",
                  "This is precision only. It says nothing about echoes the matcher missed; "
                  "recall comes from the 30-pair ground truth above.", ""]
```

`summary_frame(result, reviewed: dict | None = None)`: after the loop, if `reviewed` and `reviewed["n_reviewed"]`:

```python
        n, ok = reviewed["n_reviewed"], reviewed["n_confirmed"]
        rows.append({"matcher": "llm_reviewed", "n": n, "positives": None, "tp": ok,
                     "fp": n - ok, "fn": None, "tn": None,
                     "precision": round(ok / n, 4) if n >= MIN_N else None, "precision_n": n,
                     "recall": None, "recall_n": None, "f1": None, "f1_n": None,
                     "threshold": None,
                     "note": "precision from human review of LLM same pairs; not recall"})
```

Module docstring: add "Also reports the matcher's precision from the human review of LLM `same` pairs (table matches_reviewed), and uses a filled `human_label` in ground_truth.csv in place of Claude's label."

Add the empty column to the ground truth (structural change only; no label touched):

```bash
.venv/Scripts/python.exe -c "import pandas as pd; p='data/reference/ground_truth.csv'; d=pd.read_csv(p, dtype=str, keep_default_na=False); d.insert(d.columns.get_loc('label')+1, 'human_label', ''); d.to_csv(p, index=False)"
git diff --stat data/reference/ground_truth.csv
```

Expected diff: every line changes only by one added empty field.

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_evaluate.py tests/test_dashboard_data.py -q` → pass (the dashboard filters `matcher == "llm"`, so the new row does not disturb the F1 tile). Full suite.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py data/reference/ground_truth.csv
git commit -m "Evaluation: precision from human review; human_label wins over Claude label

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Dashboard Review tab

**Files:**
- Modify: `app/data_sources.py`, `app/dashboard.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: `data/clean/review_queue.csv` (Task 7); `review.record`, `review.undo`, `review.outstanding`, `review.FILES`, `review.DEFAULT_ORDERS` (Tasks 4–5); `model_family.load_vocab`; `eval_summary.csv` row `llm_reviewed` (Task 9).
- Produces:
  - `ds.FILES["review_queue"] = "data/clean/review_queue.csv"`; `ds.review_queue() -> pd.DataFrame | None` (all columns as text).
  - `ds.ref_dir() -> Path` = `root() / "data/reference"`.
  - `ds.decisions_newer_than_queue() -> bool` — any decision file's mtime > the queue file's mtime.
  - `ds.stamp()` also covers the decision files (so the cache refreshes after a write).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dashboard_data.py`)

```python
# ---------- review ----------

QUEUE = ("item_id,order,kind,side,make,key,status,llm_answer,confidence,us_text,india_text,"
         "us_evidence,india_evidence,reason,fingerprint,previous_decision,decision,note,"
         "from_name,source_url,retrieved_on\n"
         "family|nhtsa|BIGCO|FIT,1,family,nhtsa,BIGCO,FIT,open,CITY,,FIT,,,,an LLM chose this family,"
         "aaaaaaaaaaaaaaaa,,,,,,\n"
         "pair||BIGCO|20V001000-1,5,pair,,BIGCO,20V001000-1,open,same,0.9,"
         "Defect: The driver air bag may rupture.,Defect: Driver airbag inflator may rupture.,"
         "driver air bag may rupture,airbag inflator may rupture,,bbbbbbbbbbbbbbbb,,,,,,\n"
         "pair||BIGCO|20V003000-3,5,pair,,BIGCO,20V003000-3,reviewed,same,0.8,a,b,,,,"
         "cccccccccccccccc,same,,,,,\n")


@pytest.fixture
def review_project(project):
    (project / "data/clean/review_queue.csv").write_text(QUEUE, encoding="utf-8")
    ref = project / "data/reference"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "model_families.csv").write_text("make,family,source\nBIGCO,CITY,india_models\n",
                                            encoding="utf-8")
    return project


def test_review_queue_reads_as_text(review_project):
    q = ds.review_queue()
    assert len(q) == 3 and q.key.tolist()[0] == "FIT" and q.fingerprint.iloc[0] == "a" * 16


def test_decisions_newer_than_queue(review_project):
    assert not ds.decisions_newer_than_queue()
    time.sleep(0.05)
    (review_project / "data/reference/pair_reviews.csv").write_text(
        "pair_id,decision,note,fingerprint,reviewed_at\n", encoding="utf-8")
    assert ds.decisions_newer_than_queue()


def test_review_tab_shows_progress_and_first_item(review_project):
    at = _run(review_project)
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "1 of 2" in text                      # pairs reviewed (one reviewed of two)
    assert "FIT" in text                         # families first


def test_review_tab_records_a_decision_through_review_record(review_project):
    at = _run(review_project)
    at.selectbox(key="review_family_choice").select("CITY").run()
    at.button(key="review_confirm").click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = pd.read_csv(review_project / "data/reference/model_family_overrides.csv", dtype=str)
    assert df[["side", "make", "key", "family"]].values.tolist() == [["nhtsa", "BIGCO", "FIT", "CITY"]]
    text = " ".join(m.value for m in at.markdown)
    assert "src.review --rebuild" in text        # family decisions need a rebuild


def test_review_tab_without_queue_says_what_to_run(project):
    text = " ".join(m.value for m in _run(project).markdown)
    assert "python -m src.review" in text


def test_explorer_names_awaiting_review(project):
    path = project / "data/clean/echo_explorer.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",same,", ",awaiting_review,"),
                    encoding="utf-8")
    at = _run(project)
    assert not at.exception
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_data.py -q`
Expected: FAIL (`AttributeError: module 'data_sources' has no attribute 'review_queue'`).

- [ ] **Step 3: Implement**

`app/data_sources.py`:

```python
FILES = {
    # ... existing entries ...
    "review_queue": "data/clean/review_queue.csv",
}
DECISION_FILES = ("pair_reviews.csv", "model_family_overrides.csv",
                  "vocabulary_reviews.csv", "model_aliases.csv")


def ref_dir() -> Path:
    return root() / "data/reference"


def stamp() -> tuple:
    """(file, mtime) for every output and decision file: changes whenever one is rewritten."""
    files = [path(n) for n in FILES] + [ref_dir() / f for f in DECISION_FILES]
    return tuple((str(p), p.stat().st_mtime if p.exists() else None) for p in files)


def review_queue() -> pd.DataFrame | None:
    """The review queue written by `python -m src.review`, every cell as text."""
    p = path("review_queue")
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    need = ["item_id", "order", "kind", "side", "make", "key", "status", "fingerprint"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise FormatError(f"{FILES['review_queue']} is missing columns {missing}; "
                          "re-run python -m src.review")
    return df


def decisions_newer_than_queue() -> bool:
    """True when a decision was saved after the queue (and matches_reviewed) were built."""
    q = path("review_queue")
    if not q.exists():
        return False
    built = q.stat().st_mtime
    return any((ref_dir() / f).exists() and (ref_dir() / f).stat().st_mtime > built
               for f in DECISION_FILES)
```

Update the module docstring's file list with the queue line. `test_stamp_changes_when_a_file_is_rewritten` keeps passing (tuple still changes).

`app/dashboard.py`:

1. Import `src.review` (repo root on `sys.path`, next to the existing `sys.path` insert for `APP`):

```python
ROOT = APP.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import model_family, review  # noqa: E402
```

2. Tabs: `tab_overview, tab_how, tab_explore, tab_review, tab_method = st.tabs(["Overview", "How it works", "Recall explorer", "Review", "Method & limits"])`.

3. Explorer: add `"awaiting_review": "Awaiting review"` to both label maps; rename `"same"` to `"India echo confirmed"` in both maps; the multiselect option list becomes `["same", "awaiting_review", "different", "no_candidate"]`. Replace the caption block:

```python
            if label == "same":
                st.caption("A person confirmed this match. Highlighted text is the evidence the "
                           "LLM quoted; Python confirmed both quotes exist in the source text.")
            elif label == "different":
                st.caption("No match: the LLM's verdict, or a person's where the pair was reviewed. "
                           "Highlighted text is the evidence the LLM quoted.")
            elif label == "awaiting_review":
                st.caption("Awaiting review: the LLM proposed a match or its answer could not be "
                           "checked. It does not count as an echo until a person confirms it.")
```

4. Overview echo-rate tile subtitle: append `f" · {int(r['n_reviewed'])} of {int(r['n'])} campaigns fully reviewed"` when `"n_reviewed" in r.index`.

5. The Review tab (before `with tab_method:`):

```python
# ------------------------------------------------------------------ review ---
KIND_NAMES = {"family": "Model family", "vocabulary": "Vocabulary", "pair": "Echo candidate"}


def _progress(q: pd.DataFrame, kind: str) -> str:
    part = q[(q["kind"] == kind) & q["order"].astype(int).isin(review.DEFAULT_ORDERS)]
    done = int((part["status"] == "reviewed").sum())
    return f"{done} of {len(part)}"


def _save(item: dict, decision: str, **kw) -> None:
    try:
        changes = review.record(item, decision, st.session_state.get("review_note", ""),
                                ref_dir=ds.ref_dir(), **kw)
    except ValueError as exc:
        st.session_state["review_error"] = str(exc)
        return
    st.session_state["review_last"] = changes
    st.session_state["review_error"] = None
    if item["kind"] != "pair":
        st.session_state["review_needs_rebuild"] = True


with tab_review:
    q = data("review_queue")
    if q is None:
        empty_state("review queue", "python -m src.review")
    else:
        st.markdown("<div class='tiles'>" + "".join([
            tile("Families reviewed", _progress(q, "family"), "LLM mappings and rows with no family", None, INK_2),
            tile("Vocabulary reviewed", _progress(q, "vocabulary"), "possible same-model spellings", None, INK_2),
            tile("Echo candidates reviewed", _progress(q, "pair"), "every LLM same and needs_review pair", None, MATCH),
            tile("Stale", str(int((q["status"] == "stale").sum())), "text changed since review", None, US),
        ]) + "</div>", unsafe_allow_html=True)
        prec = evals[evals["matcher"].astype(str) == "llm_reviewed"] if evals is not None else pd.DataFrame()
        if not prec.empty and pd.notna(prec.iloc[0]["precision"]):
            p = prec.iloc[0]
            st.markdown(f"<span class='note'>Matcher precision from your review: {p['precision']:.0%} "
                        f"(n={int(p['precision_n'])}).</span>", unsafe_allow_html=True)
        if ds.decisions_newer_than_queue():
            st.markdown("<div class='card note'>Decisions saved since the last run are not in the "
                        "KPI tabs yet. Run <code>python -m src.review</code> and refresh.</div>",
                        unsafe_allow_html=True)
        if st.session_state.get("review_needs_rebuild"):
            st.markdown("<div class='card note'>Family and vocabulary decisions change which "
                        "candidate pairs exist. Run <code>python -m src.review --rebuild</code>.</div>",
                        unsafe_allow_html=True)
        if st.session_state.get("review_error"):
            st.error(st.session_state["review_error"])

        f1, f2, f3 = st.columns(3)
        kinds = f1.multiselect("Kind", list(KIND_NAMES), format_func=KIND_NAMES.get)
        makes = f2.multiselect("Carmaker", sorted(q["make"].unique()))
        extra = f3.checkbox("Also show LLM 'none' family answers")
        todo = review.outstanding(q, ds.ref_dir())
        orders = list(review.DEFAULT_ORDERS) + ([6] if extra else [])
        todo = todo[todo["order"].astype(int).isin(orders)]
        if kinds:
            todo = todo[todo["kind"].isin(kinds)]
        if makes:
            todo = todo[todo["make"].isin(makes)]
        orphans = q[q["status"] == "orphaned"]
        if len(orphans):
            st.caption(f"{len(orphans)} saved decision(s) match nothing in the current data; "
                       "they are kept but not applied.")

        if todo.empty:
            st.markdown("<div class='card note'>Nothing left to review with these filters.</div>",
                        unsafe_allow_html=True)
        else:
            item = todo.iloc[0].to_dict()
            st.markdown(f"<span class='note'>{len(todo):,} item(s) to review · "
                        f"{html.escape(KIND_NAMES[item['kind']])} · {html.escape(item['make'])}"
                        "</span>", unsafe_allow_html=True)
            if item["status"] == "stale":
                st.markdown(f"<div class='card note'>The text changed since you decided "
                            f"<b>{html.escape(item['previous_decision'])}</b>. Please decide again."
                            "</div>", unsafe_allow_html=True)
            st.text_input("Note (optional)", key="review_note")

            if item["kind"] == "pair":
                a, b = st.columns(2, gap="large")
                a.markdown(f"<div class='card'><div class='note'>US · {html.escape(item['key'])}</div>"
                           f"<p>{highlight(item['us_text'], item.get('us_evidence'))}</p></div>",
                           unsafe_allow_html=True)
                b.markdown(f"<div class='card'><div class='note'>India</div>"
                           f"<p>{highlight(item['india_text'], item.get('india_evidence'))}</p></div>",
                           unsafe_allow_html=True)
                st.caption(f"LLM said {item['llm_answer']}"
                           + (f" (confidence {float(item['confidence']):.2f})" if item.get("confidence") else "")
                           + (f" · {item['reason']}" if item.get("reason") else ""))
                c1, c2, _ = st.columns([1, 1, 4])
                c1.button("Same", key="review_same", on_click=_save, args=(item, "same"))
                c2.button("Different", key="review_different", on_click=_save, args=(item, "different"))

            elif item["kind"] == "family":
                shown = item["us_text"] or item["india_text"]
                st.markdown(f"<div class='card'><div class='note'>{html.escape(item['side'].upper())} "
                            f"model text</div><h4>{html.escape(shown)}</h4>"
                            f"<p class='note'>{html.escape(item.get('reason') or '')}</p></div>",
                            unsafe_allow_html=True)
                vocab = model_family.load_vocab(ds.ref_dir() / "model_families.csv").get(item["make"], [])
                options = vocab + ["none"]
                llm_choice = item.get("llm_answer") or "none"
                st.selectbox("Family", options, key="review_family_choice",
                             index=options.index(llm_choice) if llm_choice in options else len(options) - 1)
                st.caption(f"LLM chose: {llm_choice}")
                st.button("Confirm", key="review_confirm", on_click=lambda it=item: _save(
                    it, st.session_state["review_family_choice"]))

            else:  # vocabulary
                st.markdown(f"<div class='card'><h4>{html.escape(item['key'])}</h4>"
                            f"<p class='note'>{html.escape(item.get('reason') or '')}</p></div>",
                            unsafe_allow_html=True)
                c1, _ = st.columns([1, 5])
                c1.button("Distinct models", key="review_distinct", on_click=_save,
                          args=(item, "distinct"))
                with st.expander("Same model: record an alias (needs a source)"):
                    st.text_input("Other spelling", key="review_from_name")
                    st.text_input("Source URL", key="review_source_url")
                    st.text_input("Retrieved on (YYYY-MM-DD)", key="review_retrieved_on")
                    st.button("Save alias", key="review_alias", on_click=lambda it=item: _save(
                        it, "alias", from_name=st.session_state.get("review_from_name"),
                        source_url=st.session_state.get("review_source_url"),
                        retrieved_on=st.session_state.get("review_retrieved_on")))

            s1, s2, _ = st.columns([1, 1, 4])
            if s1.button("Skip", key="review_skip"):
                st.session_state.setdefault("review_skipped", set()).add(item["item_id"])
            if st.session_state.get("review_last") and s2.button("Undo last", key="review_undo"):
                review.undo(st.session_state.pop("review_last"), ds.ref_dir())
```

Skip: apply it before picking `item` — after the `makes` filter add
`todo = todo[~todo["item_id"].isin(st.session_state.get("review_skipped", set()))]`.

`tile`, `evals`, `highlight` are defined inside other tab blocks today; move `tile` and `highlight` to module level (they are pure functions) and read `evals = data("eval_summary")` again inside the Review tab. `highlight` must exist before the Review tab; check its current definition site and move it next to `pct`.

6. Method tab: add one bullet: "**Human review.** Every LLM `same` pair and every flagged family or vocabulary item is confirmed by a person in the Review tab (or `data/clean/review_queue.csv` + `python -m src.review --import`). Only confirmed echoes count in the KPIs."

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dashboard_data.py tests/test_security.py -q` → pass. Full suite. If an AppTest widget key differs from the test, fix the test to use the key in the code above (keys are part of the interface).

- [ ] **Step 5: Commit**

```bash
git add app/data_sources.py app/dashboard.py tests/test_dashboard_data.py
git commit -m "Add dashboard Review tab writing decisions through review.record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Run on the real data, document, explain

**Files:**
- Modify: `CLAUDE.md` (Commands), `notes.md`
- Generated: `data/clean/review_queue.csv`, `matches_reviewed`, KPI CSVs, `reports/eval.md`, `reports/eval_summary.csv`

- [ ] **Step 1: Matcher rerun with fingerprints (expected free)**

```bash
ls data/interim/llm_cache | wc -l
.venv/Scripts/python.exe -m src.matcher --all
ls data/interim/llm_cache | wc -l
```

Expected: label counts unchanged (`same=238, different=1367, needs_review=15`); `pairs sent` > 0 this first time (old rows have no fingerprint) but the cache file count is **unchanged** (same batches → cache hits, no network). If the count grew, stop and report: the prompt text changed in Task 2.

Run again: expected `verdicts reused: 1605, pairs sent to the LLM: 0` (1,620 minus the 15 screened/no-text pairs; report the real numbers).

- [ ] **Step 2: Apply and build the queue**

```bash
.venv/Scripts/python.exe -m src.review
```

Expected: `matches_reviewed: reviewed=0 stale=0 unreviewed=1620 orphaned=0`; queue counts by kind: family 68 open (42 + 26), vocabulary 18, pair 253 (15 + 238). If any differs, investigate before continuing.

- [ ] **Step 3: KPIs and evaluation**

```bash
.venv/Scripts/python.exe -m src.kpis
.venv/Scripts/python.exe -m src.evaluate
```

Expected: overall `echoing = 0` (no confirmations yet) with `n_reviewed` beside n; `reports/eval.md` has the precision section saying nothing reviewed yet. This is the intended starting state.

- [ ] **Step 4: Docs**

`CLAUDE.md` → Commands section, add:

```markdown
- Human review: `.venv/Scripts/python.exe -m src.review` (apply decisions, write data/clean/review_queue.csv); `--import <sheet>`; `--rebuild` after family decisions
```

`notes.md` → add a dated section "2026-10-16 human review" with: queue size by kind, the fact that KPIs now show 0 confirmed echoes until review, and that 238 LLM `same` pairs await confirmation.

- [ ] **Step 5: Full suite, restart dashboard, commit**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Restart the dashboard (stop background task, rerun `./.venv/Scripts/python.exe -m streamlit run app/dashboard.py --server.port 8501`) and check the Review tab loads with 339 items.

```bash
git add CLAUDE.md notes.md data/clean/review_queue.csv data/clean/kpi_*.csv data/clean/echo_explorer.csv reports/eval.md reports/eval_summary.csv
git commit -m "Run human review pipeline: queue built, KPIs count confirmed echoes only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Explain the new modules in 5 plain lines each** (`src/fingerprint.py`, `src/review.py`) to the user, per CLAUDE.md.
