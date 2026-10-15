"""Give every NHTSA model and every SIAM row a model family: rules, then LLM, then overrides.

Families come from data/reference/model_families.csv: india_models.csv plus
families SIAM's own model text names, each with its evidence. Generate or
refresh that vocabulary with `--vocab`; a normal run only reads it, so edits
made during review are never overwritten.

1. Rules. BMW, Mercedes and Audi have trim rules (330I -> 3 SERIES,
   GLE 450 -> GLE, RS 7 -> A7); other makes take the longest vocabulary
   family the model starts with (CR-V HYBRID -> CR-V). SIAM text is split
   after expanding shorthand ("X1/3/5/6", "2 & 5 Series") and Mercedes
   platform codes. method = 'rule'.
2. LLM, NHTSA side only. Distinct (make, MODELTXT) the rules cannot map go
   through src/llm.py, 40 per call, one make per call, with that make's
   allowed list. Only an allowed family or "none" is accepted; anything else
   is stored as NULL with a note. method = 'llm', unconfirmed until checked
   in data/clean/model_family_review.csv.
3. Overrides. data/reference/model_family_overrides.csv, if present, is
   applied last. method = 'override'.

Cross-market names (Rogue vs X-Trail) can also be linked by rule through
data/reference/model_aliases.csv, whose rows must carry a source.

Writes tables nhtsa_model_family (make, MODELTXT, family, method) and
siam_row_family (csv_row, make, family, method).

Run:  .venv/Scripts/python.exe -m src.model_family           # map
      .venv/Scripts/python.exe -m src.model_family --vocab   # (re)build vocabulary
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
from rapidfuzz import fuzz

from src.llm import batched, complete_json
from src.load_nhtsa import DB_PATH, MAKETXT_ALIASES
from src.load_siam import clean_text
from src.mercedes_codes import REF_CSV as CODES_CSV
from src.mercedes_codes import _DASHES, parse_segment, split_segments

ROOT = Path(__file__).resolve().parents[1]
INDIA_MODELS_CSV = ROOT / "data/raw/india_models.csv"
VOCAB_CSV = ROOT / "data/reference/model_families.csv"
ALIASES_CSV = ROOT / "data/reference/model_aliases.csv"
OVERRIDES_CSV = ROOT / "data/reference/model_family_overrides.csv"
REVIEW_CSV = ROOT / "data/clean/model_family_review.csv"
VOCAB_REVIEW_CSV = ROOT / "data/clean/vocabulary_review.csv"
SQL_DIR = ROOT / "sql"

ALIAS_COLUMNS = ["make", "from_name", "family", "source_url", "retrieved_on", "note"]
OVERRIDE_COLUMNS = ["side", "make", "key", "family", "note"]
SPELLING_THRESHOLD = 85  # rapidfuzz ratio for flagging possible typos for review
LLM_BATCH = 40
NONE = "none"
# Tables written by the earlier two-module version; dropped on every build.
OLD_TABLES = ("nhtsa_model_families", "siam_row_families", "family_issues")

log = logging.getLogger(__name__)

_MAKE_WORDS = re.compile(
    r"^(?:MERCEDES-BENZ|MERCEDES BENZ|MERCEDES-AMG|MERCEDES|TOYOTA|BMW|VOLKSWAGEN|"
    r"AUDI|HYUNDAI|HONDA|KIA|NISSAN|VOLVO)\b[\s,]*")
_QUALIFIER = re.compile(r"\s*\((?![^)]*\d)[^)]*\)")  # "(NON ABS)", not "(205)"
_MB_CLASS = re.compile(
    r"^(GLA|GLB|GLC|GLE|GLS|GLK|CLA|CLE|CLK|CLS|SLK|SLC|SLR|SL|EQA|EQB|EQC|EQE|EQS|"
    r"ML|GL|G|A|B|C|E|S|V|R)(?=[\s\-]*(?:\d|CLASS\b)|\s|$)")


# A trailing powertrain / body word does not make a new family: SIAM's
# "XC90 Hybrid" and India's "Creta Electric" belong to XC90 and CRETA, the
# same way the BMW and Audi rules collapse trims. Applied to vocabulary names
# of prefix-matched makes only.
_VARIANT_SUFFIX = re.compile(
    r"(?:\s+(?:PLUG-IN HYBRID|HYBRID|PHEV|EV|ELECTRIC|BEV|CROSS COUNTRY|CC|N LINE))+$")


def family_name(make: str, name: str) -> str:
    """Vocabulary entry for a name: rule makes untouched, others lose variant suffixes."""
    if make in _RULES:
        return name
    return _VARIANT_SUFFIX.sub("", name).strip() or name


def _key(name: str) -> str:
    """Comparison key: spaces, hyphens and dots removed ('IONIQ 5' == 'IONIQ5')."""
    return re.sub(r"[\s\-.]", "", name.upper())


def normalise_model(text: object) -> str | None:
    """Uppercase, unify dashes, drop make words / '4 Wheeler' / non-code brackets."""
    s = clean_text(text)
    if s is None:
        return None
    s = _QUALIFIER.sub("", s.translate(_DASHES).upper())
    s = " ".join(re.sub(r"\b4\s*WHEELERS?\b", " ", s).split())
    previous = None
    while previous != s:
        previous, s = s, _MAKE_WORDS.sub("", s).strip()
    s = s.strip(" -,.")
    return s or None


def expand_siam_shorthand(text: str) -> str:
    """'X1/3/5/6/' -> 'X1, X3, X5, X6'; '1/3 Series', '2 & 5 Series' -> one per digit."""
    def series(m: re.Match) -> str:
        return ", ".join(f"{d} Series" for d in re.findall(r"\d", m.group(1)))

    def letters(m: re.Match) -> str:
        digits = [m.group(2), *re.findall(r"\d", m.group(3))]
        return ", ".join(m.group(1) + d for d in digits)

    text = re.sub(r"\b(\d(?:\s*(?:/|&|,|\band\b)\s*\d)+)\s*series", series, text, flags=re.I)
    return re.sub(r"\b([XxIi])(\d)((?:\s*/\s*\d)+)/?", letters, text)


def _bmw(s: str) -> str | None:
    if m := re.match(r"^(\d)\s*SERIES\b", s):
        return f"{m[1]} SERIES"
    if m := re.match(r"^M?(\d)\d{2}(?!\d)", s):          # 330I, 740LI, M340I
        return f"{m[1]} SERIES"
    if m := re.match(r"^M(\d)(?!\d)", s):                # M3, M5, M8
        return f"{m[1]} SERIES"
    if m := re.match(r"^(IX\d?|I\d|X\d|Z\d)(?!\d)", s):  # X5, X3M, IX1, I4, Z4
        return m[1]
    return None


def _mercedes(s: str) -> str | None:
    s = re.sub(r"^(?:MERCEDES-AMG|AMG|MAYBACH)[\s\-]*", "", s)
    s = re.sub(r"^(?:AMG|MAYBACH)[\s\-]*", "", s)
    for van in ("SPRINTER", "METRIS", "VIANO"):
        if s.startswith(van):
            return van
    if m := re.match(r"^(EQE|EQS)[\s\-]*SUV\b", s):
        return f"{m[1]} SUV"
    if s.startswith("GT"):
        return "AMG GT"
    if m := _MB_CLASS.match(s):
        return f"{m[1]}-CLASS" if len(m[1]) == 1 else m[1]
    return None


def _audi(s: str) -> str | None:
    s = re.sub(r"\bE\s*-?\s*TRON\b", "E-TRON", s)   # ETRON, E TRON, E-TRON
    if "E-TRON GT" in s:
        return "E-TRON GT"
    if "E-TRON" in s:
        return "E-TRON"
    if m := re.match(r"^(?:RS\s?|S)?([AQ]\d|TT|R8)(?!\d)", s):  # RS Q8, SQ5, A4 ALLROAD
        return m[1]
    if m := re.match(r"^(?:RS\s?|S)(\d)(?!\d)", s):              # S6, RS 7 -> A6, A7
        return f"A{m[1]}"
    return None


_RULES: dict[str, Callable[[str], str | None]] = {
    "BMW": _bmw, "MERCEDES-BENZ": _mercedes, "AUDI": _audi,
}


def _longest_prefix(s: str, families) -> str | None:
    k, best = _key(s), None
    for fam in families:
        fk = _key(fam)
        if fk and k.startswith(fk) and (best is None or len(fk) > len(_key(best))):
            best = fam
    return best


def reduce_family(make: str, text: object, vocab: dict[str, set[str]] | None = None,
                  aliases: dict[tuple[str, str], str] | None = None
                  ) -> tuple[str | None, str | None]:
    """(family, None) or (None, reason). vocab=None means 'building the vocabulary'."""
    s = normalise_model(text)
    if s is None:
        return None, "no model text left after removing make / filler words"
    if aliases and (fam := aliases.get((make, _key(s)))):
        return fam, None
    rule = _RULES.get(make)
    if rule is None:
        if vocab is None:
            return s, None
        fam = _longest_prefix(s, vocab.get(make, ()))
        return (fam, None) if fam else (None, "matches no family in the vocabulary")
    fam = rule(s)
    if fam is None:
        return None, "no family rule matches"
    if vocab is not None and fam not in vocab.get(make, ()):
        return None, f"reduces to {fam}, which is not in the India vocabulary"
    return fam, None


def siam_segments(make: str, model_text: str | None,
                  codes: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """(segment, name, how) per piece of a SIAM model cell; Mercedes codes expanded."""
    out = []
    for seg in split_segments(expand_siam_shorthand(model_text or "")):
        if make == "MERCEDES-BENZ":
            if pairs := parse_segment(seg):
                out += [(seg, name, "platform_code") for _, name in pairs]
                continue
            m = re.fullmatch(r"(?:model\s*)?(\d{3})", seg.strip(), re.I)
            if m and codes.get(m[1]):
                out += [(seg, name, "platform_code") for name in codes[m[1]]]
                continue
        out.append((seg, seg, "vocab"))
    return out


def _usable(family: str) -> bool:
    return bool(re.search(r"[A-Z]", family)) and len(_key(family)) >= 2


def build_vocabulary(india: pd.DataFrame, siam: pd.DataFrame,
                     codes: dict[str, list[str]]) -> tuple[pd.DataFrame, list[tuple]]:
    """Families from india_models.csv plus families SIAM model text names."""
    entries: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"sources": set(), "india": [], "siam_rows": set(), "urls": []})
    issues = []
    # One entry per comparison key: SIAM's "IONIQ5" joins India's "IONIQ 5".
    # India rows are read first, so their spelling is the one kept.
    canonical: dict[tuple[str, str], str] = {}
    for r in india.itertuples(index=False):
        make = MAKETXT_ALIASES[r.make][0]
        fam, reason = reduce_family(make, r.model)
        fam = family_name(make, fam) if fam else fam
        fam = canonical.setdefault((make, _key(fam)), fam) if fam else fam
        if fam and _usable(fam):
            e = entries[(make, fam)]
            e["sources"].add("india_models"); e["india"].append(r.model)
            if r.source_url not in e["urls"]:
                e["urls"].append(r.source_url)
        else:
            issues.append(("india_models", make, r.model, r.model, reason or "unusable name"))
    for r in siam.itertuples(index=False):
        for _seg, name, _how in siam_segments(r.make, r.model, codes):
            fam, _ = reduce_family(r.make, name)
            fam = family_name(r.make, fam) if fam else fam
            fam = canonical.setdefault((r.make, _key(fam)), fam) if fam else fam
            if fam and _usable(fam):
                e = entries[(r.make, fam)]
                e["sources"].add("siam_text"); e["siam_rows"].add(int(r.csv_row))
                if r.source_url not in e["urls"]:
                    e["urls"].append(r.source_url)
    rows = []
    for (make, fam), e in sorted(entries.items()):
        evidence = []
        if e["india"]:
            evidence.append("india_models: " + "; ".join(sorted(set(e["india"]))))
        if e["siam_rows"]:
            evidence.append("siam csv_row: " + ";".join(map(str, sorted(e["siam_rows"]))))
        rows.append((make, fam, ";".join(sorted(e["sources"])), " | ".join(evidence),
                     " ".join(e["urls"])))
    vocab = pd.DataFrame(rows, columns=["make", "family", "source", "evidence", "source_url"])
    return vocab, issues


def load_aliases(path: Path = ALIASES_CSV) -> dict[tuple[str, str], str]:
    """Sourced cross-market names. A row without a source_url is refused."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    unsourced = df[df["source_url"].str.strip() == ""]
    if len(unsourced):
        raise ValueError(f"model_aliases.csv rows without source_url: {unsourced['from_name'].tolist()}")
    return {(r.make, _key(r.from_name)): r.family.upper() for r in df.itertuples(index=False)}


def siam_row_families(siam: pd.DataFrame, vocab: dict[str, set[str]],
                      codes: dict[str, list[str]], aliases=None
                      ) -> tuple[pd.DataFrame, list[tuple]]:
    rows, issues, seen = [], [], set()
    for r in siam.itertuples(index=False):
        segments = siam_segments(r.make, r.model, codes)
        if not segments:
            issues.append(("siam", r.make, str(r.csv_row), r.model, "blank model text"))
        for seg, name, how in segments:
            fam, reason = reduce_family(r.make, name, vocab, aliases)
            if fam is None:
                shown = seg if name == seg else f"{seg} -> {name}"  # name a code expanded to
                issues.append(("siam", r.make, str(r.csv_row), shown, reason))
            elif (r.csv_row, fam) not in seen:
                seen.add((r.csv_row, fam))
                rows.append((int(r.csv_row), r.make, seg, fam, how))
    return (pd.DataFrame(rows, columns=["csv_row", "make", "segment", "family", "how"]),
            issues)


def vocabulary_review(vocab: pd.DataFrame, nhtsa_models: pd.DataFrame) -> pd.DataFrame:
    """Things a person should look at once: name collisions and possible typos."""
    rows = []
    for make, g in vocab.groupby("make"):
        if make in _RULES:  # explicit rules decide these makes, not prefixes
            continue
        fams = list(g.family)
        for a in fams:
            for b in fams:
                if a != b and _key(b).startswith(_key(a)):
                    rows.append((make, b, f"'{a}' is a prefix of '{b}': confirm they are "
                                          "different models (longest match wins)"))
        us = {_key(m): m for m in nhtsa_models.loc[nhtsa_models.make == make, "MODELTXT"]}
        for r in g[g.source == "siam_text"].itertuples(index=False):
            fk = _key(r.family)
            if any(k.startswith(fk) for k in us):
                continue
            best = max(us, key=lambda k: fuzz.ratio(fk, k), default=None)
            if best and fuzz.ratio(fk, best) >= SPELLING_THRESHOLD:
                rows.append((make, r.family, f"close to NHTSA '{us[best]}' "
                                             f"(ratio {fuzz.ratio(fk, best):.0f}); "
                                             "possible spelling variant - not merged"))
    return pd.DataFrame(rows, columns=["make", "family", "review_note"])


SYSTEM = (
    "You map US (NHTSA) vehicle model names to the India model family that is the "
    "same vehicle line. You only choose from the list you are given. If a US model is "
    "not sold in India under any listed family, or is not a passenger car (motorcycle, "
    "truck, bus, trailer, accessory), or you are not sure, answer \"none\". Never "
    "invent a family name."
)


def load_vocab(path: Path = VOCAB_CSV) -> dict[str, list[str]]:
    """make -> sorted allowed families, from the reviewed vocabulary file."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {make: sorted(g.family) for make, g in df.groupby("make")}


def rule_map_nhtsa(models: pd.DataFrame, vocab: dict[str, list[str]],
                   aliases: dict | None = None) -> pd.DataFrame:
    """models has make, MODELTXT (distinct). Adds family and method ('rule' or None)."""
    sets = {m: set(f) for m, f in vocab.items()}
    fams = [reduce_family(r.make, r.MODELTXT, sets, aliases)[0]
            for r in models.itertuples(index=False)]
    out = models[["make", "MODELTXT"]].copy()
    out["family"] = fams
    out["method"] = ["rule" if f else None for f in fams]
    return out


def build_prompt(make: str, models: list[str], allowed: list[str]) -> str:
    """One batch prompt: a make, its allowed families, and up to LLM_BATCH US names."""
    lines = "\n".join(f"- {m}" for m in models)
    return (
        f"Make: {make}\n"
        f"Allowed India families for {make} (choose exactly one of these, or \"none\"):\n"
        f"{', '.join(allowed)}\n\n"
        f"US model names:\n{lines}\n\n"
        "Return ONLY a JSON object mapping every US model name above, spelled exactly as "
        "given, to one allowed family or \"none\". Example: "
        "{\"US NAME 1\": \"FAMILY\", \"US NAME 2\": \"none\"}"
    )


def parse_llm_answer(answer: object, models: list[str], allowed: list[str]
                     ) -> list[tuple[str, str | None, str]]:
    """(MODELTXT, family or None, note) for every model; invalid answers become None."""
    by_upper = {a.upper(): a for a in allowed}
    answer = answer if isinstance(answer, dict) else {}
    out = []
    for m in models:
        raw = answer.get(m)
        if raw is None:
            out.append((m, None, "llm gave no answer for this model"))
        elif str(raw).strip().lower() == NONE:
            out.append((m, None, "llm: not sold in India / not a listed family"))
        elif str(raw).strip().upper() in by_upper:
            out.append((m, by_upper[str(raw).strip().upper()], "llm choice from allowed list"))
        else:
            out.append((m, None, f"llm answer {raw!r} is not in the allowed list - rejected"))
    return out


def llm_map(unmapped: pd.DataFrame, vocab: dict[str, list[str]],
            ask: Callable[..., object] = complete_json,
            batch_size: int = LLM_BATCH) -> pd.DataFrame:
    """Ask the LLM, one make per call and batch_size names per call."""
    rows = []
    for make, g in unmapped.groupby("make", sort=True):
        allowed = vocab.get(make, [])
        names = sorted(g.MODELTXT.unique())
        if not allowed:
            rows += [(make, n, None, "llm", "make has no allowed families") for n in names]
            continue
        for chunk in batched(names, batch_size):
            answer = ask(build_prompt(make, chunk, allowed), system=SYSTEM, max_tokens=4096)
            rows += [(make, m, fam, "llm", note)
                     for m, fam, note in parse_llm_answer(answer, chunk, allowed)]
    return pd.DataFrame(rows, columns=["make", "MODELTXT", "family", "method", "note"])


def siam_families(siam: pd.DataFrame, vocab: dict[str, list[str]],
                  codes: dict[str, list[str]], aliases: dict | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per family a SIAM row mentions, SIAM rows with zero families)."""
    sets = {m: set(f) for m, f in vocab.items()}
    rows, _issues = siam_row_families(siam, sets, codes, aliases)
    fam = rows[["csv_row", "make", "family"]].drop_duplicates().assign(method="rule")
    zero = siam[~siam.csv_row.isin(fam.csv_row)][["csv_row", "make", "model"]]
    return fam.reset_index(drop=True), zero.reset_index(drop=True)


def load_overrides(path: Path = OVERRIDES_CSV) -> pd.DataFrame:
    """Override rows (side, make, key, family, note), or an empty frame if no file."""
    if not path.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(OVERRIDE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    bad = df[~df.side.isin(["nhtsa", "siam"])]
    if len(bad):
        raise ValueError(f"{path.name}: side must be 'nhtsa' or 'siam', got {bad.side.tolist()}")
    return df


def apply_overrides(nhtsa: pd.DataFrame, siam: pd.DataFrame, overrides: pd.DataFrame,
                    vocab: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply overrides last. Blank or 'none' family means 'no family'."""
    def target(r) -> str | None:
        fam = r.family.strip()
        if fam == "" or fam.lower() == NONE:
            return None
        if fam.upper() not in {f.upper() for f in vocab.get(r.make, [])}:
            raise ValueError(f"override family {fam!r} for {r.make} is not in the vocabulary")
        return next(f for f in vocab[r.make] if f.upper() == fam.upper())

    nhtsa, siam = nhtsa.copy(), siam.copy()
    for r in overrides[overrides.side == "nhtsa"].itertuples(index=False):
        hit = (nhtsa.make == r.make) & (nhtsa.MODELTXT == r.key)
        if not hit.any():
            log.warning("override for %s %r matches no NHTSA model", r.make, r.key)
            continue
        nhtsa.loc[hit, ["family", "method"]] = [target(r), "override"]

    siam_rows = overrides[overrides.side == "siam"]
    for key, group in siam_rows.groupby("key"):
        row = int(key)
        make = group.make.iloc[0]
        siam = siam[siam.csv_row != row]
        new = [(row, make, fam, "override") for fam in map(target, group.itertuples(index=False))
               if fam is not None]
        siam = pd.concat([siam, pd.DataFrame(new, columns=siam.columns)], ignore_index=True)
    return nhtsa, siam


def review_frame(llm_rows: pd.DataFrame, zero_rows: pd.DataFrame) -> pd.DataFrame:
    """Every LLM-mapped NHTSA value and every SIAM row with zero families, to confirm."""
    a = pd.DataFrame({"side": "nhtsa", "make": llm_rows.make, "key": llm_rows.MODELTXT,
                      "text": llm_rows.MODELTXT, "family": llm_rows.family,
                      "method": "llm", "note": llm_rows.note})
    b = pd.DataFrame({"side": "siam", "make": zero_rows.make, "key": zero_rows.csv_row.astype(str),
                      "text": zero_rows.model, "family": None, "method": "rule",
                      "note": "SIAM row produced zero families"})
    out = pd.concat([a, b], ignore_index=True)
    out["confirmed"] = ""
    return out


def sql_statements(path: Path) -> list[str]:
    """Statements in a .sql file. Comment lines are dropped first, so a ';' inside a
    comment cannot split a query."""
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("--")]
    return [q.strip() for q in "\n".join(lines).split(";") if q.strip()]


def build(db_path: Path = DB_PATH, vocab_csv: Path = VOCAB_CSV,
          codes_csv: Path = CODES_CSV, aliases_csv: Path = ALIASES_CSV,
          overrides_csv: Path = OVERRIDES_CSV, review_csv: Path = REVIEW_CSV,
          ask: Callable[..., object] = complete_json) -> dict[str, pd.DataFrame]:
    """Write nhtsa_model_family, siam_row_family and the review CSV; return summaries."""
    if not vocab_csv.exists():
        raise FileNotFoundError(f"{vocab_csv} is missing - run "
                                "`python -m src.model_family --vocab` first, then review it")
    vocab = load_vocab(vocab_csv)
    codes_df = pd.read_csv(codes_csv, dtype=str, keep_default_na=False)
    codes = codes_df.groupby("platform_code").model_name.apply(list).to_dict()
    aliases = load_aliases(aliases_csv)

    con = duckdb.connect(str(db_path))
    try:
        models = con.execute("SELECT DISTINCT make, MODELTXT FROM nhtsa_focus "
                             "WHERE MODELTXT IS NOT NULL ORDER BY make, MODELTXT").df()
        siam = con.execute("SELECT csv_row, make, model, source_url FROM siam_recalls "
                           "ORDER BY csv_row").df()

        ruled = rule_map_nhtsa(models, vocab, aliases)
        llm_rows = llm_map(ruled[ruled.method.isna()], vocab, ask)
        nhtsa = pd.concat([ruled[ruled.method == "rule"],
                           llm_rows[["make", "MODELTXT", "family", "method"]]],
                          ignore_index=True)
        siam_fam, zero = siam_families(siam, vocab, codes, aliases)
        nhtsa, siam_fam = apply_overrides(nhtsa, siam_fam, load_overrides(overrides_csv), vocab)

        review_csv.parent.mkdir(parents=True, exist_ok=True)
        review_frame(llm_rows, zero).to_csv(review_csv, index=False, encoding="utf-8")

        for name, df in (("nhtsa_model_family", nhtsa), ("siam_row_family", siam_fam)):
            con.register("df", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
            con.unregister("df")
        for old in OLD_TABLES:
            con.execute(f"DROP TABLE IF EXISTS {old}")
        by_method, share = (con.execute(q).df()
                            for q in sql_statements(SQL_DIR / "model_family_report.sql"))
        return {"by_method": by_method, "share": share}
    finally:
        con.close()


def write_vocabulary(db_path: Path = DB_PATH, india_csv: Path = INDIA_MODELS_CSV,
                     codes_csv: Path = CODES_CSV, vocab_csv: Path = VOCAB_CSV,
                     review_csv: Path = VOCAB_REVIEW_CSV,
                     aliases_csv: Path = ALIASES_CSV) -> tuple[int, int]:
    """(Re)build the vocabulary file and its review list; return (families, review rows).

    Overwrites data/reference/model_families.csv, so run it deliberately and
    re-review the result. Also creates an empty, headed model_aliases.csv.
    """
    india = pd.read_csv(india_csv, dtype=str, keep_default_na=False)
    codes_df = pd.read_csv(codes_csv, dtype=str, keep_default_na=False)
    codes = codes_df.groupby("platform_code").model_name.apply(list).to_dict()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        siam = con.execute("SELECT csv_row, make, model, source_url FROM siam_recalls "
                           "ORDER BY csv_row").df()
        models = con.execute("SELECT DISTINCT make, MODELTXT FROM nhtsa_focus "
                             "WHERE MODELTXT IS NOT NULL").df()
    finally:
        con.close()
    vocab, _issues = build_vocabulary(india, siam, codes)
    review = vocabulary_review(vocab, models)
    for path, df in ((vocab_csv, vocab), (review_csv, review)):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    if not aliases_csv.exists():
        pd.DataFrame(columns=ALIAS_COLUMNS).to_csv(aliases_csv, index=False)
    return len(vocab), len(review)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if "--vocab" in (sys.argv[1:] if argv is None else argv):
        n_fam, n_review = write_vocabulary()
        print(f"wrote {n_fam} families to {VOCAB_CSV.relative_to(ROOT)} and "
              f"{n_review} items to {VOCAB_REVIEW_CSV.relative_to(ROOT)} - review both")
        return
    out = build()
    pd.set_option("display.width", 200)
    print("Mappings by method:")
    print(out["by_method"].to_string(index=False))
    print("\nShare of nhtsa_focus records with a family (rates suppressed where n < 10):")
    print(out["share"].to_string(index=False))
    print(f"\nreview file: {REVIEW_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
