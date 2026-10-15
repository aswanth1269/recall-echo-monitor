"""Map NHTSA trims and SIAM model text onto one shared model-family vocabulary.

The matching unit is one NHTSA campaign vs one SIAM row, compared by `make`.
The two sides name models differently - NHTSA gives trims ("330I",
"GLE 450", "RS 7"), SIAM gives lists and codes ("BMW 1/3 Series, X1/3/5/6",
"GLE (167)") - so both are reduced to a family by ONE set of rules:

  * BMW, Mercedes-Benz and Audi have make-specific rules (330I -> 3 SERIES,
    GLE 450 -> GLE, RS 7 -> A7);
  * every other make takes the longest vocabulary family that prefixes the
    model ("CR-V HYBRID" -> CR-V, "V60CC" -> V60).

The vocabulary is data/raw/india_models.csv plus families SIAM's own model
text names, each kept with its evidence. Cross-market names (Rogue vs
X-Trail) are only linked through data/reference/model_aliases.csv, whose
rows need a source and which starts empty. Anything that does not reduce to
a vocabulary family stays NULL and is written to family_issues.

Run:  .venv/Scripts/python.exe -m src.model_families
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
from rapidfuzz import fuzz

from src.load_nhtsa import DB_PATH, MAKETXT_ALIASES
from src.load_siam import clean_text
from src.mercedes_codes import REF_CSV as CODES_CSV
from src.mercedes_codes import _DASHES, parse_segment, split_segments

ROOT = Path(__file__).resolve().parents[1]
INDIA_MODELS_CSV = ROOT / "data/raw/india_models.csv"
VOCAB_CSV = ROOT / "data/reference/model_families.csv"
ALIASES_CSV = ROOT / "data/reference/model_aliases.csv"
REVIEW_CSV = ROOT / "data/clean/family_review.csv"
SQL_DIR = ROOT / "sql"

ALIAS_COLUMNS = ["make", "from_name", "family", "source_url", "retrieved_on", "note"]
SPELLING_THRESHOLD = 85  # rapidfuzz ratio for flagging possible typos for review

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


def nhtsa_families(models: pd.DataFrame, vocab: dict[str, set[str]], aliases=None
                   ) -> tuple[pd.DataFrame, list[tuple]]:
    """models has CAMPNO, make, MODELTXT. family is NULL when nothing matches."""
    cache: dict[tuple[str, str], tuple] = {}
    families = []
    for r in models.itertuples(index=False):
        k = (r.make, r.MODELTXT)
        if k not in cache:
            cache[k] = reduce_family(r.make, r.MODELTXT, vocab, aliases)
        families.append(cache[k][0])
    out = models.assign(family=families)
    counts = out[out.family.isna()].groupby(["make", "MODELTXT"]).CAMPNO.nunique()
    issues = [("nhtsa", make, model, model, f"{cache[(make, model)][1]} ({n} campaigns)")
              for (make, model), n in counts.items()]
    return out, issues


def review_rows(vocab: pd.DataFrame, nhtsa_models: pd.DataFrame) -> pd.DataFrame:
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


def build(db_path: Path = DB_PATH, india_csv: Path = INDIA_MODELS_CSV,
          codes_csv: Path = CODES_CSV, vocab_csv: Path = VOCAB_CSV,
          aliases_csv: Path = ALIASES_CSV, review_csv: Path = REVIEW_CSV) -> pd.DataFrame:
    """Write the vocabulary, review list and three tables; return the coverage report."""
    india = pd.read_csv(india_csv, dtype=str, keep_default_na=False)
    codes_df = pd.read_csv(codes_csv, dtype=str, keep_default_na=False)
    codes = codes_df.groupby("platform_code").model_name.apply(list).to_dict()
    if not aliases_csv.exists():
        aliases_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=ALIAS_COLUMNS).to_csv(aliases_csv, index=False)
    aliases = load_aliases(aliases_csv)

    con = duckdb.connect(str(db_path))
    try:
        siam = con.execute("SELECT csv_row, make, model, source_url FROM siam_recalls "
                           "ORDER BY csv_row").df()
        models = con.execute("SELECT DISTINCT CAMPNO, make, MODELTXT FROM nhtsa_focus "
                             "WHERE MODELTXT IS NOT NULL ORDER BY CAMPNO, MODELTXT").df()

        vocab_df, vocab_issues = build_vocabulary(india, siam, codes)
        vocab = vocab_df.groupby("make").family.apply(set).to_dict()
        siam_rows, siam_issues = siam_row_families(siam, vocab, codes, aliases)
        nhtsa_rows, nhtsa_issues = nhtsa_families(models, vocab, aliases)
        issues = pd.DataFrame(vocab_issues + siam_issues + nhtsa_issues,
                              columns=["side", "make", "key", "raw_text", "reason"])

        for path, df in ((vocab_csv, vocab_df), (review_csv, review_rows(vocab_df, models))):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8")

        for name, df in (("nhtsa_model_families", nhtsa_rows),
                         ("siam_row_families", siam_rows), ("family_issues", issues)):
            con.register("df", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
            con.unregister("df")
        return con.execute((SQL_DIR / "family_coverage.sql").read_text(encoding="utf-8")).df()
    finally:
        con.close()


def main() -> None:
    coverage = build()
    pd.set_option("display.width", 200)
    print("Family coverage per make (rates suppressed where n < 10):")
    print(coverage.to_string(index=False))
    print(f"\nwrote {VOCAB_CSV.relative_to(ROOT)} and {REVIEW_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
