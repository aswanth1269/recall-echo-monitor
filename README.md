# recall-echo-monitor

Does a safety recall in the US "echo" into India for the same carmaker and the
same model? This project puts NHTSA's US recall file next to SIAM's Indian
voluntary-recall list for 10 carmakers that sell in both markets, and looks at
what gets recalled in one market but not the other.

## Status

Day 0 setup is done: environment, both datasets, and the India reference
tables. No analysis yet.

## Setup

Python 3.11+ required (built and verified on 3.13.7).

```bash
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

Then add an LLM key for the Day 3 enrichment step:

```bash
cp .env.example .env     # then edit .env and set LLM_API_KEY
.venv/Scripts/python.exe -m src.llm --smoke   # verify it works
```

`.env` is gitignored; `.env.example` is the tracked template. Never commit a
real key.

### The LLM step runs on a free tier

[src/llm.py](src/llm.py) talks to any **OpenAI-compatible** provider, so the
provider is three environment variables rather than a code change. The
default is Google AI Studio, which needs no credit card:

1. Get a key at <https://aistudio.google.com/apikey>.
2. Put it in `.env` as `LLM_API_KEY`.
3. Run the smoke check above.

**Use a Flash-Lite model, not Flash.** As of late 2026 the Gemini Flash
models are capped near 20 requests/day on the free tier while Flash-Lite is
around 500/day — a 25x difference for this workload. Google no longer
publishes these numbers, so confirm your own at
<https://aistudio.google.com/rate-limit>. `.env.example` also carries ready
blocks for Groq, Mistral and OpenRouter.

Two habits the free tier forces, both already handled:

- **Every response is cached on disk** (`data/interim/llm_cache/`, keyed by
  model + prompt), so re-running a script costs zero requests and a crash
  mid-way never loses completed work.
- **Batch your records.** The binding limit is requests/day, not tokens, so
  use `batched()` to send many recalls per call — 355 rows at `size=20` is
  18 requests, not 355.

```python
from src.llm import complete, complete_json, batched

for chunk in batched(recalls, 20):
    rows = complete_json(f"Classify each defect...\n{chunk}")
```

Anthropic has no free tier, so `ANTHROPIC_API_KEY` is commented out in
`.env.example`; the `anthropic` package stays installed if you later switch.

## Layout

```
data/raw/          source data, as retrieved (see docs/SOURCES.md)
data/interim/      intermediate artifacts        (gitignored)
data/clean/        recalls.duckdb (gitignored) + cleaned outputs
src/               pipeline code
tests/             pytest
docs/SOURCES.md    provenance, shapes, and the data gotchas
```

Large and re-downloadable raw files (the NHTSA zip, the SIAM HTML snapshot)
are gitignored. The small curated CSVs are tracked.

## Data

| file | rows | what it is |
|---|---:|---|
| `data/raw/FLAT_RCL_POST_2010.zip` | 245,855 | NHTSA US recall records, 29 tab-delimited fields |
| `data/raw/RCL.txt` | — | NHTSA field layout |
| `data/raw/siam_recalls.csv` | 355 | SIAM India recalls, the 10 picked makes |
| `data/raw/siam_recalls_all.csv` | 546 | SIAM India recalls, every filer |
| `data/raw/india_makes.csv` | 10 | the picked makes + their join keys |
| `data/raw/india_models.csv` | 96 | what each make sells in India today |

The 10 makes — Toyota, Honda, Hyundai, Kia, Nissan, Volkswagen (mass-market)
and Mercedes-Benz, BMW, Audi, Volvo (luxury) — all sell cars in both markets
today and all appear in the SIAM data.

Rebuild the SIAM CSVs from the saved HTML snapshot:

```bash
.venv/Scripts/python.exe src/parse_siam.py            # from snapshot
.venv/Scripts/python.exe src/parse_siam.py --refresh   # re-download first
```

## Read this before writing matching code

`docs/SOURCES.md` has the full list, but the two that will cost you the most
time:

- **SIAM files recalls under the legal entity, not the brand.** One entity
  covers Volkswagen, Audi, Škoda, Porsche and Lamborghini. Grouping by
  manufacturer name hides all 35 Audi recalls.
- **Mercedes puts platform codes in the model field** (`205 (C-Class)`,
  `167 (GLE, GLS)`), not marketing names — and Mercedes is 149 of the 355
  India-side rows. Fuzzy matching on model text alone will miss them.

## Caveat on the India model lists

51 of the 96 model rows were read off official India sites. The other 45 —
all of Mercedes-Benz, BMW, Audi and Volvo — came from CarWale / Autocar India,
because those four official sites block automated fetches. Those rows are
marked `secondary_source` and should be verified by hand before being relied
on.
