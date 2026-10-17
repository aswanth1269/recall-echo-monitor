# Recall Echo Monitor - main numbers

Run of 2026-10-15. Sources: NHTSA flat file (retrieved 2026-10-13), SIAM voluntary-recall
page (retrieved 2026-10-13), NHTSA recalls API (OTA flag, fetched 2026-10-15). Every rate
shows its n; rates with n < 10 are suppressed (CLAUDE.md). Full tables:
`data/clean/kpi_*.csv`.

**How to read this.** A US campaign "echoes" when the LLM matcher found at least one SIAM
row with the same defect on the same model family. **A missing echo is a transparency
signal, not proof that no India recall happened or that anyone did anything wrong**: SIAM
is a voluntary list, and the matcher can miss.

## Scope

| | count |
|---|---:|
| India-relevant campaign × make rows (unit for every KPI) | 1,383 |
| distinct NHTSA campaigns (20 cover both Audi and VW) | 1,363 |
| candidate pairs judged by the LLM matcher | 1,620 |
| matcher verdicts: same / different / needs_review | 238 / 1,367 / 15 |

## Echo rate (`kpi_echo_rate.csv`)

**Overall: 152 of 1,383 = 11.0%.**

| make | echoing / n | echo rate |
|---|---:|---:|
| HONDA | 18 / 87 | 20.7% |
| TOYOTA | 15 / 84 | 17.9% |
| MERCEDES-BENZ | 69 / 462 | 14.9% |
| AUDI | 16 / 163 | 9.8% |
| HYUNDAI | 7 / 86 | 8.1% |
| BMW | 18 / 225 | 8.0% |
| VOLVO | 4 / 55 | 7.3% |
| KIA | 2 / 69 | 2.9% |
| VOLKSWAGEN | 3 / 107 | 2.8% |
| NISSAN | 0 / 45 | 0.0% |

| segment | echoing / n | echo rate |
|---|---:|---:|
| luxury | 107 / 905 | 11.8% |
| mass_market | 45 / 478 | 9.4% |

**Hand check:** recounting HONDA from the raw `matches` rows in plain Python gives n = 87,
echoing = 18, the same as the SQL.

**Nissan's 0 is not meaningful yet.** Every Nissan campaign is India-relevant only through
unconfirmed LLM family mappings (e.g. SENTRA -> SUNNY, which looks wrong).

**`needs_review` is not counted as an echo.** It affects 12 campaigns that have no other
`same` pair, 9 of them Toyota. Most come from SIAM rows 300 and 311, which have no defect
text, so the LLM never saw them.

## Echo lag (`kpi_echo_lag.csv`): US report date to earliest matched India recall

These are percentiles, not rates, so they are shown at every n.

| group | n | median days | p25 | p75 | India first |
|---|---:|---:|---:|---:|---:|
| all | 152 | 93.5 | 31 | 210 | 20 |
| luxury | 107 | 131 | 42 | 236 | 11 |
| mass_market | 45 | 51 | 0 | 145 | 9 |
| MERCEDES-BENZ | 69 | 151 | 60 | 271 | 7 |
| HONDA | 18 | 27 | -54 | 98 | 6 |
| BMW | 18 | 196 | 15 | 287 | 1 |
| VOLVO | 4 | 0 | -1 | 0 | 1 |

**Volvo files in both markets at once.** Its 4 echoes were reported 0, 0, -4 and +1 days
apart, and the texts match (latch weakened by heat, ASDM software, seat cable, fuel-pump
fuse).

## Echo rate by component group (`kpi_echo_by_component.csv`, groups with n >= 10)

| highest | echoing / n | rate |
|---|---:|---:|
| ELECTRICAL SYSTEM | 40 / 196 | 20.4% |
| HYBRID PROPULSION SYSTEM | 2 / 10 | 20.0% |
| AIR BAGS | 40 / 264 | 15.2% |

| lowest | echoing / n | rate |
|---|---:|---:|
| POWER TRAIN | 1 / 61 | 1.6% |
| EQUIPMENT | 1 / 59 | 1.7% |
| BACK OVER PREVENTION | 1 / 43 | 2.3% |

US-only regulatory items (FMVSS labels, back-over cameras, equipment) rarely echo, which is
expected rather than suspicious.

## Over-the-air remedies (`kpi_ota.csv`)

- **OTA share: 13 of 1,379 campaigns with a known flag = 0.9%.**
  - 4 campaigns came back NULL: the API returned results but none had a true/false flag. They are never guessed.
- **By make,** only Volvo (3 / 55 = 5.5%), Hyundai (2 / 86), Mercedes (7 / 462) and Kia (1 / 69) have any OTA campaigns.
- **By year,** the first OTA campaign is in 2019, and 2021 has the most (4 / 141).
- **Echo rate:** OTA 3 / 13 = 23.1% vs non-OTA 149 / 1,366 = 10.9%.
  - n = 13 is small, so treat the difference as a lead to follow, not a finding.
- **The API flag is a lower bound.** All 13 flagged campaigns mention OTA in their remedy text, but 10 more remedies say OTA while the API flag is `false` (e.g. 26V341000).
  - Going by the remedy text, there would be about 23 OTA campaigns.

## Caveats

- **The matcher is only checked on 30 pairs.** Those pairs were labelled by Claude, not by a person, and both the prompt and one label were adjusted after seeing those pairs (`reports/eval.md`). 30/30 agreement there is not an accuracy figure.
- **LLM family mappings are unconfirmed.** 42 NHTSA model names were mapped to India families by the LLM (`data/clean/model_family_review.csv`); some look wrong (HR-V -> WR-V, SENTRA -> SUNNY).
- **Echo is per make.** A campaign covering Audi and VW is counted once under each.

## 2026-10-17 human review

- **KPIs now count only echoes a person confirmed** (`matches_reviewed.final_label = 'same'`). Before any review the headline is 0 of 1,383 campaigns; it grows as pairs are confirmed.
  - The earlier figure of 152 / 1,383 (11.0%) is the LLM's unconfirmed count. All 152 campaigns are in the review queue.
- **Review queue: 338 items** (`data/clean/review_queue.csv`, also in the dashboard's Review tab):
  - 42 LLM family mappings and 26 SIAM rows with no family.
  - 17 vocabulary items. `LEXUS LS500H` was flagged twice, once against `LEXUS LS` and once against `LEXUS LS500`, and is one decision.
  - 15 `needs_review` pairs and 238 LLM `same` pairs.
  - The 359 LLM "none" family answers can be reviewed through a filter.
- **`n_reviewed`** counts campaigns with nothing waiting for a person: 1,219 of 1,383 today. These have only LLM `different` pairs or no candidates, which are not queued, so this does not mean a person checked them.
- **Matcher reruns are now free** when texts and the prompt are unchanged: 1,610 verdicts reused, 0 sent.
- **Precision from review** appears in `reports/eval.md` once at least one LLM `same` pair is reviewed; the rate itself shows from n >= 10.

