# Recall Echo Monitor - main numbers

Run of **2026-10-24**, after the family review (all 68 items), the 15 `needs_review` pairs
and a fixed random sample of 60 LLM `same` pairs were reviewed by a person. Sources: NHTSA
flat file (retrieved 2026-10-13), SIAM voluntary-recall page (retrieved 2026-10-13), NHTSA
recalls API (OTA flag, fetched 2026-10-15). Every rate shows its n; rates with n < 10 are
suppressed (CLAUDE.md). Full tables: `data/clean/kpi_*.csv`, `reports/eval.md`.

**How to read this.** A US campaign "echoes" when India has a recall with the same defect on
the same model family. **A missing echo is a transparency signal, not proof that no India
recall happened or that anyone did anything wrong**: SIAM is a voluntary list, and the
matcher can miss.

**Where each number comes from.** Every figure below is marked:
- **[person]**: rests only on decisions a person made: the reviewed families, the 75
  reviewed pairs, and the 30 blind labels in `data/reference/human_labels.csv`.
- **[LLM]**: rests on the LLM matcher's verdicts that no person has checked.
- **[API]**: comes from the NHTSA API, with no LLM involved.

## What a person reviewed

| item | reviewed | result |
|---|---:|---|
| model-family items (42 LLM mappings + 26 India rows with no family) | 68 of 68 | 27 LLM mappings kept, 15 changed |
| `needs_review` pairs | 15 of 15 | 15 different (10 have no India defect text) |
| random sample of LLM `same` pairs (seed 42, `review_sample.csv`) | 60 of 60 | **53 confirmed, 7 rejected** |
| other LLM `same` pairs | 0 of 195 | not reviewed |
| vocabulary items | 0 of 17 | not reviewed |

Overall, a person confirmed 53 pairs and rejected 22 (7 sample pairs and 15 `needs_review`)
**[person]**. Five of the 7 rejected sample pairs carry the note "unsure".

## Scope (after the family review)

| | before review (2026-10-15) | now |
|---|---:|---:|
| India-relevant campaign × make rows (unit for every KPI) | 1,383 | **1,355** |
| distinct NHTSA campaigns (20 cover both Audi and VW) | 1,363 | 1,335 |
| candidate pairs | 1,620 | **1,674** |
| matcher verdicts: same / different / needs_review **[LLM]** | 238 / 1,367 / 15 | 255 / 1,404 / 15 |

The family review removed 28 campaign rows: Nissan 45 → 22 (SENTRA and VERSA NOTE are not
the Sunny and Micra), Mercedes −3 (EQE sedans), Honda −2 (HR-V is not WR-V). It added 54
candidate pairs, because 19 India rows that named no family now have one.

## Matcher accuracy

- **Precision on the random sample: 53 of 60 = 88.3% (n = 60) [person].** This is the
  unbiased figure, because the sample was drawn before review. If the 5 "unsure" rejections
  were instead "can't judge", it would be 53 of 55 = 96.4% (n = 55).
- **Against the 30 blind human labels [person]:** 29 pairs scored. Pair 15V287000-241
  (Nissan, labelled `different`) is no longer a candidate pair after the family review.
  - LLM matcher: precision 4 / 5, recall 4 / 4, F1 8 / 9.
  - TF-IDF baseline: precision 3 / 4, recall 3 / 4, F1 6 / 8.
  - All of these have n < 10, so the rates are suppressed and only the fractions are shown.
  - The only LLM error is 16V899000-228 (Mercedes EPS), which you labelled `different`,
    marked "unsure".
- **Recall has n = 4**, too small to state as a rate. Missed echoes among the 1,404 LLM
  `different` pairs are not measured.

## Echo rate (`kpi_echo_rate.csv`)

**Confirmed by a person: 46 of 1,355 = 3.4% [person].** This is a **lower bound**:
- Only the 60 sampled LLM `same` pairs have been reviewed. The other 195 count as no
  echo until a person confirms them.
- **The LLM alone finds 169 of 1,355 = 12.5% [LLM].** That count includes pairs a person
  would reject; on the sample, about 1 in 8 were rejected.
- **The true echo rate lies between these two figures.**

| make | confirmed / n [person] | rate | LLM alone / n [LLM] | rate |
|---|---:|---:|---:|---:|
| MERCEDES-BENZ | 21 / 459 | 4.6% | 69 / 459 | 15.0% |
| AUDI | 9 / 163 | 5.5% | 31 / 163 | 19.0% |
| HONDA | 7 / 85 | 8.2% | 18 / 85 | 21.2% |
| BMW | 4 / 225 | 1.8% | 20 / 225 | 8.9% |
| TOYOTA | 2 / 84 | 2.4% | 15 / 84 | 17.9% |
| HYUNDAI | 1 / 86 | 1.2% | 7 / 86 | 8.1% |
| KIA | 1 / 69 | 1.4% | 2 / 69 | 2.9% |
| VOLVO | 1 / 55 | 1.8% | 4 / 55 | 7.3% |
| VOLKSWAGEN | 0 / 107 | 0.0% | 3 / 107 | 2.8% |
| NISSAN | 0 / 22 | 0.0% | 0 / 22 | 0.0% |

| segment | confirmed / n [person] | rate | LLM alone / n [LLM] | rate |
|---|---:|---:|---:|---:|
| luxury | 35 / 902 | 3.9% | 124 / 902 | 13.7% |
| mass_market | 11 / 453 | 2.4% | 45 / 453 | 9.9% |

**Do not rank makes on the confirmed column.** It mostly reflects which campaigns happened
to fall into the 60-pair sample. Per-make confirmed rates need the remaining 195 LLM `same`
pairs reviewed.

**Hand check:** recounting HONDA from `matches_reviewed` in plain Python, without the KPI
SQL, gives 85 relevant campaigns and 7 with a pair a person confirmed as `same`. This is the
same as `kpi_echo_rate.csv`.

**`n_reviewed`** in the KPI files counts campaigns with nothing waiting for a person: 1,216
of 1,355. Most of these have only LLM `different` pairs or no candidates, which are never
queued, so this does **not** mean a person checked them.

## Echo lag (`kpi_echo_lag.csv`): US report date to earliest matched India recall

These are percentiles, not rates, so they are shown at every n.

| group | n | median days | p25 | p75 | India first |
|---|---:|---:|---:|---:|---:|
| all, confirmed **[person]** | 46 | 112 | 33 | 226 | 6 |
| all, LLM alone **[LLM]** | 169 | 104 | 32 | 210 | 20 |
| luxury **[person]** | 35 | 133 | 43 | 221 | 4 |
| mass_market **[person]** | 11 | 32 | 11 | 197 | 2 |
| MERCEDES-BENZ **[person]** | 21 | 113 | 39 | 201 | 3 |
| AUDI **[person]** | 9 | 140 | 80 | 329 | 0 |
| HONDA **[person]** | 7 | 32 | 21 | 224 | 1 |

The median lag is about the same whether it rests on confirmed echoes (112 days, n = 46) or
on the LLM alone (104 days, n = 169). "India first" means India recalled before the US
report.

## Echo rate by component group (`kpi_echo_by_component.csv`, groups with n >= 10) [person]

With only 46 confirmed echoes, most groups have 0 to 2 confirmed. The biggest groups:
AIR BAGS 14 / 256 = 5.5%, ELECTRICAL SYSTEM 10 / 194 = 5.2%, STRUCTURE 6 / 77 = 7.8%.
US-only regulatory items still show no confirmed echo: EQUIPMENT 0 / 58, BACK OVER
PREVENTION 0 / 42, EXTERIOR LIGHTING 0 / 51. This is expected rather than suspicious.

## Over-the-air remedies (`kpi_ota.csv`)

- **OTA share: 13 of 1,351 campaigns with a known flag = 1.0% [API].**
  - 4 campaigns came back NULL: the API returned results but none had a true/false flag. They are never guessed.
- **By make,** only Volvo (3 / 55 = 5.5%), Hyundai (2 / 86), Mercedes (7 / 459) and Kia (1 / 69) have any OTA campaigns **[API]**.
- **By year,** the first OTA campaign is in 2019, and 2021 has the most (4 / 139) **[API]**.
- **Echo rate, confirmed [person]:** OTA 1 / 13 = 7.7% vs non-OTA 45 / 1,338 = 3.4%.
  - n = 13 and a single echo, so treat the difference as a lead to follow, not a finding.
- **The API flag is a lower bound** (checked 2026-10-15). All 13 flagged campaigns mention OTA in their remedy text, but 10 more remedies say OTA while the API flag is `false` (e.g. 26V341000).
  - Going by the remedy text, there would be about 23 OTA campaigns.

## Caveats

- **The confirmed echo rate is a lower bound** until the other 195 LLM `same` pairs are reviewed. The LLM-alone rate is not an upper bound either, because the matcher can miss echoes (recall is measured on only 4 positive pairs).
- **Precision rests on 60 sampled pairs and one reviewer.** Five of the 7 rejections are marked "unsure".
- **The 30-pair test set is small and was used to tune the prompt.**
  - The LLM prompt was sharpened on these same pairs, and the TF-IDF threshold is chosen on them.
  - Both matcher scores are therefore flattered. Labels are now a person's, given blind (`human_labels.csv`).
- **The vocabulary (17 items) has not been reviewed.** Possible same-model spellings (e.g. XCENT vs XCIENT) are not merged.
- **6 India rows that name several models are mapped to only one family** (e.g. row 267 "A6, A7 & A8" → A6). Echoes for the other models in those rows can be missed until they are given all their families.
- **10 of the 15 `needs_review` decisions are "different" because the India row has no defect text.** Nothing could be compared; "can't judge" would describe them better. They are not echoes either way.
- **Echo is per make.** A campaign covering Audi and VW is counted once under each.
- **Matcher reruns cost only new or changed pairs.** The last rebuild reused 1,572 verdicts and sent 92 new pairs to the LLM.
