# Matcher evaluation

Ground truth: `data/reference/ground_truth.csv` - 30 pairs, 5 labelled `same`, 25 `different`. Positive class: `same`.

**Who labelled it:** Claude Opus 5.5 (LLM), 2026-10-15, from label_sheet.csv text only - not human-verified; human_label filled for 0 of 30. Until a person confirms these labels, the numbers below measure agreement with those labels, not accuracy.

## Read this first

- **The TF-IDF baseline is flattered.** Its threshold (score >= **0.2177**) was chosen to maximise F1 on these same 30 pairs, so it is tuned on its own test set. A fair comparison needs a threshold picked on separate pairs.
- **The LLM matcher is flattered too.** Its prompt was sharpened after reviewing its disagreements on these same pairs (2026-10-15, round 1: same component and position; each quote from its own side), so it has also been tuned on its own test set.
- **Labels revised after seeing matcher output** (2 note(s) in the ground-truth file):
  - `16V899000-228` (now `same`): Round 1 review, 2026-10-15: changed different -> same. The original reason relied on platform and build period (E-Class 212 built 2012-2015 vs C 205/E 213), but the definition allows a different model-year range; the defect and the model family match.
  - `17V047000-21` (now `different`): Round 1 review, 2026-10-15: kept different after the LLM said same. US is the driver-side replacement airbag module; SIAM is the front passenger airbag gas generator - same failure mode, different component.
- **`needs_review` counts as `different`** for the LLM matcher (0 pair(s) here).
- **Small sample.** With only 5 positive pairs, recall has n=5. Per CLAUDE.md every rate shows its n and the percentage is suppressed when n < 10; the fraction is always given.

## Results

| metric | LLM matcher | TF-IDF baseline |
|---|---|---|
| precision | 5 / 5 (n=5 < 10, rate suppressed) | 4 / 4 (n=4 < 10, rate suppressed) |
| recall | 5 / 5 (n=5 < 10, rate suppressed) | 4 / 5 (n=5 < 10, rate suppressed) |
| f1 | 10 / 10 = 1.000 (n=10) | 8 / 9 (n=9 < 10, rate suppressed) |

F1 is shown as 2·TP / (2·TP + FP + FN).

### Confusion matrix - LLM matcher

| | predicted same | predicted different |
|---|---:|---:|
| **truth same** | 5 | 0 |
| **truth different** | 0 | 25 |

### Confusion matrix - TF-IDF baseline (score >= 0.2177)

| | predicted same | predicted different |
|---|---:|---:|
| **truth same** | 4 | 1 |
| **truth different** | 0 | 25 |

## Matcher precision (human-reviewed)

No LLM `same` pairs have been reviewed yet (run `python -m src.review`).

## Disagreements

1 pair(s) where at least one matcher differs from the ground truth.

### 18V042000-38 (HONDA) - wrong: baseline

- **Ground truth:** `same` (high confidence) - Airbag inflator ruptures from excessive internal pressure; Civic on both sides (US 2010-2011 Civic in scope)
- **LLM:** `same` (confidence 0.90)
  - US evidence: "may explode due to propellant degradation occurring after long-term exposure"
  - India evidence: "the inflator body may get ruptured due to excessive internal pressure"
- **Baseline:** score 0.0249 -> `different`

**US text**

> Component: AIR BAGS:FRONTAL:PASSENGER SIDE:INFLATOR MODULE
> Defect: Honda (American Honda Motor Co.) is recalling certain 2010-2013 Acura TSX, ZDX, Honda Accord Crosstour, Fit, Insight and Pilot vehicles, 2011-13 Acura TSX Sportswagon, 2010-2012 Honda Accord,, 2010-2011 Honda Civic, Civic NGV, Civic Hybrid and CR-V vehicles and 2013 Honda FCX Clarity and Fit EV vehicles ever registered in the states of Alabama, California, Florida, Georgia, Hawaii, Louisiana, Mississippi, South Carolina, Texas, Puerto Rico, American Samoa, Guam, the Northern Mariana Islands (Saipan), and the U.S. Virgin Islands or "Zone A."    Additionally, Honda is recalling certain 2010-2012 Acura TSX, Honda Accord, Accord Crosstour, Fit and Pilot, 2010-2011 Honda Civic, Civic NGV, Civic Hybrid and CR-V, and 2010 Acura ZDX and Honda Insight vehicles ever registered in the states of Arizona, Arkansas, Delaware, District of Columbia, Illinois, Indiana, Kansas, Kentucky, Maryland, Missouri, Nebraska, Nevada, New Jersey, New Mexico, North Carolina, Ohio, Oklahoma, Pennsylvania, Tennessee, Virginia, and West Virginia or "Zone B."    Unless included in Zones A or B above, Honda is recalling certain 2009 Acura TSX, Honda Accord, Civic, Civic NGV, Civic Hybrid, CR-V, Fit, and Pilot vehicles ever registered in the states of Alaska, Colorado, Connecticut, Idaho, Iowa, Maine, Massachusetts, Michigan, Minnesota, Montana, New Hampshire, New York, North Dakota, Oregon, Rhode Island, South Dakota, Utah, Vermont, Washington, Wisconsin, and Wyoming or "Zone C."    These vehicles are equipped with certain air bag inflators assembled as part of the frontal air bag modules, and used as original equipment or replacement equipment (such as after a vehicle crash necessitating replacement of the original air bags), may explode due to propellant degradation occurring after long-term exposure to higher absolute humidity, temperature and temperature cycling.
> Consequence: An inflator explosion may result in sharp metal fragments striking the driver or other occupants resulting in serious injury or death.

**India text**

> Model: Civic
> Defect: At the time of Driver/Passenger side airbag deployment, the inflator body may get ruptured due to excessive internal pressure. Replacement being done as a preventive measure.
