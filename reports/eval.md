# Matcher evaluation

Ground truth: `data/reference/ground_truth.csv` - 30 pairs, 4 labelled `same`, 26 `different`. Positive class: `same`.

**Who labelled it:** Claude Opus 5.5 (LLM), 2026-10-15, from label_sheet.csv text only - not human-verified. Until a person confirms these labels, the numbers below measure agreement with those labels, not accuracy.

## Read this first

- **The TF-IDF baseline is flattered.** Its threshold (score >= **0.2177**) was chosen to maximise F1 on these same 30 pairs, so it is tuned on its own test set. The LLM matcher was not tuned on them. A fair comparison needs a threshold picked on separate pairs.
- **`needs_review` counts as `different`** for the LLM matcher (1 pair(s) here).
- **Small sample.** With only 4 positive pairs, recall has n=4. Per CLAUDE.md every rate shows its n and the percentage is suppressed when n < 10; the fraction is always given.

## Results

| metric | LLM matcher | TF-IDF baseline |
|---|---|---|
| precision | 3 / 5 (n=5 < 10, rate suppressed) | 3 / 4 (n=4 < 10, rate suppressed) |
| recall | 3 / 4 (n=4 < 10, rate suppressed) | 3 / 4 (n=4 < 10, rate suppressed) |
| f1 | 6 / 9 (n=9 < 10, rate suppressed) | 6 / 8 (n=8 < 10, rate suppressed) |

F1 is shown as 2·TP / (2·TP + FP + FN).

### Confusion matrix - LLM matcher

| | predicted same | predicted different |
|---|---:|---:|
| **truth same** | 3 | 1 |
| **truth different** | 2 | 24 |

### Confusion matrix - TF-IDF baseline (score >= 0.2177)

| | predicted same | predicted different |
|---|---:|---:|
| **truth same** | 3 | 1 |
| **truth different** | 1 | 25 |

## Disagreements

3 pair(s) where at least one matcher differs from the ground truth.

### 16V899000-228 (MERCEDES-BENZ) - wrong: LLM, baseline

- **Ground truth:** `different` (low confidence) - Same symptom (power steering shuts off) but US cause is soldered contacts in cars built 2012-2015 (E-Class 212, S, SL); SIAM covers C 205/E 213 platforms built later - looks like a separate recall
- **LLM:** `same` (confidence 0.90)
  - US evidence: "electric power steering system that may have improperly soldered contacts inside the control unit"
  - India evidence: "The power steering system may become deactivated due to internal error in the electric power steering."
- **Baseline:** score 0.2837 -> `same`

**US text**

> Component: STEERING:ELECTRIC POWER ASSIST SYSTEM
> Defect: Mercedes-Benz USA, LLC. (MBUSA) is recalling certain model year 2013-2015 SL550, 2013-2014 SL63 AMG, 2014-2016 E350, 2014 E350 Cabriolet, E350 Coupe, 2014-2015 S550, 2015 E350 Bluetec, 2015-2016 E400, 2016 E550 Coupe, and Maybach S600 vehicles manufactured August 23, 2012, to August 14, 2015.  The affected vehicles use an electric power steering system that may have improperly soldered contacts inside the control unit, potentially resulting in the power steering system deactivating while driving.
> Consequence: If the control unit pins lose connection, there would be a loss of electric power steering assist, increasing the risk of a crash.  The increase in pin contact resistance can also increase the risk of a fire, even while the vehicle is parked and ignition is off.

**India text**

> Model: C-Class (205 Platform) ; E-Class (213 platform)
> Defect: The power steering system may become deactivated due to internal error in the electric power steering. This could increase the risk of accident.

### 17V047000-21 (BMW) - wrong: LLM

- **Ground truth:** `different` (low confidence) - Both inflator rupture on 3 Series, but US is the DRIVER-side replacement airbag module and SIAM the front PASSENGER gas generator
- **LLM:** `same` (confidence 0.90)
  - US evidence: "These replacement modules contain an air bag inflator that may rupture due to propellant degradation"
  - India evidence: "Malfunctions may occur in the front passenger airbag's gas generator"
- **Baseline:** score 0.0554 -> `different`

**US text**

> Component: AIR BAGS:FRONTAL:DRIVER SIDE:INFLATOR MODULE
> Defect: BMW of North America, LLC (BMW) is recalling certain 2000-2002 BMW 320i, 323i, 325i, 325xi, 330i, 330xi, 323Ci, 325Ci, 330Ci, M3, 323iT, 325iT and 325xiT vehicles, 2001-2002 525i, 530i, 540i, M5, 525iT and 540iT vehicles, and 2001-2003 X5 3.0i, X5 4.4i, and X5 4.6is vehicles.  These vehicles may have had a driver-side air bag module installed as replacement equipment such as after a vehicle crash necessitating replacement of the original air bag, or as a remedy part for a prior recall.  These replacement modules contain an air bag inflator that may rupture due to propellant degradation occurring after long-term exposure to lower absolute humidity, temperature and temperature cycling.
> Consequence: In the event of a crash necessitating deployment of the driver's frontal air bag, the inflator could rupture with metal fragments striking the driver or other occupants resulting in serious injury or death.

**India text**

> Model: 3 Series
> Defect: Malfunctions may occur in the front passenger airbag's gas generator, if this is exposed to high absolute air humidity or high temperature fluctuations over a period of years and/or there are quality deficiencies in the production of the gas generator. When the front passenger airbag is activated in the regular fashion, it may result in high internal pressure in the gas generator. In unfavourable cases, the gas generator may split and metal may escape from the airbag. BMW has not received any reports and is not otherwise aware of any injuries or fatalities related to this issue.

### 18V042000-38 (HONDA) - wrong: LLM, baseline

- **Ground truth:** `same` (high confidence) - Airbag inflator ruptures from excessive internal pressure; Civic on both sides (US 2010-2011 Civic in scope)
- **LLM:** `needs_review` (confidence 1.00) - review reason: us_evidence not found in US text
  - US evidence: "inflator body may get ruptured due to excessive internal pressure"
  - India evidence: "inflator body may get ruptured due to excessive internal pressure"
- **Baseline:** score 0.0249 -> `different`

**US text**

> Component: AIR BAGS:FRONTAL:PASSENGER SIDE:INFLATOR MODULE
> Defect: Honda (American Honda Motor Co.) is recalling certain 2010-2013 Acura TSX, ZDX, Honda Accord Crosstour, Fit, Insight and Pilot vehicles, 2011-13 Acura TSX Sportswagon, 2010-2012 Honda Accord,, 2010-2011 Honda Civic, Civic NGV, Civic Hybrid and CR-V vehicles and 2013 Honda FCX Clarity and Fit EV vehicles ever registered in the states of Alabama, California, Florida, Georgia, Hawaii, Louisiana, Mississippi, South Carolina, Texas, Puerto Rico, American Samoa, Guam, the Northern Mariana Islands (Saipan), and the U.S. Virgin Islands or "Zone A."    Additionally, Honda is recalling certain 2010-2012 Acura TSX, Honda Accord, Accord Crosstour, Fit and Pilot, 2010-2011 Honda Civic, Civic NGV, Civic Hybrid and CR-V, and 2010 Acura ZDX and Honda Insight vehicles ever registered in the states of Arizona, Arkansas, Delaware, District of Columbia, Illinois, Indiana, Kansas, Kentucky, Maryland, Missouri, Nebraska, Nevada, New Jersey, New Mexico, North Carolina, Ohio, Oklahoma, Pennsylvania, Tennessee, Virginia, and West Virginia or "Zone B."    Unless included in Zones A or B above, Honda is recalling certain 2009 Acura TSX, Honda Accord, Civic, Civic NGV, Civic Hybrid, CR-V, Fit, and Pilot vehicles ever registered in the states of Alaska, Colorado, Connecticut, Idaho, Iowa, Maine, Massachusetts, Michigan, Minnesota, Montana, New Hampshire, New York, North Dakota, Oregon, Rhode Island, South Dakota, Utah, Vermont, Washington, Wisconsin, and Wyoming or "Zone C."    These vehicles are equipped with certain air bag inflators assembled as part of the frontal air bag modules, and used as original equipment or replacement equipment (such as after a vehicle crash necessitating replacement of the original air bags), may explode due to propellant degradation occurring after long-term exposure to higher absolute humidity, temperature and temperature cycling.
> Consequence: An inflator explosion may result in sharp metal fragments striking the driver or other occupants resulting in serious injury or death.

**India text**

> Model: Civic
> Defect: At the time of Driver/Passenger side airbag deployment, the inflator body may get ruptured due to excessive internal pressure. Replacement being done as a preventive measure.
