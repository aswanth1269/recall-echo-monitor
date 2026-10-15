# Data sources and provenance

Everything in `data/raw/` is recorded here: where it came from, when it was
retrieved, and what shape it was in on that date. Re-retrieving any of these
will produce different row counts over time, so the counts below are the
as-of-retrieval baseline.

---

## 1. NHTSA recall flat file (US side)

| | |
|---|---|
| Dataset index page | <https://www.nhtsa.gov/nhtsa-datasets-and-apis> |
| Data file | <https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip> |
| Field layout | <https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt> |
| Retrieved | 2026-10-13 |
| Server `Last-Modified` | Tue, 13 Oct 2026 07:04:12 GMT |
| Local files | `data/raw/FLAT_RCL_POST_2010.zip`, `data/raw/RCL.txt` |

SHA-256 as downloaded:

```
5616b921bb1e3bea10dbfb618bac20a0e60023c0b1cf74447a78ac0f380436e0  FLAT_RCL_POST_2010.zip
3438cc7e4131efa401dfe6437dbe035607e092c129155dfb411d459d58802203  RCL.txt
```

Shape on the retrieval date:

- Zip contains one member, `FLAT_RCL_POST_2010.txt`, 311,503,875 bytes.
- 245,855 records. **Every record split to exactly 29 fields** on tab, so the
  file parses cleanly with no ragged rows.
- Tab-delimited, no header row, no quoting. Dates are `YYYYMMDD`.
- UTF-8. Checked 2026-10-15: all 40,007 bytes above 0x7F decode as strict
  UTF-8 with no errors. Do not use `latin-1`: it never raises, but it garbles
  the curly quotes and dashes in 10,640 records without changing the row count.
- 1,782 distinct `MAKETXT` values.
- Field names and sizes are in `RCL.txt`; field 20 (`DESC_DEFECT`) and 22
  (`CORRECTIVE_ACTION`) were widened to 6000 chars and fields 28/29
  (`DO_NOT_DRIVE`, `PARK_OUTSIDE`) added in the May 2025 revision.

A record is one make/model/year within a campaign, not one campaign, so
record counts are much larger than recall counts. `CAMPNO` is the campaign ID.

The flat file has no over-the-air (OTA) field, so the OTA flag will come from
the NHTSA recalls API, whose `overTheAirUpdate` boolean was confirmed on
2026-10-15 at `https://api.nhtsa.gov/recalls/recallsByVehicle`.

### Join key check (run 2026-10-13)

All 10 picked makes resolve against `MAKETXT` exactly, uppercased:

| make | `MAKETXT` | records |
|---|---|---:|
| Toyota | `TOYOTA` | 9,528 |
| Honda | `HONDA` | 9,653 |
| Hyundai | `HYUNDAI` | 1,465 |
| Kia | `KIA` | 1,400 |
| Nissan | `NISSAN` | 2,821 |
| Volkswagen | `VOLKSWAGEN` | 1,211 |
| Mercedes-Benz | `MERCEDES-BENZ` | 45,435 |
| BMW | `BMW` | 4,009 |
| Audi | `AUDI` | 2,753 |
| Volvo | `VOLVO` | 1,696 |

Note `LEXUS` is its own `MAKETXT` (5,554 records) on the US side, but on the
India side SIAM files Lexus under Toyota Kirloskar. That asymmetry has to be
handled explicitly when matching — see the gotchas below.

---

## 2. SIAM voluntary recall portal (India side)

| | |
|---|---|
| Source page | <https://www.siam.in/siam-voluntary-recall.aspx> |
| Retrieved | 2026-10-13 |
| HTML snapshot | `data/raw/siam_voluntary_recall_snapshot.html` |
| Parser | `src/parse_siam.py` |
| Outputs | `data/raw/siam_recalls.csv`, `data/raw/siam_recalls_all.csv` |

**Deviation from the original plan:** the brief said to copy these rows by
hand. The portal has no API and no CSV export, but it renders the entire
table server-side into a single `<table>` on first page load, so it was
parsed from a saved snapshot instead. Same result, reproducible, and no
transcription errors. The snapshot is kept so the CSV can always be rebuilt
from exactly what the page showed on 2026-10-13.

Shape on the retrieval date:

- 546 data rows, 10 columns, covering voluntary recalls from 2012 to 2026.
- Columns: `#`, Manufacturer's Name, Voluntary Recall Date, Model, Variant,
  Status, Total no. of target Recall Vehicles, Manufactured From,
  Manufactured To, Description of Potential Defect.
- Recall dates are `DD-MMM-YYYY`; manufacturing dates are `DD/MM/YYYY`.
- Covers two-wheelers and commercial vehicles as well as cars.

`siam_recalls_all.csv` is all 546 rows. `siam_recalls.csv` is the 355 rows
belonging to the 10 picked makes:

| make | recalls | vehicles |
|---|---:|---:|
| Mercedes-Benz | 149 | 124,108 |
| Toyota | 51 | 102,536 |
| Honda | 47 | 963,939 |
| Audi | 35 | 54,105 |
| BMW | 31 | 26,828 |
| Nissan | 13 | 42,105 |
| Volkswagen | 12 | 485,700 |
| Hyundai | 10 | 45,839 |
| Volvo | 6 | 5,760 |
| Kia | 1 | 1,380 |
| **total** | **355** | **1,852,300** |

Three columns are added by the parser and are *derived, not from SIAM*:
`make` (resolved brand), `brands_detected` (every brand found in the model
text), `make_resolution` (how `make` was decided: `entity`, `model_token`,
or `out_of_scope`).

---

## 3. India makes and models

| file | rows | how it was built |
|---|---:|---|
| `data/raw/india_makes.csv` | 10 | hand-picked, 6 mass-market + 4 luxury |
| `data/raw/india_models.csv` | 96 | 51 rows from official India sites, 45 from secondary sources |

Selection rule: the brand must sell cars in **both** the US and India today,
and must appear in the SIAM data. That rules out some obvious India names —
Maruti Suzuki, Tata and Mahindra have no US passenger-car presence; Suzuki
left the US in 2012; Ford left India in 2021; Renault does not sell in the US.

Model lists read off each brand's official India site on 2026-10-13:
Toyota, Honda, Hyundai, Kia, Nissan, Volkswagen.

Mercedes-Benz, BMW, Audi and Volvo India all return HTTP 403 to automated
fetches, so their model lists came from CarWale / Autocar India instead.
Those 45 rows carry `source_type=secondary_source` and a `VERIFY on official
site` note. **They should be checked by hand before being treated as
authoritative.**

---

## Gotchas that will bite the matching step

1. **SIAM files under the legal entity, not the brand.** `Skoda Auto
   Volkswagen India Pvt. Ltd.` covers Volkswagen, Audi, Škoda, Porsche *and*
   Lamborghini. Those 86 rows can only be split by reading the model text —
   that is what `BRAND_TOKENS` in `src/parse_siam.py` does. Audi alone is 35
   rows that would be invisible if you grouped by manufacturer name.

2. **Mercedes uses internal platform codes in the model field.** Values like
   `167 (GLE, GLS)`, `205 (C-Class)`, `213 (E-Class)`, `177 (A-Class)`,
   `223 (S-Class)` appear instead of marketing names. Fuzzy-matching these
   against NHTSA's `MODELTXT` will fail without a platform-code lookup.
   Mercedes is the single largest India-side make (149 of 355 rows), so this
   is not an edge case.

3. **Sub-brands ride inside a parent's rows.** 26 rows carry a sub-brand in
   the model text: Lexus under Toyota Kirloskar, Mini under BMW India,
   Maybach under Mercedes. `brands_detected` flags them; decide per analysis
   whether to split them out. Rows like
   `Innova Crysta; Fortuner; Corolla; Lexus ...` mix parent and sub-brand in
   one row, so a blind reassignment would be wrong.

4. **One SIAM row can list many models.** e.g.
   `Polo, Vento, Jetta, Pasat` — the model field needs splitting before any
   per-model join.

5. **Names differ across markets** for the same car: X-Trail/Rogue,
   Land Cruiser 300/Land Cruiser, Verna/Elantra. Name equality alone will
   under-count overlap.

6. **Dirty source values.** Expect `Porche` for Porsche, `Pasat` for Passat,
   `Scropio` for Scorpio, `Breeza` for Brezza, and an impossible
   manufacturing date `30/15/2015`. Validate, don't trust.

7. **Thin India-side samples.** Kia has 1 SIAM recall and Volvo 6, against
   1,400 and 1,696 NHTSA records. Any per-make rate comparison for those two
   is noise, not signal.
