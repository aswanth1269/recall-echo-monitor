-- nhtsa_focus: NHTSA records for the makes in data/raw/india_makes.csv
-- whose report date (RCDATE, "Part 573 report received") is 2015-01-01 or later.
--
-- Inputs : nhtsa_raw (all text, empty = NULL),
--          focus_makes (make, maketxt, nhtsa_make)
-- make   : the primary NHTSA spelling (e.g. MERCEDES-BENZ), the same values as
--          siam_recalls.make, so brands compare on `make`. make_label keeps
--          the india_makes.csv name.
-- Matching: MAKETXT is compared after upper(trim()), because the source has
--           case/spacing variants such as 'MERCEDES-BENz'.
-- Nulls   : YEARTXT '9999' means unknown in RCL.txt, so model_year is NULL.
--           Rows with no parseable RCDATE cannot be shown to be 2015+ and are
--           excluded; load_issues counts them.
CREATE OR REPLACE TABLE nhtsa_focus AS
SELECT
    f.nhtsa_make AS make,
    f.make       AS make_label,
    CAST(try_strptime(r.RCDATE, '%Y%m%d') AS DATE)                   AS report_date,
    CASE WHEN r.YEARTXT = '9999' THEN NULL
         ELSE try_cast(r.YEARTXT AS INTEGER) END                    AS model_year,
    r.*
FROM nhtsa_raw AS r
JOIN focus_makes AS f
  ON upper(trim(r.MAKETXT)) = f.maketxt
WHERE CAST(try_strptime(r.RCDATE, '%Y%m%d') AS DATE) >= DATE '2015-01-01';
