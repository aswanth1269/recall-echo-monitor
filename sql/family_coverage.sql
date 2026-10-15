-- family_coverage: per make, how many SIAM rows and NHTSA campaigns map to at
-- least one model family. Every rate shows its n, and rates are NULL where
-- n < 10 (CLAUDE.md: suppress rates where n < 10).
--
-- Inputs: siam_recalls, siam_row_families, nhtsa_model_families
WITH siam AS (
    SELECT s.make,
           count(*)                                         AS n,
           count(*) FILTER (WHERE f.csv_row IS NOT NULL)    AS hit
    FROM siam_recalls AS s
    LEFT JOIN (SELECT DISTINCT csv_row FROM siam_row_families) AS f USING (csv_row)
    GROUP BY s.make
),
nhtsa AS (
    SELECT make,
           count(*)                                 AS n,
           count(*) FILTER (WHERE has_family)       AS hit
    FROM (SELECT CAMPNO, make, bool_or(family IS NOT NULL) AS has_family
          FROM nhtsa_model_families GROUP BY CAMPNO, make)
    GROUP BY make
)
SELECT make,
       siam.n                                                   AS siam_rows,
       siam.hit                                                 AS siam_with_family,
       CASE WHEN siam.n >= 10 THEN round(100.0 * siam.hit / siam.n, 1) END   AS siam_pct,
       nhtsa.n                                                  AS nhtsa_campaigns,
       nhtsa.hit                                                AS nhtsa_with_family,
       CASE WHEN nhtsa.n >= 10 THEN round(100.0 * nhtsa.hit / nhtsa.n, 1) END AS nhtsa_pct
FROM siam FULL OUTER JOIN nhtsa USING (make)
ORDER BY make;
