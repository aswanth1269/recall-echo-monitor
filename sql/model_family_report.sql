-- model_family_report: two queries, run in order by src/model_family.py.
-- Every rate shows its n; rates are NULL where n < 10 (CLAUDE.md).

-- 1. Mappings by method and side. "with_family" excludes rows whose family
--    is NULL (rules found nothing, the LLM said "none", or an override cleared it).
SELECT 'nhtsa (distinct make+MODELTXT)' AS side, method,
       count(*) AS n, count(family) AS with_family
FROM nhtsa_model_family GROUP BY method
UNION ALL
SELECT 'siam (row-family pairs)', method, count(*), count(family)
FROM siam_row_family GROUP BY method
ORDER BY side, method;

-- 2. Share of nhtsa_focus RECORDS (not distinct names) that got a family, per make.
SELECT f.make,
       count(*)                                          AS records,
       count(m.family)                                   AS with_family,
       count(*) FILTER (WHERE m.method = 'rule'     AND m.family IS NOT NULL) AS via_rule,
       count(*) FILTER (WHERE m.method = 'llm'      AND m.family IS NOT NULL) AS via_llm,
       count(*) FILTER (WHERE m.method = 'override' AND m.family IS NOT NULL) AS via_override,
       CASE WHEN count(*) >= 10 THEN round(100.0 * count(m.family) / count(*), 1) END AS pct
FROM nhtsa_focus AS f
LEFT JOIN nhtsa_model_family AS m USING (make, MODELTXT)
GROUP BY ROLLUP (f.make)
ORDER BY f.make NULLS LAST;
