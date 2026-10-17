-- echo_lag: days from US report_date to the earliest India recall_date among pairs a person confirmed as `same`, per echoing campaign; median, p25, p75 and how many had India first, by make, by segment and overall.
--
-- Inputs: india_relevant_campaigns, matches_reviewed, siam_recalls, make_segments.
-- n counts confirmed echoes only, so it is already the reviewed count.
-- lag_days = earliest matched recall_date - report_date (negative = India
-- recalled first). n = echoing campaigns (campaign x make). Percentiles use
-- linear interpolation (quantile_cont); they are not rates, so they are shown
-- for every n, with n beside them.
WITH lag AS (
    SELECT c.CAMPNO, c.make, s.segment,
           CAST(min(r.recall_date) - c.report_date AS INTEGER) AS lag_days
    FROM india_relevant_campaigns AS c
    JOIN make_segments AS s USING (make)
    JOIN matches_reviewed AS m
      ON m.CAMPNO = c.CAMPNO AND m.make = c.make AND m.final_label = 'same'
    JOIN siam_recalls AS r ON r.csv_row = m.csv_row
    WHERE r.recall_date IS NOT NULL
    GROUP BY c.CAMPNO, c.make, s.segment, c.report_date
)
SELECT CASE WHEN grouping(make) = 0 THEN 'make'
            WHEN grouping(segment) = 0 THEN 'segment' ELSE 'all' END   AS level,
       coalesce(make, segment, 'ALL')                                  AS grp,
       count(*)                                                        AS n,
       quantile_cont(lag_days, 0.5)                                    AS median_days,
       quantile_cont(lag_days, 0.25)                                   AS p25_days,
       quantile_cont(lag_days, 0.75)                                   AS p75_days,
       count(*) FILTER (WHERE lag_days < 0)                            AS india_first
FROM lag
GROUP BY GROUPING SETS ((make), (segment), ())
ORDER BY level, grp;
