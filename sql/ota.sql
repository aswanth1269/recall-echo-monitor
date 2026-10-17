-- ota: share of India-relevant campaigns whose remedy is over-the-air, by make and by report year; and echo rate for OTA vs non-OTA campaigns.
--
-- Inputs: india_relevant_campaigns, campaign_ota (CAMPNO, ota true/false/NULL),
-- matches_reviewed (echo = confirmed by a person). part = 'ota_share' rows: n = relevant campaigns (campaign x make),
-- n_known_ota = those with a known flag (NULL is unknown, never counted as
-- false), ota_share = ota_true / n_known_ota. part = 'echo_by_ota' rows: ota =
-- true / false / NULL (unknown), echo_rate = echoing / n. Rates are NULL where
-- their own n < 10.
WITH camp AS (
    SELECT c.CAMPNO, c.make, year(c.report_date) AS report_year, o.ota,
           coalesce(bool_or(m.final_label = 'same'), FALSE) AS echoes,
           coalesce(bool_or(m.label IN ('same', 'needs_review')
                            AND m.review_status <> 'reviewed'), FALSE) AS awaiting
    FROM india_relevant_campaigns AS c
    LEFT JOIN campaign_ota AS o USING (CAMPNO)
    LEFT JOIN matches_reviewed AS m ON m.CAMPNO = c.CAMPNO AND m.make = c.make
    GROUP BY c.CAMPNO, c.make, year(c.report_date), o.ota
),
share AS (
    SELECT 'ota_share' AS part,
           CASE WHEN grouping(make) = 0 THEN 'make'
                WHEN grouping(report_year) = 0 THEN 'report_year' ELSE 'all' END AS level,
           coalesce(make, CAST(report_year AS VARCHAR), 'ALL') AS grp,
           CAST(NULL AS BOOLEAN)                           AS ota,
           count(*)                                        AS n,
           count(ota)                                      AS n_known_ota,
           count(*) FILTER (WHERE ota)                     AS ota_true,
           CASE WHEN count(ota) >= 10
                THEN round(count(*) FILTER (WHERE ota) / count(ota), 4) END AS ota_share,
           CAST(NULL AS BIGINT)                            AS echoing,
           CAST(NULL AS DOUBLE)                            AS echo_rate,
           CAST(NULL AS BIGINT)                            AS n_reviewed
    FROM camp
    GROUP BY GROUPING SETS ((make), (report_year), ())
),
by_ota AS (
    SELECT 'echo_by_ota' AS part, 'ota' AS level,
           CASE WHEN ota IS NULL THEN 'unknown' WHEN ota THEN 'ota' ELSE 'not_ota' END AS grp,
           ota,
           count(*)                                        AS n,
           CAST(NULL AS BIGINT)                            AS n_known_ota,
           CAST(NULL AS BIGINT)                            AS ota_true,
           CAST(NULL AS DOUBLE)                            AS ota_share,
           count(*) FILTER (WHERE echoes)                  AS echoing,
           CASE WHEN count(*) >= 10
                THEN round(count(*) FILTER (WHERE echoes) / count(*), 4) END AS echo_rate,
           count(*) FILTER (WHERE NOT awaiting)            AS n_reviewed
    FROM camp
    GROUP BY ota
)
SELECT * FROM share
UNION ALL
SELECT * FROM by_ota
ORDER BY part DESC, level, grp;
