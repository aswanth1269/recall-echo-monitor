-- echo_rate: share of India-relevant campaigns (campaign x make) with at least one pair labelled `same` in matches, by make, by segment and overall.
--
-- Inputs: india_relevant_campaigns, matches, make_segments (make, segment from
-- india_makes.csv). needs_review is not an echo; it is counted separately.
-- echo_rate is NULL where n < 10 (CLAUDE.md).
WITH camp AS (
    SELECT c.CAMPNO, c.make, s.segment,
           coalesce(bool_or(m.label = 'same'), FALSE)         AS echoes,
           coalesce(bool_or(m.label = 'needs_review'), FALSE) AS has_review
    FROM india_relevant_campaigns AS c
    JOIN make_segments AS s USING (make)
    LEFT JOIN matches AS m ON m.CAMPNO = c.CAMPNO AND m.make = c.make
    GROUP BY c.CAMPNO, c.make, s.segment
)
SELECT CASE WHEN grouping(make) = 0 THEN 'make'
            WHEN grouping(segment) = 0 THEN 'segment' ELSE 'all' END   AS level,
       coalesce(make, segment, 'ALL')                                  AS grp,
       count(*)                                                        AS n,
       count(*) FILTER (WHERE echoes)                                  AS echoing,
       CASE WHEN count(*) >= 10
            THEN round(count(*) FILTER (WHERE echoes) / count(*), 4) END AS echo_rate,
       count(*) FILTER (WHERE NOT echoes AND has_review)               AS needs_review_not_echoing
FROM camp
GROUP BY GROUPING SETS ((make), (segment), ())
ORDER BY level, grp;
