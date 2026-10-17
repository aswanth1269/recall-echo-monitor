-- echo_rate: share of India-relevant campaigns (campaign x make) with at least one pair a person confirmed as `same`, by make, by segment and overall.
--
-- Inputs: india_relevant_campaigns, matches_reviewed, make_segments (make,
-- segment from india_makes.csv). An echo needs final_label = 'same' (a current
-- human decision); an LLM `same` nobody confirmed is not an echo. A campaign is
-- settled when none of its LLM `same` / needs_review pairs still awaits a
-- current decision; n_reviewed counts settled campaigns.
-- echo_rate is NULL where n < 10 (CLAUDE.md).
WITH camp AS (
    SELECT c.CAMPNO, c.make, s.segment,
           coalesce(bool_or(m.final_label = 'same'), FALSE)                     AS echoes,
           coalesce(bool_or(m.label IN ('same', 'needs_review')
                            AND m.review_status <> 'reviewed'), FALSE)          AS awaiting
    FROM india_relevant_campaigns AS c
    JOIN make_segments AS s USING (make)
    LEFT JOIN matches_reviewed AS m ON m.CAMPNO = c.CAMPNO AND m.make = c.make
    GROUP BY c.CAMPNO, c.make, s.segment
)
SELECT CASE WHEN grouping(make) = 0 THEN 'make'
            WHEN grouping(segment) = 0 THEN 'segment' ELSE 'all' END   AS level,
       coalesce(make, segment, 'ALL')                                  AS grp,
       count(*)                                                        AS n,
       count(*) FILTER (WHERE echoes)                                  AS echoing,
       CASE WHEN count(*) >= 10
            THEN round(count(*) FILTER (WHERE echoes) / count(*), 4) END AS echo_rate,
       count(*) FILTER (WHERE NOT awaiting)                            AS n_reviewed,
       count(*) FILTER (WHERE NOT echoes AND awaiting)                 AS awaiting_review_not_echoing
FROM camp
GROUP BY GROUPING SETS ((make), (segment), ())
ORDER BY level, grp;
