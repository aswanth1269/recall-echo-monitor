-- echo_by_component: echo rate by component group (COMPNAME text before the first colon), only groups with n >= 10.
--
-- Inputs: india_relevant_campaigns (COMPNAME = first non-null per campaign),
-- matches. Campaigns with no COMPNAME have no group and are left out. n =
-- campaigns (campaign x make) in the group.
WITH camp AS (
    SELECT c.CAMPNO, c.make,
           nullif(trim(split_part(c.COMPNAME, ':', 1)), '')      AS component_group,
           coalesce(bool_or(m.label = 'same'), FALSE)            AS echoes
    FROM india_relevant_campaigns AS c
    LEFT JOIN matches AS m ON m.CAMPNO = c.CAMPNO AND m.make = c.make
    GROUP BY c.CAMPNO, c.make, c.COMPNAME
)
SELECT component_group,
       count(*)                                                AS n,
       count(*) FILTER (WHERE echoes)                          AS echoing,
       round(count(*) FILTER (WHERE echoes) / count(*), 4)     AS echo_rate
FROM camp
WHERE component_group IS NOT NULL
GROUP BY component_group
HAVING count(*) >= 10
ORDER BY echo_rate DESC, n DESC, component_group;
