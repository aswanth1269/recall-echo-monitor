-- candidate_window: every (India-relevant NHTSA campaign, SIAM row) pair that
-- could be the same recall, before text scoring.
--
-- A pair needs: the same make; a SIAM recall_date that is not NULL and falls
-- from 180 days before to 730 days after the NHTSA report_date (both ends
-- inclusive); and at least one family in common.
-- llm_only: TRUE when every shared family is llm-only on the NHTSA side.
CREATE OR REPLACE TEMP TABLE candidate_window AS
WITH siam_fams AS (
    SELECT csv_row, list_sort(list(DISTINCT family)) AS families
    FROM siam_row_family GROUP BY csv_row
),
camp_fam_flags AS (     -- per campaign family: did only LLM-mapped models give it?
    SELECT f.CAMPNO, m.family, bool_and(m.method = 'llm') AS family_llm_only
    FROM nhtsa_focus AS f
    JOIN nhtsa_model_family AS m USING (make, MODELTXT)
    WHERE m.family IS NOT NULL
    GROUP BY f.CAMPNO, m.family
),
pairs AS (
    SELECT c.CAMPNO, c.make, s.csv_row, c.report_date, s.recall_date,
           CAST(s.recall_date - c.report_date AS INTEGER)       AS days_after_report,
           list_sort(list_intersect(c.families, sf.families))    AS shared_families
    FROM india_relevant_campaigns AS c
    JOIN siam_recalls AS s
      ON s.make = c.make
     AND s.recall_date IS NOT NULL
     AND s.recall_date BETWEEN c.report_date - INTERVAL 180 DAY
                           AND c.report_date + INTERVAL 730 DAY
    JOIN siam_fams AS sf ON sf.csv_row = s.csv_row
    WHERE len(list_intersect(c.families, sf.families)) > 0
)
SELECT p.*,
       (SELECT bool_and(ff.family_llm_only) FROM camp_fam_flags AS ff
        WHERE ff.CAMPNO = p.CAMPNO AND list_contains(p.shared_families, ff.family)) AS llm_only
FROM pairs AS p;
