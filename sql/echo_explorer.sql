-- echo_explorer: one row per India-relevant campaign x make with its best pair (same > needs_review > different), or label no_candidate when it had no candidate pair.
--
-- Inputs: india_relevant_campaigns, matches, candidate_pairs, siam_recalls,
-- make_segments. "Best pair": a `same` pair with the earliest India recall_date
-- (the same date echo_lag uses), else a needs_review pair, else the different
-- pair with the best TF-IDF rank. lag_days is only filled for `same`. Missing
-- values stay NULL.
WITH ranked AS (
    SELECT m.CAMPNO, m.make, m.csv_row, m.label, m.confidence,
           m.us_evidence, m.india_evidence,
           s.recall_date, s.model, s.defect_description,
           row_number() OVER (
               PARTITION BY m.CAMPNO, m.make
               ORDER BY CASE m.label WHEN 'same' THEN 0 WHEN 'needs_review' THEN 1 ELSE 2 END,
                        CASE WHEN m.label = 'same' THEN s.recall_date END NULLS LAST,
                        p.rank, m.csv_row) AS pick
    FROM matches AS m
    JOIN candidate_pairs AS p
      ON p.CAMPNO = m.CAMPNO AND p.make = m.make AND p.csv_row = m.csv_row
    JOIN siam_recalls AS s ON s.csv_row = m.csv_row
)
SELECT c.CAMPNO, c.make, ms.segment, c.report_date,
       c.COMPNAME                                     AS component,
       c.DESC_DEFECT                                  AS us_defect,
       coalesce(b.label, 'no_candidate')              AS label,
       b.recall_date                                  AS india_recall_date,
       b.model                                        AS india_model,
       b.defect_description                           AS india_defect,
       CASE WHEN b.label = 'same'
            THEN CAST(b.recall_date - c.report_date AS INTEGER) END AS lag_days,
       b.confidence, b.us_evidence, b.india_evidence
FROM india_relevant_campaigns AS c
JOIN make_segments AS ms USING (make)
LEFT JOIN ranked AS b ON b.CAMPNO = c.CAMPNO AND b.make = c.make AND b.pick = 1
ORDER BY c.report_date DESC, c.CAMPNO, c.make;
