-- candidate_pairs: the top-scored SIAM candidates per India-relevant campaign.
--
-- Inputs: candidate_window (temp, from candidate_window.sql) and kept_scores
--         (CAMPNO, csv_row, score, rank) registered by src/candidates.py after
--         TF-IDF scoring and the top-3 cut.
CREATE OR REPLACE TABLE candidate_pairs AS
SELECT w.CAMPNO, w.make, w.csv_row, w.report_date, w.recall_date,
       w.days_after_report, w.shared_families, w.llm_only,
       k.score, CAST(k.rank AS INTEGER) AS rank
FROM candidate_window AS w
JOIN kept_scores AS k USING (CAMPNO, csv_row)
ORDER BY w.make, w.CAMPNO, k.rank;
