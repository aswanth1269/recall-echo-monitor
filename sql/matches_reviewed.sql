-- matches_reviewed: matches plus the human decision from pair_reviews.csv; final_label is that decision when its fingerprint matches the current texts, else NULL.
--
-- Input: matches, pair_reviews_df (registered from data/reference/pair_reviews.csv).
-- review_status: unreviewed (no decision), reviewed (fingerprint matches),
-- stale (texts changed since review: decision kept but not applied).
-- The LLM's label and llm_label are copied unchanged.
CREATE OR REPLACE TABLE matches_reviewed AS
SELECT m.*,
       r.decision                                                  AS human_decision,
       CASE WHEN r.pair_id IS NULL THEN 'unreviewed'
            WHEN r.fingerprint = m.fingerprint THEN 'reviewed'
            ELSE 'stale' END                                       AS review_status,
       CASE WHEN r.fingerprint = m.fingerprint THEN r.decision END AS final_label
FROM matches AS m
-- cast: an empty decisions file arrives with untyped columns
LEFT JOIN (SELECT CAST(pair_id AS VARCHAR) AS pair_id, CAST(decision AS VARCHAR) AS decision,
                  CAST(fingerprint AS VARCHAR) AS fingerprint
           FROM pair_reviews_df) AS r ON r.pair_id = m.pair_id
ORDER BY m.CAMPNO, m.csv_row;
