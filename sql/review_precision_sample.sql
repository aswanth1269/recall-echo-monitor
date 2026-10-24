-- review_precision_sample: review_precision restricted to the saved random sample
-- of LLM `same` pairs (data/reference/review_sample.csv, registered as
-- review_sample_df). Precision is NULL where n_reviewed < 10.
--
-- n_reviewed counts same/different decisions only; cannot_judge is counted on
-- its own. n_not_in_matches counts sampled pairs that no longer exist (e.g.
-- after a family change), so a shrinking sample is visible, not silent.
WITH s AS (SELECT DISTINCT pair_id FROM review_sample_df),
     m AS (SELECT r.* FROM matches_reviewed r JOIN s USING (pair_id)
           WHERE r.label = 'same' AND r.review_status = 'reviewed')
SELECT (SELECT count(*) FROM m WHERE final_label IN ('same', 'different'))      AS n_reviewed,
       (SELECT count(*) FROM m WHERE final_label = 'same')                       AS n_confirmed,
       CASE WHEN (SELECT count(*) FROM m WHERE final_label IN ('same', 'different')) >= 10
            THEN round((SELECT count(*) FROM m WHERE final_label = 'same')
                       / (SELECT count(*) FROM m WHERE final_label IN ('same', 'different')), 4)
            END                                                                  AS precision,
       (SELECT count(*) FROM m WHERE final_label = 'cannot_judge')               AS n_cannot_judge,
       (SELECT count(*) FROM s WHERE pair_id NOT IN (SELECT pair_id FROM matches_reviewed))
                                                                                 AS n_not_in_matches;
