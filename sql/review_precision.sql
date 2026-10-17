-- review_precision: of the pairs the LLM matcher labelled same and a person reviewed, how many the person confirmed; precision is NULL where n_reviewed < 10.
--
-- Input: matches_reviewed. This is precision only: it says nothing about echoes
-- the matcher missed (LLM `different` pairs are not reviewed).
SELECT count(*)                                              AS n_reviewed,
       count(*) FILTER (WHERE final_label = 'same')          AS n_confirmed,
       CASE WHEN count(*) >= 10
            THEN round(count(*) FILTER (WHERE final_label = 'same') / count(*), 4) END AS precision
FROM matches_reviewed
WHERE label = 'same' AND review_status = 'reviewed';
