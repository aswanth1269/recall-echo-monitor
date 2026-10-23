-- review_precision: of the pairs the LLM matcher labelled same and a person reviewed, how many the person confirmed; precision is NULL where n_reviewed < 10.
--
-- Input: matches_reviewed. This is precision only: it says nothing about echoes
-- the matcher missed (LLM `different` pairs are not reviewed). Pairs a person
-- marked cannot_judge (nothing to compare) are left out of n_reviewed and
-- counted on their own in n_cannot_judge.
SELECT count(*) FILTER (WHERE final_label IN ('same', 'different'))   AS n_reviewed,
       count(*) FILTER (WHERE final_label = 'same')                   AS n_confirmed,
       CASE WHEN count(*) FILTER (WHERE final_label IN ('same', 'different')) >= 10
            THEN round(count(*) FILTER (WHERE final_label = 'same')
                       / count(*) FILTER (WHERE final_label IN ('same', 'different')), 4)
            END                                                       AS precision,
       count(*) FILTER (WHERE final_label = 'cannot_judge')           AS n_cannot_judge
FROM matches_reviewed
WHERE label = 'same' AND review_status = 'reviewed';
