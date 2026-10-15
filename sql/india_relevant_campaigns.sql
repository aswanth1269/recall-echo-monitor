-- india_relevant_campaigns: NHTSA campaigns with at least one model whose
-- family is one of that make's India families.
--
-- Inputs : nhtsa_focus, nhtsa_model_family, nhtsa_campaigns,
--          india_families (make, family) registered from model_families.csv
-- First  : "first non-null value per campaign" = the value on the lowest
--          RECORD_ID among records where that field is not NULL.
-- llm_only: TRUE when every India family of the campaign comes only from
--          NHTSA models mapped by the LLM (unconfirmed; see
--          data/clean/model_family_review.csv).
CREATE OR REPLACE TABLE india_relevant_campaigns AS
WITH model_fams AS (
    SELECT DISTINCT f.CAMPNO, f.make, m.family, m.method
    FROM nhtsa_focus AS f
    JOIN nhtsa_model_family AS m USING (make, MODELTXT)
    JOIN india_families AS i ON i.make = m.make AND i.family = m.family
    WHERE m.family IS NOT NULL
),
fam_flags AS (          -- a family is llm-only if no rule/override model gave it
    SELECT CAMPNO, make, family, bool_and(method = 'llm') AS family_llm_only
    FROM model_fams GROUP BY CAMPNO, make, family
),
camp_fams AS (
    SELECT CAMPNO, make,
           list_sort(list(family))  AS families,
           bool_and(family_llm_only) AS llm_only
    FROM fam_flags GROUP BY CAMPNO, make
),
firsts AS (
    SELECT CAMPNO,
           arg_min(COMPNAME, CAST(RECORD_ID AS BIGINT))
               FILTER (WHERE COMPNAME IS NOT NULL)            AS COMPNAME,
           arg_min(DESC_DEFECT, CAST(RECORD_ID AS BIGINT))
               FILTER (WHERE DESC_DEFECT IS NOT NULL)         AS DESC_DEFECT,
           arg_min(CONEQUENCE_DEFECT, CAST(RECORD_ID AS BIGINT))
               FILTER (WHERE CONEQUENCE_DEFECT IS NOT NULL)   AS CONEQUENCE_DEFECT,
           arg_min(CORRECTIVE_ACTION, CAST(RECORD_ID AS BIGINT))
               FILTER (WHERE CORRECTIVE_ACTION IS NOT NULL)   AS CORRECTIVE_ACTION,
           list_sort(list(DISTINCT COMPNAME) FILTER (WHERE COMPNAME IS NOT NULL)) AS components
    FROM nhtsa_focus
    GROUP BY CAMPNO
)
SELECT c.CAMPNO, c.make, n.report_date, c.families, c.llm_only,
       f.COMPNAME, f.components, f.DESC_DEFECT, f.CONEQUENCE_DEFECT, f.CORRECTIVE_ACTION
FROM camp_fams AS c
JOIN nhtsa_campaigns AS n USING (CAMPNO)
JOIN firsts AS f USING (CAMPNO)
ORDER BY c.make, c.CAMPNO;
