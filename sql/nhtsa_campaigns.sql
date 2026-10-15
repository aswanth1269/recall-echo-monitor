-- nhtsa_campaigns: one row per NHTSA campaign number (CAMPNO) in nhtsa_focus,
-- keeping which models and model years it affects.
--
-- A source record is one make/model/year inside a campaign, so a campaign
-- spans many records. Lists are de-duplicated and sorted. Unknown model
-- years (NULL) are left out of model_years and counted in
-- n_records_unknown_year instead of being guessed. A list is NULL when
-- every value it would hold is unknown.
CREATE OR REPLACE TABLE nhtsa_campaigns AS
SELECT
    CAMPNO,
    list_sort(list(DISTINCT make))                                   AS makes,
    min(report_date)                                                 AS report_date,
    count(*)                                                         AS n_records,
    list_sort(list(DISTINCT MODELTXT) FILTER (WHERE MODELTXT IS NOT NULL))   AS models,
    list_sort(list(DISTINCT model_year) FILTER (WHERE model_year IS NOT NULL)) AS model_years,
    list_sort(list(DISTINCT struct_pack(model := MODELTXT, model_year := model_year))) AS model_year_pairs,
    count(*) FILTER (WHERE model_year IS NULL)                       AS n_records_unknown_year
FROM nhtsa_focus
GROUP BY CAMPNO
ORDER BY CAMPNO;
