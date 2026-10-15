-- migrated from legacy/referrer_hierarchy.sql by migrationkit (snowflake)
-- line 5: TRUNC(date) -> DATE_TRUNC
-- line 5: SYSDATE -> CURRENT_TIMESTAMP
-- REVIEW: line 8: CONNECT BY hierarchy - rewrite as a recursive CTE
-- REVIEW: line 8: PRIOR (hierarchical query) - rewrite as a recursive CTE
SELECT LEVEL AS depth,
       a.referrer_id AS referrer_id,
       a.referrer_name AS referrer_name,
       DATE_TRUNC('DAY', CURRENT_TIMESTAMP()) AS snapshot_date
FROM referrers a
START WITH a.parent_referrer_id IS NULL
CONNECT BY PRIOR a.referrer_id = a.parent_referrer_id
