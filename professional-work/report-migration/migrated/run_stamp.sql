-- migrated from legacy/run_stamp.sql by migrationkit (snowflake)
-- line 3: TRUNC(date) -> DATE_TRUNC
-- line 2: SYSDATE -> CURRENT_TIMESTAMP
-- line 3: SYSDATE -> CURRENT_TIMESTAMP
-- line 4: FROM DUAL removed
SELECT TO_CHAR(CURRENT_TIMESTAMP(), 'YYYY-MM-DD') AS run_date,
       DATE_TRUNC('MONTH', CURRENT_TIMESTAMP()) AS month_start
