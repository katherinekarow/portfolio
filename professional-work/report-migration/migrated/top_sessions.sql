-- migrated from legacy/top_sessions.sql by migrationkit (snowflake)
-- line 5: NVL -> COALESCE
-- line 7: NVL -> COALESCE
-- line 11: ROWNUM -> LIMIT
SELECT session_id, revenue
FROM (
    SELECT b.session_id AS session_id,
           SUM(COALESCE(b.charge, 0) * b.units) AS revenue
    FROM appointments b
    WHERE COALESCE(b.status, 'A') <> 'C'
    GROUP BY b.session_id
    ORDER BY revenue DESC, session_id
)
LIMIT 5
