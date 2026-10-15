-- migrated from legacy/recent_cancellations.sql by migrationkit (snowflake)
-- line 8: ROWNUM -> LIMIT
-- REVIEW: line 8: ROWNUM with ORDER BY - Oracle picks the rows BEFORE sorting; LIMIT sorts first. Confirm which result the report intended.
SELECT b.appointment_id AS appointment_id,
       b.appointment_date AS appointment_date,
       TO_CHAR(b.appointment_date, 'YYYY-MM') AS appointment_month
FROM appointments b
WHERE b.status = 'C'
ORDER BY b.appointment_date DESC
LIMIT 10
