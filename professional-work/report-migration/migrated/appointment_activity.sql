-- migrated from legacy/appointment_activity.sql by migrationkit (snowflake)
-- line 2: optimizer hint removed
-- line 4: TRUNC(date) -> DATE_TRUNC
-- line 5: DECODE -> CASE (NULL-safe)
-- line 6: NVL -> COALESCE
-- line 8: NVL -> COALESCE
-- line 12: ADD_MONTHS -> DATEADD
-- line 14: TRUNC(date) -> DATE_TRUNC
-- line 15: DECODE -> CASE (NULL-safe)
-- line 16: NVL -> COALESCE
-- line 11: (+) outer join -> LEFT JOIN
SELECT d.session_id,
       DATE_TRUNC('MONTH', b.appointment_date)                                                   AS appointment_month,
       CASE WHEN b.status = 'A' THEN 'Active' WHEN b.status = 'C' THEN 'Cancelled' WHEN b.status IS NULL THEN 'Unknown' ELSE 'Other' END    AS status_label,
       COALESCE(a.channel, 'DIRECT')                                                      AS channel,
       SUM(b.units)                                                                  AS units,
       SUM(COALESCE(b.charge, 0) * b.units)                                                 AS revenue
FROM sessions d
INNER JOIN appointments b ON b.session_id = d.session_id
LEFT JOIN referrers a ON b.referrer_id = a.referrer_id
WHERE d.session_date >= DATEADD(MONTH, -12, TO_DATE('2026-03-01', 'YYYY-MM-DD'))
GROUP BY d.session_id,
         DATE_TRUNC('MONTH', b.appointment_date),
         CASE WHEN b.status = 'A' THEN 'Active' WHEN b.status = 'C' THEN 'Cancelled' WHEN b.status IS NULL THEN 'Unknown' ELSE 'Other' END,
         COALESCE(a.channel, 'DIRECT')
