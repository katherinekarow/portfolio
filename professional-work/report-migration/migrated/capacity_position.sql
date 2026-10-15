-- migrated from legacy/capacity_position.sql by migrationkit (snowflake)
-- line 6: DECODE -> CASE (NULL-safe)
-- line 7: DECODE -> CASE (NULL-safe)
-- line 8: NVL2 -> CASE
-- line 10: (+) outer join -> LEFT JOIN
-- column 2: renamed duplicate 'session_id' to 'b_session_id'
-- column 4: added alias 'sum_status_units'
-- column 5: added alias 'capacity_minus_sum_status'
SELECT d.session_id,
       b.session_id AS b_session_id,
       d.capacity,
       SUM(CASE WHEN b.status = 'C' THEN 0 ELSE b.units END) AS sum_status_units,
       d.capacity - SUM(CASE WHEN b.status = 'C' THEN 0 ELSE b.units END) AS capacity_minus_sum_status,
       CASE WHEN MAX(b.referrer_id) IS NOT NULL THEN 'REFERRED' ELSE 'DIRECT' END AS referral_mix
FROM sessions d
LEFT JOIN appointments b ON d.session_id = b.session_id
GROUP BY d.session_id, b.session_id, d.capacity
