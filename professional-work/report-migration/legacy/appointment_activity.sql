-- Legacy report: monthly appointment activity by session, status and referral channel (Oracle)
SELECT /*+ INDEX(b appt_sess_idx) */
       d.session_id,
       TRUNC(b.appointment_date, 'MM')                                                   AS appointment_month,
       DECODE(b.status, 'A', 'Active', 'C', 'Cancelled', NULL, 'Unknown', 'Other')    AS status_label,
       NVL(a.channel, 'DIRECT')                                                      AS channel,
       SUM(b.units)                                                                  AS units,
       SUM(NVL(b.charge, 0) * b.units)                                                 AS revenue
FROM sessions d, appointments b, referrers a
WHERE b.session_id = d.session_id
  AND b.referrer_id = a.referrer_id(+)
  AND d.session_date >= ADD_MONTHS(TO_DATE('2026-03-01', 'YYYY-MM-DD'), -12)
GROUP BY d.session_id,
         TRUNC(b.appointment_date, 'MM'),
         DECODE(b.status, 'A', 'Active', 'C', 'Cancelled', NULL, 'Unknown', 'Other'),
         NVL(a.channel, 'DIRECT')
