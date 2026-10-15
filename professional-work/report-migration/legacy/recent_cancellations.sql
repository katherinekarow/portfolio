-- Legacy report: "latest 10 cancellations" - but ROWNUM is applied before ORDER BY in Oracle,
-- so this has always returned 10 arbitrary cancellations, sorted. The migration has to decide what was meant.
SELECT b.appointment_id AS appointment_id,
       b.appointment_date AS appointment_date,
       TO_CHAR(b.appointment_date, 'YYYY-MM') AS appointment_month
FROM appointments b
WHERE b.status = 'C'
  AND ROWNUM <= 10
ORDER BY b.appointment_date DESC
