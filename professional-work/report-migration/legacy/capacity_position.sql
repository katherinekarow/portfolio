-- Legacy report: slot position per session (Oracle). Columns were never aliased;
-- the BI layer named them positionally, which broke when the column order changed.
SELECT d.session_id,
       b.session_id,
       d.capacity,
       SUM(DECODE(b.status, 'C', 0, b.units)),
       d.capacity - SUM(DECODE(b.status, 'C', 0, b.units)),
       NVL2(MAX(b.referrer_id), 'REFERRED', 'DIRECT') referral_mix
FROM sessions d, appointments b
WHERE d.session_id = b.session_id(+)
GROUP BY d.session_id, b.session_id, d.capacity
