-- 1) Reason-coded rejects: rows that must not reach reporting, kept for follow-up.
CREATE OR REPLACE TABLE planning.plan_rejects AS
SELECT session_id,
       CASE
           WHEN segment IS NULL                     THEN 'MISSING_SEGMENT'
           WHEN capacity IS NULL OR capacity <= 0   THEN 'INVALID_CAPACITY'
           WHEN net_appointments > capacity             THEN 'APPOINTMENTS_ABOVE_CAPACITY'
           WHEN benchmark_support < 20              THEN 'THIN_BENCHMARK'
       END AS reason_code
FROM planning.vw_session_plan
WHERE segment IS NULL OR capacity IS NULL OR capacity <= 0
   OR net_appointments > capacity OR benchmark_support < 20;

-- 2) Control totals: published rows + rejects must equal the open-session population.
SELECT (SELECT COUNT(*) FROM planning.sessions WHERE is_open)          AS expected_rows,
       (SELECT COUNT(*) FROM planning.vw_session_plan)                  AS plan_rows,
       (SELECT COUNT(*) FROM planning.plan_rejects)                       AS reject_rows,
       expected_rows = plan_rows                                          AS reconciled;

-- 3) Allocation reconciliation: outreach plans must add up to each session's gap.
SELECT p.session_id, p.gap_to_target, SUM(a.outreach_units) AS allocated,
       ABS(p.gap_to_target - SUM(a.outreach_units)) <= 0.5      AS ok
FROM planning.vw_session_plan p
JOIN planning.outreach_plan a ON a.session_id = p.session_id
GROUP BY p.session_id, p.gap_to_target
HAVING NOT ok;          -- should return zero rows
