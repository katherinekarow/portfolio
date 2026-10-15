-- model: core.sessions
-- description: Conformed session dimension with segment assignment and days to session as of the run date.
-- grain: one row per session
-- owner: analytics engineering
-- column session_id: session identifier
-- column session_date: date of session
-- column segment: region / season / length used for benchmarks
-- column capacity: sellable units
-- column is_open: 1 while the session is still taking appointments
-- column days_out_today: days to session on the run date (negative once held)
-- test: unique(session_id)
-- test: not_null(session_id, session_date, segment, capacity)
-- test: expression(capacity > 0)
-- test: accepted_values(is_open: 0, 1)
SELECT UPPER(TRIM(m.session_code))                                   AS session_id,
       m.session_date,
       m.region || ' / ' || m.season || ' / ' || m.length_class        AS segment,
       m.capacity,
       CASE WHEN m.session_date > r.as_of_date THEN 1 ELSE 0 END      AS is_open,
       CAST(JULIANDAY(m.session_date) - JULIANDAY(r.as_of_date) AS INTEGER) AS days_out_today
FROM raw.session_master AS m
CROSS JOIN raw.run_context AS r
