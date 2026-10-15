-- model: core.appointments_daily
-- description: Cumulative gross and net appointments for every session at every days-to-session point that has already happened.
-- grain: one row per session and days_out
-- owner: analytics engineering
-- column session_id: session identifier
-- column days_out: days to session (365 ... 0)
-- column gross_appointments: cumulative units booked, including later cancellations
-- column net_appointments: cumulative units booked minus cancelled
-- test: unique(session_id, days_out)
-- test: not_null(session_id, days_out, gross_appointments, net_appointments)
-- test: expression(net_appointments <= gross_appointments)
-- test: relationship(session_id -> core.sessions.session_id)
WITH RECURSIVE spine AS (
    SELECT 365 AS days_out
    UNION ALL
    SELECT days_out - 1 AS days_out FROM spine WHERE days_out > 0
),
daily AS (
    SELECT b.session_id,
           CAST(JULIANDAY(d.session_date) - JULIANDAY(b.event_date) AS INTEGER) AS days_out,
           SUM(CASE WHEN b.event_type = 'NEW' THEN b.units ELSE 0 END)           AS new_units,
           SUM(CASE WHEN b.event_type = 'CANCEL' THEN b.units ELSE 0 END)        AS cancelled_units
    FROM stg.appointments_clean AS b
    INNER JOIN core.sessions AS d ON d.session_id = b.session_id
    GROUP BY b.session_id, days_out
),
grid AS (
    SELECT d.session_id, s.days_out
    FROM core.sessions AS d
    INNER JOIN spine AS s ON s.days_out >= MAX(d.days_out_today, 0)
)
SELECT g.session_id,
       g.days_out,
       SUM(COALESCE(x.new_units, 0)) OVER w                                AS gross_appointments,
       SUM(COALESCE(x.new_units, 0) - COALESCE(x.cancelled_units, 0)) OVER w AS net_appointments
FROM grid AS g
LEFT JOIN daily AS x ON x.session_id = g.session_id AND x.days_out = g.days_out
WINDOW w AS (PARTITION BY g.session_id ORDER BY g.days_out DESC ROWS UNBOUNDED PRECEDING)
