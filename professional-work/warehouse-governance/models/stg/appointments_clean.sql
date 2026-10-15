-- model: stg.appointments_clean
-- description: Appointment events with standardized session ids; re-sent duplicates collapsed and invalid rows removed (see stg.rejected_rows).
-- grain: one row per appointment event
-- owner: analytics engineering
-- column event_id: unique id of the event
-- column appointment_id: reservation id
-- column session_id: session id, trimmed and upper-cased
-- column event_type: NEW or CANCEL
-- column units: units affected, always positive
-- column event_date: date of the event
-- test: unique(event_id)
-- test: not_null(event_id, session_id, event_type, units, event_date)
-- test: accepted_values(event_type: NEW, CANCEL)
-- test: expression(units > 0)
-- test: relationship(session_id -> core.sessions.session_id)
WITH ranked AS (
    SELECT event_id,
           appointment_id,
           UPPER(TRIM(session_code)) AS session_id,
           UPPER(TRIM(event_type))     AS event_type,
           units,
           DATE(event_ts)              AS event_date,
           ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_ts DESC) AS version_rank
    FROM raw.appointments_feed
)
SELECT event_id,
       appointment_id,
       session_id,
       event_type,
       units,
       event_date
FROM ranked
WHERE version_rank = 1
  AND session_id <> ''
  AND units > 0
  AND event_type IN ('NEW', 'CANCEL')
