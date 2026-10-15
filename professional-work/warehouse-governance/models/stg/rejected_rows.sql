-- model: stg.rejected_rows
-- description: Raw rows that failed staging rules, with a reason code so they can be fixed at the source instead of silently dropped.
-- grain: one row per rejected raw row
-- owner: analytics engineering
-- column source_table: table the row came from
-- column source_key: key of the rejected row
-- column reason_code: MISSING_SESSION, INVALID_UNITS or UNKNOWN_EVENT_TYPE
-- test: unique(source_table, source_key)
-- test: not_null(source_key, reason_code)
-- test: accepted_values(reason_code: MISSING_SESSION, INVALID_UNITS, UNKNOWN_EVENT_TYPE)
SELECT 'raw.appointments_feed' AS source_table,
       event_id            AS source_key,
       CASE
           WHEN TRIM(COALESCE(session_code, '')) = ''           THEN 'MISSING_SESSION'
           WHEN units IS NULL OR units <= 0                        THEN 'INVALID_UNITS'
           WHEN UPPER(TRIM(event_type)) NOT IN ('NEW', 'CANCEL')   THEN 'UNKNOWN_EVENT_TYPE'
       END                 AS reason_code
FROM raw.appointments_feed
WHERE TRIM(COALESCE(session_code, '')) = ''
   OR units IS NULL OR units <= 0
   OR UPPER(TRIM(event_type)) NOT IN ('NEW', 'CANCEL')
