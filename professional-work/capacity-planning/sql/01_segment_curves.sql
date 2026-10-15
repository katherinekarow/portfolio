-- Benchmarks per segment and days-to-session (days_out) from completed sessions.
CREATE OR REPLACE VIEW planning.vw_segment_curves AS
WITH finals AS (
    SELECT session_id, net_appointments AS final_net
    FROM planning.appointments_daily
    WHERE days_out = 0
),
history AS (
    SELECT d.segment,
           b.days_out,
           b.session_id,
           b.net_appointments / d.capacity                          AS fill_rate,
           b.net_appointments / NULLIF(f.final_net, 0)              AS share_of_final,
           IFF(b.gross_appointments >= 20, b.net_appointments / b.gross_appointments, NULL) AS net_to_gross
    FROM planning.appointments_daily b
    JOIN planning.sessions d ON d.session_id = b.session_id AND NOT d.is_open
    JOIN finals f              ON f.session_id = b.session_id
),
stats AS (
    SELECT segment,
           days_out,
           PERCENTILE_CONT(0.60) WITHIN GROUP (ORDER BY fill_rate)    AS target_fill_raw,
           MEDIAN(fill_rate)                                          AS median_fill,
           MEDIAN(share_of_final)                                       AS median_share_raw,
           STDDEV(share_of_final)                                       AS share_std,
           MEDIAN(net_to_gross)                                         AS survival,
           COUNT(DISTINCT session_id)                                 AS n_sessions
    FROM history
    GROUP BY segment, days_out
)
SELECT segment,
       days_out,
       -- a target can never fall as session gets closer
       LEAST(MAX(target_fill_raw)   OVER (PARTITION BY segment ORDER BY days_out DESC ROWS UNBOUNDED PRECEDING), 1) AS target_fill,
       LEAST(MAX(median_share_raw) OVER (PARTITION BY segment ORDER BY days_out DESC ROWS UNBOUNDED PRECEDING), 1) AS median_share,
       median_fill,
       share_std,
       survival,
       1 - survival AS cancel_rate,
       n_sessions
FROM stats;
