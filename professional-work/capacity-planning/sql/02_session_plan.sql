-- Position, projection and gap decomposition for every open session.
CREATE OR REPLACE VIEW planning.vw_session_plan AS
WITH today AS (
    SELECT d.session_id, d.segment, d.capacity, d.days_out_today, b.net_appointments, b.gross_appointments
    FROM planning.sessions d
    JOIN planning.appointments_daily b
      ON b.session_id = d.session_id AND b.days_out = d.days_out_today
    WHERE d.is_open
),
finals AS (
    SELECT segment, target_fill AS target_final_fill, median_fill AS median_final_fill
    FROM planning.vw_segment_curves WHERE days_out = 0
),
joined AS (
    SELECT t.*, c.target_fill, c.median_share, c.n_sessions, f.target_final_fill, f.median_final_fill,
           LEAST(c.median_share / 0.30, 1)                       AS credibility,      -- little data -> lean on history
           t.net_appointments / GREATEST(c.median_share, 0.02)       AS pickup_projection
    FROM today t
    JOIN planning.vw_segment_curves c ON c.segment = t.segment AND c.days_out = t.days_out_today
    JOIN finals f                     ON f.segment = t.segment
)
SELECT session_id,
       segment,
       capacity,
       days_out_today,
       net_appointments,
       ROUND(GREATEST(LEAST(credibility * pickup_projection + (1 - credibility) * median_final_fill * capacity,
                            capacity), net_appointments))                          AS projected_final,
       ROUND(target_fill * capacity)                                             AS target_today,
       ROUND(target_final_fill * capacity)                                       AS target_final,
       ROUND(target_fill * capacity) - net_appointments                              AS pace_gap_today,
       GREATEST(ROUND(target_final_fill * capacity) - projected_final, 0)        AS gap_to_target,
       capacity - projected_final                                              AS gap_to_full,
       n_sessions                                                            AS benchmark_support
FROM joined;
