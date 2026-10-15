-- model: mart.session_plan
-- description: Position, credibility-weighted projection and gap decomposition for every open session, with a status of on pace, watch or behind.
-- grain: one row per open session
-- owner: operations analytics
-- column session_id: session identifier
-- column segment: demand segment
-- column capacity: sellable units
-- column days_out_today: days to session today
-- column net_appointments: net appointments today
-- column projected_final: projected net appointments at session
-- column target_today: target appointments for today
-- column target_final: target appointments at session
-- column pace_gap_today: target today minus appointments today (positive = behind)
-- column gap_to_target: target at session minus projection, never negative
-- column gap_to_full: capacity minus projection
-- column status: on pace, watch or behind
-- test: unique(session_id)
-- test: not_null(session_id, projected_final, status)
-- test: accepted_values(status: on pace, watch, behind)
-- test: expression(projected_final BETWEEN net_appointments AND capacity)
-- test: expression(gap_to_target >= 0)
-- test: assert_empty(SELECT 1 WHERE (SELECT COUNT(*) FROM core.sessions WHERE is_open = 1) <> (SELECT COUNT(*) FROM mart.session_plan))
WITH today AS (
    SELECT d.session_id,
           d.segment,
           d.capacity,
           d.days_out_today,
           b.net_appointments
    FROM core.sessions AS d
    INNER JOIN core.appointments_daily AS b ON b.session_id = d.session_id AND b.days_out = d.days_out_today
    WHERE d.is_open = 1
),
finals AS (
    SELECT segment,
           target_fill AS target_final_fill,
           median_fill AS median_final_fill
    FROM mart.segment_curves
    WHERE days_out = 0
),
projected AS (
    SELECT t.session_id,
           t.segment,
           t.capacity,
           t.days_out_today,
           t.net_appointments,
           c.target_fill,
           f.target_final_fill,
           ROUND(MAX(MIN(MIN(c.median_share / 0.30, 1.0) * t.net_appointments / MAX(c.median_share, 0.02)
                         + (1 - MIN(c.median_share / 0.30, 1.0)) * f.median_final_fill * t.capacity,
                         t.capacity),
                     t.net_appointments)) AS projected_final
    FROM today AS t
    INNER JOIN mart.segment_curves AS c ON c.segment = t.segment AND c.days_out = t.days_out_today
    INNER JOIN finals AS f ON f.segment = t.segment
),
gaps AS (
    SELECT session_id,
           segment,
           capacity,
           days_out_today,
           net_appointments,
           projected_final,
           ROUND(target_fill * capacity)                                   AS target_today,
           ROUND(target_final_fill * capacity)                             AS target_final,
           ROUND(target_fill * capacity) - net_appointments                    AS pace_gap_today,
           MAX(ROUND(target_final_fill * capacity) - projected_final, 0)   AS gap_to_target,
           capacity - projected_final                                    AS gap_to_full
    FROM projected
)
SELECT session_id,
       segment,
       capacity,
       days_out_today,
       net_appointments,
       projected_final,
       target_today,
       target_final,
       pace_gap_today,
       gap_to_target,
       gap_to_full,
       CASE
           WHEN pace_gap_today <= 0 AND gap_to_target <= 0.01 * capacity      THEN 'on pace'
           WHEN pace_gap_today <= 0.03 * capacity AND gap_to_target <= 0.05 * capacity THEN 'watch'
           ELSE 'behind'
       END AS status
FROM gaps
