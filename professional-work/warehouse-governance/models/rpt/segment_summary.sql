-- model: rpt.segment_summary
-- description: Segment roll-up for leadership - open sessions, booked and projected fill rate, total gap to target and status counts.
-- grain: one row per segment
-- owner: operations analytics
-- column segment: demand segment
-- column open_sessions: sessions still taking appointments
-- column booked_fill: booked units / capacity
-- column projected_fill: projected units / capacity
-- column gap_to_target: total units short of target
-- column behind_count: sessions with status behind
-- test: unique(segment)
-- test: assert_empty(SELECT 1 WHERE (SELECT SUM(gap_to_target) FROM rpt.segment_summary) <> (SELECT SUM(gap_to_target) FROM mart.session_plan))
SELECT segment,
       COUNT(*)                                              AS open_sessions,
       ROUND(1.0 * SUM(net_appointments) / SUM(capacity), 3)     AS booked_fill,
       ROUND(1.0 * SUM(projected_final) / SUM(capacity), 3)  AS projected_fill,
       SUM(gap_to_target)                                    AS gap_to_target,
       SUM(CASE WHEN status = 'behind' THEN 1 ELSE 0 END)    AS behind_count
FROM mart.session_plan
GROUP BY segment
