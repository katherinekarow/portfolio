-- model: rpt.pace_dashboard
-- description: Reporting view behind the pace dashboard - one row per open session with friendly, percentage-based columns and a priority rank.
-- grain: one row per open session
-- owner: operations analytics
-- column session_id: session identifier
-- column segment: demand segment
-- column status: on pace, watch or behind
-- column booked_pct: net appointments / capacity
-- column projected_pct: projected final / capacity
-- column gap_to_target: units short of target
-- column priority_rank: 1 = largest gap to target
-- test: unique(session_id)
-- test: unique(priority_rank)
-- test: expression(booked_pct <= projected_pct)
SELECT session_id,
       segment,
       status,
       ROUND(1.0 * net_appointments / capacity, 3)      AS booked_pct,
       ROUND(1.0 * projected_final / capacity, 3)   AS projected_pct,
       gap_to_target,
       ROW_NUMBER() OVER (ORDER BY gap_to_target DESC, session_id) AS priority_rank
FROM mart.session_plan
