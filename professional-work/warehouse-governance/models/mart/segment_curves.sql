-- model: mart.segment_curves
-- description: Benchmarks from completed sessions per segment and days_out - 60th-percentile target fill rate (never decreasing toward session) and the median share of final appointments.
-- grain: one row per segment and days_out
-- owner: operations analytics
-- column segment: demand segment
-- column days_out: days to session
-- column target_fill: target fill rate, running maximum toward session, capped at 1
-- column median_fill: median fill rate of completed sessions
-- column median_share: median share of final appointments already on the books, running maximum
-- column n_sessions: completed sessions supporting the benchmark
-- test: unique(segment, days_out)
-- test: not_null(segment, days_out, target_fill, median_share)
-- test: expression(target_fill BETWEEN 0 AND 1)
-- test: expression(median_share BETWEEN 0 AND 1)
-- test: assert_empty(SELECT a.segment FROM mart.segment_curves AS a INNER JOIN mart.segment_curves AS b ON a.segment = b.segment AND b.days_out = a.days_out - 1 WHERE b.target_fill < a.target_fill)
WITH finals AS (
    SELECT session_id,
           net_appointments AS final_net
    FROM core.appointments_daily
    WHERE days_out = 0
),
history AS (
    SELECT d.segment,
           b.days_out,
           1.0 * b.net_appointments / d.capacity              AS fill_rate,
           1.0 * b.net_appointments / NULLIF(f.final_net, 0)  AS share_of_final
    FROM core.appointments_daily AS b
    INNER JOIN core.sessions AS d ON d.session_id = b.session_id AND d.is_open = 0
    INNER JOIN finals AS f ON f.session_id = b.session_id
),
ranked AS (
    SELECT segment,
           days_out,
           fill_rate,
           share_of_final,
           ROW_NUMBER() OVER (PARTITION BY segment, days_out ORDER BY fill_rate)    AS fill_rank,
           ROW_NUMBER() OVER (PARTITION BY segment, days_out ORDER BY share_of_final) AS share_rank,
           COUNT(*) OVER (PARTITION BY segment, days_out)                             AS n
    FROM history
),
stats AS (
    SELECT segment,
           days_out,
           n,
           MAX(CASE WHEN fill_rank = CAST(0.6 * n + 0.999999 AS INTEGER) THEN fill_rate END) AS target_fill_raw,
           MAX(CASE WHEN fill_rank = (n + 1) / 2 THEN fill_rate END)                         AS median_fill,
           MAX(CASE WHEN share_rank = (n + 1) / 2 THEN share_of_final END)                   AS median_share_raw
    FROM ranked
    GROUP BY segment, days_out, n
)
SELECT segment,
       days_out,
       MIN(MAX(target_fill_raw) OVER climb, 1.0)    AS target_fill,
       median_fill,
       MIN(MAX(median_share_raw) OVER climb, 1.0) AS median_share,
       n                                          AS n_sessions
FROM stats
WINDOW climb AS (PARTITION BY segment ORDER BY days_out DESC ROWS UNBOUNDED PRECEDING)
