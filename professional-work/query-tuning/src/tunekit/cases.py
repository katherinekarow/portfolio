"""Tuning cases: a slow (or wrong) query as typically found in legacy reports, and the rewrite.

Each case lists the indexes that exist for BOTH versions (`shared_indexes`) and the ones the
rewrite adds (`fast_indexes`), so the benchmark separates "rewrite the SQL" from "add an index".
`same_result=False` marks a case where the slow version was also returning the WRONG answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    name: str
    title: str
    slow_sql: str
    fast_sql: str
    lesson: str
    alt_sql: str = ""
    alt_label: str = ""
    shared_indexes: list[str] = field(default_factory=list)
    fast_indexes: list[str] = field(default_factory=list)
    same_result: bool = True


CASES = [
    Case(
        name="sargable_dates",
        title="Function on an indexed column",
        slow_sql="""
            SELECT COUNT(*) AS appointments, SUM(units) AS units
            FROM appointments
            WHERE STRFTIME('%Y-%m', booked_ts) = '2025-03'""",
        fast_sql="""
            SELECT COUNT(*) AS appointments, SUM(units) AS units
            FROM appointments
            WHERE booked_ts >= '2025-03-01' AND booked_ts < '2025-04-01'""",
        shared_indexes=["CREATE INDEX ix_appts_ts_units ON appointments (booked_ts, units)"],
        lesson="Wrapping a column in a function hides it from the index's ordering, so all 1M entries are read (plan: SCAN). A range on the raw column "
               "uses the index, and because the index also holds `units`, the table is never touched (covering index).",
    ),
    Case(
        name="latest_status",
        title="Latest status per appointment: the index beats the rewrite",
        slow_sql="""
            SELECT b.session_id,
                   SUM(CASE WHEN (SELECT e.status FROM appointment_events AS e
                                  WHERE e.appointment_id = b.appointment_id
                                  ORDER BY e.event_ts DESC, e.event_id DESC LIMIT 1) = 'CANCELLED'
                            THEN 1 ELSE 0 END) AS cancelled_appointments
            FROM appointments AS b
            WHERE b.session_id < 200
            GROUP BY b.session_id""",
        fast_sql="""
            SELECT b.session_id,
                   SUM(CASE WHEN (SELECT e.status FROM appointment_events AS e
                                  WHERE e.appointment_id = b.appointment_id
                                  ORDER BY e.event_ts DESC, e.event_id DESC LIMIT 1) = 'CANCELLED'
                            THEN 1 ELSE 0 END) AS cancelled_appointments
            FROM appointments AS b
            WHERE b.session_id < 200
            GROUP BY b.session_id""",
        alt_sql="""
            WITH last_event AS (
                SELECT e.appointment_id,
                       e.status,
                       ROW_NUMBER() OVER (PARTITION BY e.appointment_id ORDER BY e.event_ts DESC, e.event_id DESC) AS rn
                FROM appointment_events AS e
                INNER JOIN appointments AS b ON b.appointment_id = e.appointment_id AND b.session_id < 200
            )
            SELECT b.session_id,
                   SUM(CASE WHEN l.status = 'CANCELLED' THEN 1 ELSE 0 END) AS cancelled_appointments
            FROM appointments AS b
            INNER JOIN last_event AS l ON l.appointment_id = b.appointment_id AND l.rn = 1
            WHERE b.session_id < 200
            GROUP BY b.session_id""",
        alt_label="window-function rewrite (same indexes as fast)",
        shared_indexes=["CREATE INDEX ix_appts_sess ON appointments (session_id)",
                        "CREATE INDEX ix_events_appt ON appointment_events (appointment_id)"],
        fast_indexes=["CREATE INDEX ix_events_appt_ts ON appointment_events (appointment_id, event_ts, event_id, status)"],
        lesson="With only the foreign-key index, each of ~100k correlated lookups fetches every event row and sorts them. "
               "A composite index in (appointment_id, event_ts, event_id) order that also carries `status` answers the same "
               "query straight from the index, already sorted. The 'obvious' rewrite to a window function was measured too: "
               "in this engine it was slower than the indexed correlated query. Measure, don't assume.",
    ),
    Case(
        name="conditional_aggregation",
        title="One scalar subquery per column → one GROUP BY",
        slow_sql="""
            SELECT d.session_id,
                   (SELECT COALESCE(SUM(b.units), 0) FROM appointments AS b INNER JOIN referrers AS a ON a.referrer_id = b.referrer_id
                     WHERE b.session_id = d.session_id AND a.channel = 'ONLINE')  AS online_units,
                   (SELECT COALESCE(SUM(b.units), 0) FROM appointments AS b INNER JOIN referrers AS a ON a.referrer_id = b.referrer_id
                     WHERE b.session_id = d.session_id AND a.channel = 'PHONE')  AS phone_units,
                   (SELECT COALESCE(SUM(b.units), 0) FROM appointments AS b INNER JOIN referrers AS a ON a.referrer_id = b.referrer_id
                     WHERE b.session_id = d.session_id AND a.channel = 'PARTNER') AS partner_units,
                   (SELECT COALESCE(SUM(b.units), 0) FROM appointments AS b
                     WHERE b.session_id = d.session_id AND b.referrer_id IS NULL)    AS direct_units
            FROM sessions AS d
            WHERE d.region = 'north'""",
        fast_sql="""
            SELECT d.session_id,
                   COALESCE(SUM(CASE WHEN a.channel = 'ONLINE'  THEN b.units END), 0) AS online_units,
                   COALESCE(SUM(CASE WHEN a.channel = 'PHONE'  THEN b.units END), 0) AS phone_units,
                   COALESCE(SUM(CASE WHEN a.channel = 'PARTNER' THEN b.units END), 0) AS partner_units,
                   COALESCE(SUM(CASE WHEN b.referrer_id IS NULL    THEN b.units END), 0) AS direct_units
            FROM sessions AS d
            LEFT JOIN appointments AS b ON b.session_id = d.session_id
            LEFT JOIN referrers AS a ON a.referrer_id = b.referrer_id
            WHERE d.region = 'north'
            GROUP BY d.session_id""",
        shared_indexes=["CREATE INDEX ix_appts_sess ON appointments (session_id)"],
        fast_indexes=["CREATE INDEX ix_appts_sess_cover ON appointments (session_id, referrer_id, units)"],
        lesson="Four subqueries mean four trips to the appointments table per session. A single join with conditional "
               "aggregation reads each appointment once, and a covering index means it never touches the table rows.",
    ),
    Case(
        name="fan_out_join",
        title="Join fan-out double counts → semi-join (EXISTS)",
        slow_sql="""
            SELECT d.region,
                   ROUND(SUM(b.charge * b.units), 2) AS rescheduled_appointment_charges
            FROM appointments AS b
            INNER JOIN appointment_events AS e ON e.appointment_id = b.appointment_id
            INNER JOIN sessions AS d ON d.session_id = b.session_id
            WHERE e.status = 'RESCHEDULED'
            GROUP BY d.region""",
        fast_sql="""
            SELECT d.region,
                   ROUND(SUM(b.charge * b.units), 2) AS rescheduled_appointment_charges
            FROM appointments AS b
            INNER JOIN sessions AS d ON d.session_id = b.session_id
            WHERE EXISTS (SELECT 1 FROM appointment_events AS e WHERE e.appointment_id = b.appointment_id AND e.status = 'RESCHEDULED')
            GROUP BY d.region""",
        fast_indexes=["CREATE INDEX ix_events_appt_status ON appointment_events (appointment_id, status)"],
        same_result=False,
        lesson="An appointment rescheduled twice joins to two events and its charge is counted twice. The report "
               "was overstated. EXISTS asks the question that was meant (was it ever rescheduled?), counts each appointment once, "
               "and with an index on (appointment_id, status) it is also faster.",
    ),
    Case(
        name="covering_join",
        title="Join without a useful index → covering composite index",
        slow_sql="""
            SELECT b.referrer_id,
                   COUNT(*) AS appointments,
                   ROUND(SUM(b.charge * b.units), 2) AS charges
            FROM sessions AS d
            INNER JOIN appointments AS b ON b.session_id = d.session_id
            WHERE d.session_date BETWEEN '2025-06-01' AND '2025-06-30'
            GROUP BY b.referrer_id""",
        fast_sql="""
            SELECT b.referrer_id,
                   COUNT(*) AS appointments,
                   ROUND(SUM(b.charge * b.units), 2) AS charges
            FROM sessions AS d
            INNER JOIN appointments AS b ON b.session_id = d.session_id
            WHERE d.session_date BETWEEN '2025-06-01' AND '2025-06-30'
            GROUP BY b.referrer_id""",
        fast_indexes=["CREATE INDEX ix_sessions_date ON sessions (session_date)",
                      "CREATE INDEX ix_appts_sess_ref_cover ON appointments (session_id, referrer_id, charge, units)"],
        lesson="Same SQL, different physical design. Without an index on the join key the engine scans all 1M appointments "
               "and throws most away (plan: SCAN b + BLOOM FILTER). An index on the date filter plus a composite index on "
               "the join key that also carries the columns the query reads turns it into a few small index range scans.",
    ),
]
