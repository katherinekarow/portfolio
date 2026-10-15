-- Legacy report: five highest-billing sessions (Oracle top-N pattern)
SELECT session_id, revenue
FROM (
    SELECT b.session_id AS session_id,
           SUM(NVL(b.charge, 0) * b.units) AS revenue
    FROM appointments b
    WHERE NVL(b.status, 'A') <> 'C'
    GROUP BY b.session_id
    ORDER BY revenue DESC, session_id
)
WHERE ROWNUM <= 5
