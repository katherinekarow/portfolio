-- Legacy helper: run stamp used in report headers (Oracle)
SELECT TO_CHAR(SYSDATE, 'YYYY-MM-DD') AS run_date,
       TRUNC(SYSDATE, 'MM') AS month_start
FROM dual
