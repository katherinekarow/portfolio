-- Legacy report: referrer hierarchy with daily snapshot stamp (Oracle)
SELECT LEVEL AS depth,
       a.referrer_id AS referrer_id,
       a.referrer_name AS referrer_name,
       TRUNC(SYSDATE) AS snapshot_date
FROM referrers a
START WITH a.parent_referrer_id IS NULL
CONNECT BY PRIOR a.referrer_id = a.parent_referrer_id
