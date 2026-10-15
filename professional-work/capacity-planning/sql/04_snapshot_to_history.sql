-- Snapshot-first: copy the current table into HIST_<table> before a pipeline overwrites it.
-- Same-day reruns with identical data keep only the latest snapshot; changed data adds a new one.
CREATE OR REPLACE PROCEDURE planning.snapshot_to_history(source_table STRING)
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    hist_table STRING DEFAULT 'HIST_' || source_table;
    run_ts     TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP();
    new_hash   NUMBER;
    same_day   NUMBER DEFAULT 0;
BEGIN
    -- create the history table on first use (same columns + audit fields)
    EXECUTE IMMEDIATE 'CREATE TABLE IF NOT EXISTS planning.' || hist_table ||
        ' AS SELECT *, CURRENT_TIMESTAMP() AS history_added_at, 0::NUMBER AS content_hash FROM planning.'
        || source_table || ' WHERE 1 = 0';

    -- fingerprint of the current contents
    EXECUTE IMMEDIATE 'SELECT HASH_AGG(*) FROM planning.' || source_table;
    SELECT $1 INTO :new_hash FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

    -- identical snapshot already taken today? replace it instead of duplicating
    EXECUTE IMMEDIATE 'SELECT COUNT(*) FROM planning.' || hist_table ||
        ' WHERE history_added_at::DATE = CURRENT_DATE() AND content_hash = ' || new_hash;
    SELECT $1 INTO :same_day FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
    IF (same_day > 0) THEN
        EXECUTE IMMEDIATE 'DELETE FROM planning.' || hist_table ||
            ' WHERE history_added_at::DATE = CURRENT_DATE() AND content_hash = ' || new_hash;
    END IF;

    EXECUTE IMMEDIATE 'INSERT INTO planning.' || hist_table ||
        ' SELECT *, ''' || run_ts || '''::TIMESTAMP_TZ, ' || new_hash || ' FROM planning.' || source_table;
    RETURN IFF(same_day > 0, 'replaced identical same-day snapshot', 'added new snapshot');
END;
$$;
