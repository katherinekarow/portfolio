# SQL layer (Snowflake dialect)

The same logic as the Python package, written as warehouse views and controls on a **made-up schema**:

| Table | Grain |
|---|---|
| `sessions` | one row per session: `session_id`, `segment`, `capacity`, `is_open`, `days_out_today` |
| `appointments_daily` | one row per session and days-to-session: `session_id`, `days_out`, `gross_appointments`, `net_appointments` |

| File | What it shows |
|---|---|
| `01_segment_curves.sql` | percentile benchmarks per segment and days_out, with a running max so targets never decrease |
| `02_session_plan.sql` | today's position, credibility-weighted projection and gap decomposition |
| `03_quality_controls.sql` | reason-coded reject table + control-total reconciliation |
| `04_snapshot_to_history.sql` | snapshot-first history with same-day duplicate protection |
