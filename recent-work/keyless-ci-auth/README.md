# Keyless Google Cloud access from GitHub Actions
> Replaced a long-lived service-account key stored in CI with short-lived, keyless credentials (Workload Identity Federation).

**Tools:** GitHub Actions (OIDC), Google Cloud IAM, Workload Identity Federation, Python (google-auth)

## The problem
A scheduled automation needed read access to a Google Drive folder. The usual approach is to download a
service-account JSON key and store it as a CI secret — a password that never expires and can leak
(for example if a workflow ever prints it into logs).

## What I did
1. Created a **workload identity pool + OIDC provider** in Google Cloud that trusts GitHub's token issuer,
   restricted with an attribute condition to **one specific repository**.
2. Created a **dedicated, least-privilege service account** with read-only access to just one folder,
   and allowed only that repository to impersonate it.
3. Switched the workflow to `google-github-actions/auth` with `id-token: write` — each run gets a token that
   expires in about an hour. **No key exists anywhere.**
4. Updated the Python code to use `google.auth.default()` (keyless in CI, still works locally) and narrowed
   the API scope to read-only.
5. Deleted the old keys and disabled the old account.

## Files
- [`setup.sh`](setup.sh) — the one-time Google Cloud setup (placeholders, safe to read)
- [`example-workflow.yml`](example-workflow.yml) — the GitHub Actions side
- [`drive_client.py`](drive_client.py) — keyless-friendly client code
