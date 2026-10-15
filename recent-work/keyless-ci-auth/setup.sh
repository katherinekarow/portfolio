#!/usr/bin/env bash
# One-time setup: let ONE GitHub repo act as ONE read-only service account. No keys created.
set -euo pipefail
PROJECT_ID="your-project-id"
PROJECT_NUMBER="123456789012"
REPO="your-github-user/your-repo"
SA="drive-reader@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project="$PROJECT_ID"

gcloud iam service-accounts create drive-reader \
  --display-name="drive-reader" --description="Read-only access to one folder" --project="$PROJECT_ID"

gcloud iam workload-identity-pools create github --location=global \
  --display-name="GitHub Actions" --project="$PROJECT_ID"

gcloud iam workload-identity-pools providers create-oidc github-actions --location=global \
  --workload-identity-pool=github --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'" --project="$PROJECT_ID"

gcloud iam service-accounts add-iam-policy-binding "$SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${REPO}" \
  --project="$PROJECT_ID"

# Finally: share the Drive folder with $SA as Viewer, and store these as GitHub *variables* (not secrets):
#   GCP_WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/providers/github-actions
#   GCP_SERVICE_ACCOUNT = $SA
