"""List files in a Drive folder using whatever credentials the environment provides.

In GitHub Actions that's the short-lived token from google-github-actions/auth (no key);
on a laptop it's `gcloud auth application-default login`. Read-only scope on purpose.
"""
import os

import google.auth
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def main() -> None:
    creds, _ = google.auth.default(scopes=SCOPES)
    drive = build("drive", "v3", credentials=creds)
    folder = os.environ["DRIVE_FOLDER_ID"]
    resp = drive.files().list(q=f"'{folder}' in parents and trashed = false",
                              fields="files(id, name, mimeType)", pageSize=50).execute()
    for f in resp.get("files", []):
        print(f"{f['mimeType']:45} {f['name']}")


if __name__ == "__main__":
    main()
