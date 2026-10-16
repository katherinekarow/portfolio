# Score relay

A ~40-line serverless function that lets visitors submit game scores **without a GitHub account**:

```
game page ──POST {game, level, name, seconds}──▶ relay (Vercel) ──repository_dispatch──▶ GitHub Action ──▶ commit to docs/scores/scores.json
```

The browser never sees a token. The relay checks the origin, rate-limits by IP and validates the fields; the Action
(`tools/scores.py`) validates again (name rules, a polite-name filter, plausible times) before committing, so every
high score is a commit in the repository history.

## Set up (one time)
1. GitHub → Settings → Developer settings → Fine-grained tokens → *Generate new token*: repository access
   **only `portfolio`**, permission **Contents: Read and write**. Copy it (only you should handle it).
2. Deploy this folder to Vercel (root directory `tools/score-relay`), and add the environment variable
   `GITHUB_TOKEN` with that token in the Vercel project settings.
3. Put the deployed URL (e.g. `https://<project>.vercel.app/api/score`) in `docs/assets/scores-config.js`.
