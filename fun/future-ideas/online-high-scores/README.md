# Planned: online high scores for the games
> A design (with draft code) for a public Sudoku and Minesweeper leaderboard where every new score becomes a commit to this repository, with no GitHub account needed to submit.

**Tools:** JavaScript, GitHub Actions, Python, serverless function

**Status: parked for a future project.** The games currently keep each player's best times on their own device.

## How it would work
```
game page ──POST──▶ score relay (serverless) ──repository_dispatch──▶ GitHub Action ──▶ commit to docs/scores/scores.json
```
1. A static GitHub Pages site can't hold a GitHub token, so a tiny relay ([`relay/api/score.js`](relay/api/score.js)) sits in the middle. It checks the request comes from the site, rate-limits by IP and validates the fields.
2. The relay triggers a workflow ([`scores.workflow.yml`](scores.workflow.yml), which would live in `.github/workflows/`). The workflow runs [`scores.py`](scores.py): name rules, a polite-name filter, plausible-time bounds per level, keep the top 25. Then it commits.
3. The scoreboard page reads `scores.json`, so the leaderboard's history is the commit log.

## To switch it on later
- Create a fine-grained GitHub token limited to this repo, with *Contents: read and write*.
- Deploy `relay/` to a serverless host and store the token there as `GITHUB_TOKEN` (never in the repo).
- Move the workflow into `.github/workflows/`, restore `docs/scores/`, and point the games at the relay URL.
