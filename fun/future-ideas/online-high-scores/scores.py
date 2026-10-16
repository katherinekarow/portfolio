#!/usr/bin/env python3
"""Validate a submitted game score and add it to docs/scores/scores.json.

Run by .github/workflows/scores.yml when the score relay sends a `score` event:
    SCORE_PAYLOAD='{"game":"minesweeper","level":"beginner","name":"blossom","seconds":42.1}' python tools/scores.py add
Rejected scores exit with status 0 and a message (nothing is committed), so spam never fails the workflow loudly.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "docs" / "scores" / "scores.json"
LEVELS = {"sudoku": ["easy", "medium", "hard", "expert"], "minesweeper": ["beginner", "intermediate", "expert"]}
MIN_SECONDS = {"sudoku": {"easy": 40, "medium": 60, "hard": 90, "expert": 120},
               "minesweeper": {"beginner": 1, "intermediate": 8, "expert": 25}}
MAX_SECONDS = 4 * 3600
KEEP = 25
NAME = re.compile(r"^[A-Za-z0-9 _.\-]{1,16}$")
BLOCKED = {"admin", "fuck", "shit", "bitch", "cunt", "nigger", "faggot", "slut", "whore", "dick", "pussy", "rape", "nazi"}


def clean_name(name: str) -> str | None:
    name = re.sub(r"\s+", " ", str(name)).strip()
    if not NAME.match(name):
        return None
    squashed = re.sub(r"[^a-z]", "", name.lower().replace("0", "o").replace("1", "i").replace("3", "e").replace("4", "a").replace("5", "s"))
    if any(b in squashed for b in BLOCKED):
        return None
    return name


def validate(p: dict) -> tuple[dict | None, str]:
    game, level = p.get("game"), p.get("level")
    if game not in LEVELS or level not in LEVELS[game]:
        return None, "unknown game or level"
    name = clean_name(p.get("name", ""))
    if not name:
        return None, "name must be 1-16 letters, numbers, spaces or _ . - and polite"
    try:
        secs = round(float(p.get("seconds")), 1)
    except (TypeError, ValueError):
        return None, "seconds must be a number"
    if not (MIN_SECONDS[game][level] <= secs <= MAX_SECONDS):
        return None, "time is outside the plausible range"
    return {"game": game, "level": level, "name": name, "seconds": secs}, "ok"


def add(entry: dict, data: dict, today: str) -> tuple[dict, int | None]:
    board = data.setdefault(entry["game"], {}).setdefault(entry["level"], [])
    row = {"name": entry["name"], "seconds": entry["seconds"], "date": today}
    board.append(row)
    board.sort(key=lambda r: (r["seconds"], r["date"]))
    del board[KEEP:]
    data["updated"] = today
    return data, (board.index(row) + 1 if row in board else None)


def main(argv):
    if argv[1:2] != ["add"]:
        sys.exit(__doc__)
    payload = json.loads(os.environ.get("SCORE_PAYLOAD", "{}") or "{}")
    entry, why = validate(payload)
    if not entry:
        print(f"rejected: {why}")
        return
    data = json.loads(FILE.read_text()) if FILE.exists() else {}
    data, rank = add(entry, data, dt.date.today().isoformat())
    if rank is None:
        print("not fast enough for the top", KEEP)
        return
    FILE.write_text(json.dumps(data, indent=1) + "\n")
    msg = f"Score: {entry['name']} {entry['game']}/{entry['level']} {entry['seconds']}s (#{rank})"
    print(msg)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"message={msg}\n")


if __name__ == "__main__":
    main(sys.argv)
