#!/usr/bin/env python3
"""
Make today's Sudoku.

Writes:
  docs/sudoku/daily.json   -> loaded by the playable web page ("Today's puzzle")
  fun/sudoku/daily.svg     -> the picture shown on the front-page README

Run by .github/workflows/daily-sudoku.yml every morning, or by hand:
    python tools/daily_sudoku.py [YYYY-MM-DD]
"""
import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fun" / "sudoku"))
import sudoku  # noqa: E402

# Gentle on Monday, tougher toward the weekend.
LEVEL_BY_WEEKDAY = ["easy", "easy", "medium", "medium", "hard", "hard", "expert"]


def svg(puzzle, date_label, level):
    cell, pad, top = 40, 20, 56
    size = cell * 9
    w, h = size + pad * 2, size + top + pad + 28
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" rx="16" fill="#fff8f2"/>',
        f'<text x="{w/2}" y="34" text-anchor="middle" font-size="20" font-weight="700" fill="#3d2c4d">'
        f"Today's Sudoku · {date_label}</text>",
        f'<rect x="{pad}" y="{top}" width="{size}" height="{size}" fill="#ffffff"/>',
    ]
    for r in range(9):
        for c in range(9):
            if (r // 3 + c // 3) % 2 == 0:
                out.append(f'<rect x="{pad + c*cell}" y="{top + r*cell}" width="{cell}" height="{cell}" fill="#f6eefc"/>')
    for k in range(10):
        sw = 3 if k % 3 == 0 else 1
        col = "#3d2c4d" if k % 3 == 0 else "#c9b8d6"
        out.append(f'<line x1="{pad + k*cell}" y1="{top}" x2="{pad + k*cell}" y2="{top + size}" stroke="{col}" stroke-width="{sw}"/>')
        out.append(f'<line x1="{pad}" y1="{top + k*cell}" x2="{pad + size}" y2="{top + k*cell}" stroke="{col}" stroke-width="{sw}"/>')
    for i, v in enumerate(puzzle):
        if v:
            x = pad + (i % 9) * cell + cell / 2
            y = top + (i // 9) * cell + cell / 2 + 8
            out.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="22" font-weight="600" fill="#3d2c4d">{v}</text>')
    out.append(
        f'<text x="{w/2}" y="{h-14}" text-anchor="middle" font-size="14" fill="#7a6a88">'
        f"Level: {level} · click to play in your browser</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def main():
    day = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime.now(dt.timezone.utc).date()
    level = LEVEL_BY_WEEKDAY[day.weekday()]
    puzzle, solution = sudoku.generate(level, seed=int(day.strftime("%Y%m%d")))
    data = {
        "date": day.isoformat(),
        "level": level,
        "puzzle": sudoku.to_string(puzzle),
        "solution": sudoku.to_string(solution),
    }
    (ROOT / "docs" / "sudoku" / "daily.json").write_text(json.dumps(data, indent=2) + "\n")
    (ROOT / "fun" / "sudoku" / "daily.svg").write_text(svg(puzzle, day.strftime("%b %d, %Y"), level) + "\n")
    print(f"{day} {level}: {data['puzzle']}")


if __name__ == "__main__":
    main()
