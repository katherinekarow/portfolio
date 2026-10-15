#!/usr/bin/env python3
"""
Sudoku generator, solver and terminal game.

    python sudoku.py                    # a new medium puzzle
    python sudoku.py --level hard       # easy | medium | hard | expert
    python sudoku.py --seed 42          # the same puzzle every time
    python sudoku.py --solution         # show the answer too
    python sudoku.py --play             # play right in the terminal

Every generated puzzle has exactly one solution.
No third-party packages needed (Python 3.8+).
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import List, Optional

Grid = List[int]  # 81 cells, row by row, 0 = empty

# How many given numbers each level keeps (fewer clues = harder).
LEVELS = {"easy": 40, "medium": 32, "hard": 27, "expert": 24}

ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOXES = [
    [(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
    for br in range(3)
    for bc in range(3)
]
# For each cell, the 20 other cells it must not repeat a digit with.
PEERS = [
    sorted({p for unit in (ROWS[i // 9], COLS[i % 9], BOXES[(i // 27) * 3 + (i % 9) // 3]) for p in unit} - {i})
    for i in range(81)
]


def candidates(grid: Grid, i: int) -> List[int]:
    used = {grid[p] for p in PEERS[i]}
    return [d for d in range(1, 10) if d not in used]


def count_solutions(grid: Grid, limit: int = 2) -> int:
    """Backtracking search that stops as soon as `limit` solutions are found."""
    best, best_opts = -1, None
    for i in range(81):
        if grid[i] == 0:
            opts = candidates(grid, i)
            if not opts:
                return 0
            if best_opts is None or len(opts) < len(best_opts):
                best, best_opts = i, opts
                if len(opts) == 1:
                    break
    if best == -1:
        return 1
    total = 0
    for d in best_opts:
        grid[best] = d
        total += count_solutions(grid, limit - total)
        grid[best] = 0
        if total >= limit:
            break
    return total


def solve(grid: Grid) -> Optional[Grid]:
    """Return a solved copy of `grid`, or None if it has no solution."""
    g = grid[:]

    def fill() -> bool:
        best, best_opts = -1, None
        for i in range(81):
            if g[i] == 0:
                opts = candidates(g, i)
                if not opts:
                    return False
                if best_opts is None or len(opts) < len(best_opts):
                    best, best_opts = i, opts
        if best == -1:
            return True
        for d in best_opts:
            g[best] = d
            if fill():
                return True
        g[best] = 0
        return False

    return g if fill() else None


def full_grid(rng: random.Random) -> Grid:
    g = [0] * 81

    def fill(i: int = 0) -> bool:
        if i == 81:
            return True
        opts = candidates(g, i)
        rng.shuffle(opts)
        for d in opts:
            g[i] = d
            if fill(i + 1):
                return True
        g[i] = 0
        return False

    fill()
    return g


def generate(level: str = "medium", seed: Optional[int] = None):
    """Return (puzzle, solution). The puzzle always has a unique solution."""
    rng = random.Random(seed)
    solution = full_grid(rng)
    puzzle = solution[:]
    target = LEVELS[level]
    cells = list(range(81))
    rng.shuffle(cells)
    for i in cells:
        if sum(1 for v in puzzle if v) <= target:
            break
        keep = puzzle[i]
        puzzle[i] = 0
        if count_solutions(puzzle[:]) != 1:
            puzzle[i] = keep  # removing it would allow two answers
    return puzzle, solution


def to_string(grid: Grid) -> str:
    return "".join(str(v) for v in grid)


def pretty(grid: Grid, original: Optional[Grid] = None) -> str:
    lines = ["    1 2 3   4 5 6   7 8 9", "  +-------+-------+-------+"]
    for r in range(9):
        row = []
        for c in range(9):
            v = grid[r * 9 + c]
            row.append(str(v) if v else ".")
            if c in (2, 5):
                row.append("|")
        lines.append(f"{'ABCDEFGHI'[r]} | " + " ".join(row) + " |")
        if r in (2, 5):
            lines.append("  +-------+-------+-------+")
    lines.append("  +-------+-------+-------+")
    return "\n".join(lines)


def play(puzzle: Grid, solution: Grid) -> None:
    board = puzzle[:]
    print("\nEnter moves like  A1 5  (row letter, column number, digit).")
    print("Use 0 to clear a cell, 'hint' for help, 'quit' to stop.\n")
    while True:
        print(pretty(board))
        if board == solution:
            print("\n  *** Solved! Nicely done. ***\n")
            return
        try:
            move = input("> ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if move in ("Q", "QUIT", "EXIT"):
            print("Answer:\n" + pretty(solution))
            return
        if move == "HINT":
            empty = [i for i in range(81) if board[i] != solution[i]]
            i = random.choice(empty)
            board[i] = solution[i]
            print(f"Hint: {'ABCDEFGHI'[i // 9]}{i % 9 + 1} is {solution[i]}")
            continue
        parts = move.replace(",", " ").split()
        if len(parts) != 2 or len(parts[0]) != 2 or parts[0][0] not in "ABCDEFGHI" \
                or parts[0][1] not in "123456789" or parts[1] not in "0123456789":
            print("Try something like  B7 3")
            continue
        i = "ABCDEFGHI".index(parts[0][0]) * 9 + int(parts[0][1]) - 1
        if puzzle[i]:
            print("That's one of the starting numbers.")
            continue
        d = int(parts[1])
        if d and d != solution[i]:
            print("Hmm, that doesn't fit there.")
            continue
        board[i] = d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate and play Sudoku puzzles.")
    p.add_argument("--level", choices=LEVELS, default="medium")
    p.add_argument("--seed", type=int, help="same seed = same puzzle")
    p.add_argument("--solution", action="store_true", help="also print the answer")
    p.add_argument("--play", action="store_true", help="play in the terminal")
    p.add_argument("--raw", action="store_true", help="print as an 81-character string")
    args = p.parse_args(argv)

    puzzle, solution = generate(args.level, args.seed)
    if args.raw:
        print(to_string(puzzle))
        if args.solution:
            print(to_string(solution))
        return 0
    print(f"Sudoku ({args.level}, {sum(1 for v in puzzle if v)} clues)\n")
    if args.play:
        play(puzzle, solution)
        return 0
    print(pretty(puzzle))
    if args.solution:
        print("\nSolution\n" + pretty(solution))
    return 0


if __name__ == "__main__":
    sys.exit(main())
