# Sudoku generator & game
> Generates Sudoku and Killer Sudoku puzzles with exactly one solution: play in the browser, in the terminal, or solve the daily puzzle.

**Tools:** Python, JavaScript, GitHub Pages, GitHub Actions

**Live demo:** [Play in the browser](https://katherinekarow.github.io/portfolio/sudoku/)

### [Play in your browser](https://katherinekarow.github.io/portfolio/sudoku/) · [Today's puzzle](https://katherinekarow.github.io/portfolio/sudoku/?daily=1)

Four classic levels plus **Killer Sudoku**, with pencil notes, hints, undo, a timer, keyboard controls and falling petals when you win.

**Killer Sudoku:** the board is split into dashed *cages*. The digits in a cage add up to its small number and can't repeat, on top of the usual row, column and box rules, and there are almost no starting digits. The generator builds the cages from a solved grid. A solver then prunes on cage sums (the remaining cells must still be able to reach the remaining total with unused digits). It adds a starting digit wherever two answers are still possible, until exactly one solution remains.
Every puzzle link can be shared: the same link always opens the same puzzle.

### In the terminal
```bash
python sudoku.py                  # new medium puzzle
python sudoku.py --level expert   # easy | medium | hard | expert
python sudoku.py --play           # play: type moves like "B7 3", or "hint"
python sudoku.py --seed 42 --solution
```
No installs needed — just Python 3.

## How it works
1. **Fill a full grid** with randomized backtracking (try digits in random order, back up on dead ends).
2. **Dig holes**: remove numbers one at a time in random order. After each removal a solver
   counts solutions and stops at 2 — if there's more than one, the number goes back.
3. Stop when the clue count matches the level (40 easy → 24 expert).

The solver always branches on the cell with the fewest candidates, which keeps even expert puzzles fast
(~0.1 s in Python). The browser game uses the same algorithm in JavaScript with a seeded random generator,
so a seed in the URL reproduces the exact puzzle.

**Daily puzzle:** a GitHub Action (`.github/workflows/daily-sudoku.yml`) runs `tools/daily_sudoku.py` every morning,
which saves the puzzle for the website and redraws [`daily.svg`](daily.svg) on the front page.
