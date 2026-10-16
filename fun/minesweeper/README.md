# Sakura Minesweeper
> Classic Minesweeper in plain JavaScript: a safe first click, flood fill, chording, long-press on touch screens, cherry-blossom flags and best times saved on your device.

**Tools:** JavaScript, HTML/CSS

**Live demo:** [Play Minesweeper](https://katherinekarow.github.io/portfolio/minesweeper/)

## The game
| Level | Board | Mines |
|---|---|---|
| Beginner | 9 × 9 | 10 |
| Intermediate | 16 × 16 | 40 |
| Expert | 30 × 16 | 99 |

- **Fair start:** mines are placed after the first click, never on it or its neighbours, so the first click always opens an area.
- **Flood fill:** opening a zero reveals the whole connected empty region (iterative, so expert boards never overflow the stack).
- **Chording:** clicking a number that already has the right number of blossoms around it opens the rest of its neighbours.
- **Marking:** right-click, long-press on touch screens, or *Blossom mode* (key **F**) to place a cherry blossom on a suspected mine.

## Best times
Your five fastest times per level are kept in your browser. A public leaderboard where each score is committed to this repo is designed and parked as a [future project](../future-ideas/online-high-scores/).
