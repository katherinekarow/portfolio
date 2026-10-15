"""Combine price-grid upload workbooks into one validated file.

Analysts maintain price grids in several workbooks, one tab per region, and the column
headers drift between tabs ("Min Bid USD", "min_bid", "MIN"). This module:

* writes the synthetic books to real .xlsx files (so the demo exercises actual Excel I/O)
* reads every tab of every workbook
* harmonizes headers through an alias table (unknown headers fail loudly)
* validates each row and returns reason-coded rejects instead of silently dropping them
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .normalize import tier_from_label

HEADER_ALIASES = {
    "lot_id": {"lot", "lotid", "lotno"},
    "from_tier": {"fromtier", "from", "fromgrade", "fromgradename"},
    "to_tier": {"totier", "to", "tograde", "togradename"},
    "min_bid_usd": {"minbidusd", "minbid", "min"},
    "max_bid_usd": {"maxbidusd", "maxbid", "max"},
}
_ALIAS_LOOKUP = {alias: canon for canon, aliases in HEADER_ALIASES.items() for alias in aliases}


class UnknownHeaderError(ValueError):
    pass


def _canon(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def harmonize_headers(df: pd.DataFrame, where: str = "") -> pd.DataFrame:
    mapping, unknown = {}, []
    for col in df.columns:
        canon = _ALIAS_LOOKUP.get(_canon(col))
        (mapping.__setitem__(col, canon) if canon else unknown.append(col))
    if unknown:
        raise UnknownHeaderError(f"{where}: unrecognised column(s) {unknown}")
    missing = set(HEADER_ALIASES) - set(mapping.values())
    if missing:
        raise UnknownHeaderError(f"{where}: missing required column(s) {sorted(missing)}")
    return df.rename(columns=mapping)[list(HEADER_ALIASES)]


def write_books(books: dict[str, dict[str, pd.DataFrame]], folder: Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, tabs in books.items():
        path = folder / name
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for tab, df in tabs.items():
                df.to_excel(xw, sheet_name=tab, index=False)
        paths.append(path)
    return paths


def combine_uploads(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every tab of every workbook -> (combined valid grid, rejects with reason codes)."""
    frames = []
    for path in paths:
        for tab, df in pd.read_excel(path, sheet_name=None).items():
            df = harmonize_headers(df, where=f"{Path(path).name}[{tab}]")
            df.insert(0, "source", f"{Path(path).name}:{tab}")
            df.insert(1, "source_row", df.index + 2)          # Excel row number (header is row 1)
            frames.append(df)
    grid = pd.concat(frames, ignore_index=True)
    grid["lot_id"] = grid.lot_id.astype("string").str.strip().str.upper()
    grid["from_tier"] = grid.from_tier.map(tier_from_label)
    grid["to_tier"] = grid.to_tier.map(tier_from_label)

    reason = pd.Series("", index=grid.index)
    reason[grid.min_bid_usd > grid.max_bid_usd] = "MIN_ABOVE_MAX"
    reason[(grid.min_bid_usd <= 0) | grid.min_bid_usd.isna() | grid.max_bid_usd.isna()] = "INVALID_AMOUNT"
    reason[grid.from_tier.isna() | grid.to_tier.isna()] = "UNMAPPED_GRADE"
    reason[grid.lot_id.isna() | (grid.lot_id == "")] = "MISSING_LOT"
    dup = grid.duplicated(["lot_id", "from_tier", "to_tier"], keep=False) & (reason == "")
    reason[dup] = "DUPLICATE_KEY"

    bad = reason != ""
    rejects = grid[bad].assign(reason_code=reason[bad]).reset_index(drop=True)
    valid = grid[~bad].reset_index(drop=True)
    return valid, rejects
