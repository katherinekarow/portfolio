"""Parity harness: prove a migrated report returns what the legacy one did - or explain why not.

compare(legacy, new, keys) lines the two outputs up on their business key and reports
  * rows only in legacy / only in new, and duplicate keys on either side
  * per-column mismatches on matched rows, with numeric tolerances
  * column totals
  * a DIAGNOSIS per column, so "34 differences" turns into "rounding to whole dollars in the
    legacy extract" or "legacy left NULL where the new view returns 0"

Verdicts
  PASS                 identical within tolerance
  PASS_WITH_EXPLAINED  every difference falls in an explained, accepted category
  FAIL                 at least one unexplained difference (a real logic change to investigate)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACCEPTED = {"rounding", "null_vs_zero", "formatting"}


@dataclass
class ColumnDiagnosis:
    column: str
    mismatches: int
    category: str
    detail: str


@dataclass
class ParityResult:
    legacy_rows: int
    new_rows: int
    only_in_legacy: pd.DataFrame
    only_in_new: pd.DataFrame
    duplicate_keys: dict
    diffs: pd.DataFrame
    diagnoses: list[ColumnDiagnosis] = field(default_factory=list)
    totals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def verdict(self) -> str:
        if len(self.only_in_legacy) or len(self.only_in_new) or any(self.duplicate_keys.values()):
            return "FAIL"
        if self.diffs.empty:
            return "PASS"
        return "PASS_WITH_EXPLAINED" if all(d.category in ACCEPTED for d in self.diagnoses) else "FAIL"

    def summary(self) -> str:
        lines = [f"verdict: {self.verdict}",
                 f"rows: legacy {self.legacy_rows:,} · new {self.new_rows:,} · only in legacy {len(self.only_in_legacy)} · only in new {len(self.only_in_new)}"]
        for side, n in self.duplicate_keys.items():
            if n:
                lines.append(f"duplicate keys in {side}: {n}")
        for d in self.diagnoses:
            lines.append(f"  {d.column:<16} {d.mismatches:>5} differences  -> {d.category}: {d.detail}")
        return "\n".join(lines)


def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _row_category(lv, nv) -> str:
    """Classify one differing value pair."""
    if isinstance(lv, (pd.Timestamp, np.datetime64)) or isinstance(nv, (pd.Timestamp, np.datetime64)):
        return "other"                                    # dates are diagnosed as a group (shift detection)
    ln, nn = pd.to_numeric(pd.Series([lv]), errors="coerce")[0], pd.to_numeric(pd.Series([nv]), errors="coerce")[0]
    if (pd.isna(lv) and nn == 0) or (pd.isna(nv) and ln == 0):
        return "null_vs_zero"
    if pd.notna(ln) and pd.notna(nn):
        d = abs(nn - ln)
        if d <= 0.5 + 1e-9 and (round(ln) == ln or round(nn) == nn):
            return "rounding"
        if d <= 0.005 + 1e-9:
            return "rounding"
        return "numeric"
    if pd.notna(lv) and pd.notna(nv) and str(lv).strip().casefold() == str(nv).strip().casefold():
        return "formatting"
    return "other"


EXPLAIN = {
    "null_vs_zero": "one side has NULL where the other returns 0 (NVL/COALESCE default)",
    "rounding": "rounding only (whole units or cents)",
    "formatting": "case / whitespace only",
}


def _diagnose(col: str, d: pd.DataFrame) -> list[ColumnDiagnosis]:
    cats = pd.Series([_row_category(a, b) for a, b in zip(d["legacy"], d["new"])], index=d.index)
    out = []
    for cat, idx in cats.groupby(cats).groups.items():
        sub = d.loc[idx]
        if cat in EXPLAIN:
            detail = EXPLAIN[cat]
            if cat == "null_vs_zero":
                detail += f" - NULL on the {'legacy' if sub['legacy'].isna().all() else 'new'} side"
            out.append(ColumnDiagnosis(col, len(sub), cat, detail))
            continue
        lnum, nnum = pd.to_numeric(sub["legacy"], errors="coerce"), pd.to_numeric(sub["new"], errors="coerce")
        ratio = (nnum / lnum.replace(0, np.nan)).dropna()
        ld, nd = pd.to_datetime(sub["legacy"], errors="coerce"), pd.to_datetime(sub["new"], errors="coerce")
        if cat == "numeric" and len(ratio) == len(sub) and len(sub) > 1 and ratio.std() < 1e-6:
            out.append(ColumnDiagnosis(col, len(sub), "scale_factor", f"new = legacy x {ratio.iloc[0]:.4g} on every row (units / currency)"))
        elif cat == "other" and ld.notna().all() and nd.notna().all() and (nd - ld).nunique() == 1:
            out.append(ColumnDiagnosis(col, len(sub), "date_shift", f"every date moved by {(nd - ld).iloc[0]} (timezone / day boundary)"))
        else:
            net = (nnum - lnum).sum() if cat == "numeric" else float("nan")
            out.append(ColumnDiagnosis(col, len(sub), "logic", "values differ - investigate" + (f" (net {net:+,.2f})" if cat == "numeric" else "")))
    return out


def compare(legacy: pd.DataFrame, new: pd.DataFrame, keys: list[str],
            tolerance: dict[str, float] | None = None, default_tol: float = 1e-9) -> ParityResult:
    tolerance = tolerance or {}
    L, N = legacy.copy(), new.copy()
    for k in keys:
        L[k], N[k] = _norm_str(L[k]), _norm_str(N[k])
    dup = {"legacy": int(L.duplicated(keys).sum()), "new": int(N.duplicated(keys).sum())}
    m = L.merge(N, on=keys, how="outer", suffixes=("__l", "__n"), indicator=True)
    only_l = m[m._merge == "left_only"][keys].reset_index(drop=True)
    only_n = m[m._merge == "right_only"][keys].reset_index(drop=True)
    both = m[m._merge == "both"]
    rows, diags, totals = [], [], []
    for col in [c for c in L.columns if c not in keys and c in N.columns]:
        lv, nv = both[f"{col}__l"], both[f"{col}__n"]
        is_date = pd.api.types.is_datetime64_any_dtype(lv) or pd.api.types.is_datetime64_any_dtype(nv)
        lnum, nnum = pd.to_numeric(lv, errors="coerce"), pd.to_numeric(nv, errors="coerce")
        if is_date:
            same = (pd.to_datetime(lv) == pd.to_datetime(nv)) | (lv.isna() & nv.isna())
        elif lnum.notna().any() or nnum.notna().any():
            tol = tolerance.get(col, default_tol)
            same = ((lnum - nnum).abs() <= tol) | (lv.isna() & nv.isna())
            totals.append({"column": col, "legacy_total": float(lnum.sum()), "new_total": float(nnum.sum()),
                           "difference": float(nnum.sum() - lnum.sum())})
        else:
            same = (_norm_str(lv) == _norm_str(nv)).fillna(False) | (lv.isna() & nv.isna())
        bad = both[~same.to_numpy()]
        if len(bad):
            d = pd.DataFrame({**{k: bad[k].to_numpy() for k in keys}, "column": col,
                              "legacy": bad[f"{col}__l"].to_numpy(), "new": bad[f"{col}__n"].to_numpy()})
            rows.append(d)
            diags.extend(_diagnose(col, d))
    diffs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keys + ["column", "legacy", "new"])
    return ParityResult(len(L), len(N), only_l, only_n, dup, diffs, diags, pd.DataFrame(totals))
