"""Back-test: cut seasoned vintages at MOB n, project to MOB 36, compare with what actually happened.

Leave-one-vintage-out: when vintage v is tested, its own data is removed from the reference curves, the
chain-ladder factors and the credibility k. This is cross-validation, not a strict as-of-date replay: the
reference still uses other vintages' later months, which at the cut date would not have existed yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import project, reference
from .synth import TERM

CUTS = [3, 6, 9, 12, 18, 24, 30]
METHODS = {"proj_chain_ladder": "chain ladder", "proj_credibility": "credibility blend",
           "proj_reference": "reference only", "proj_naive": "naive (no further defaults)"}


def run(curves: pd.DataFrame, cuts=CUTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = curves[reference.seasoned(curves)]
    rows = []
    for seg, vint in s[["segment", "vintage"]].drop_duplicates().itertuples(index=False):
        ex = {(seg, vint)}
        ref = reference.reference(curves, exclude=ex)
        model = project.fit(curves, ref, exclude=ex)[seg]
        v = s[(s.segment == seg) & (s.vintage == vint)].set_index("mob")
        actual = v.cum_default_pct.loc[TERM]
        for n in cuts:
            p = model.project(v.cum_default_pct.loc[n], n, v.orig_balance.iloc[0])
            for col, name in METHODS.items():
                rows.append({"segment": seg, "vintage": vint, "cut_mob": n, "method": name, "projected": p[col],
                             "actual": actual, "ape": abs(p[col] - actual) / actual,
                             "in_band": p["proj_low"] <= actual <= p["proj_high"]})
    detail = pd.DataFrame(rows)
    summ = (detail.groupby(["cut_mob", "method"])
            .agg(mape=("ape", "mean"), n=("ape", "size")).reset_index())
    cov = detail[detail.method == "credibility blend"].groupby("cut_mob").in_band.mean()
    summ["band_coverage"] = np.where(summ.method == "credibility blend", summ.cut_mob.map(cov), np.nan)
    best = summ.loc[summ.groupby("cut_mob").mape.idxmin()].set_index("cut_mob").method
    summ["best_at_cut"] = summ.cut_mob.map(best)
    return detail.drop(columns="in_band").assign(in_band=detail.in_band), summ
