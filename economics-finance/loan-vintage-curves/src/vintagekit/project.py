"""Project immature vintages to a lifetime (36-MOB) cumulative default rate.

Three projections, all starting from what the vintage has already shown, C(n) = cumulative default % at MOB n:

* chain ladder:   C(n) * f(n) * f(n+1) * ... * f(T-1),   f(m) = sum C$(m+1) / sum C$(m) over vintages seen at m+1
* reference only: C(n) + [R(T) - R(n)]                   (the rest of the life follows the segment P50 curve R)
* credibility:    C(n) + theta * [R(T) - R(n)],  theta = Z * pace + (1 - Z) * mu,  pace = C(n) / R(n)

Z = e / (e + k) is a Buhlmann-Straub credibility weight. Exposure e = original balance * R(n), the dollars of
default the reference says should have emerged by now; it grows with MOB and with vintage size. k = EPV / VHM is
estimated from the seasoned vintages (``buhlmann_straub``; VHM pooled across segments), so it comes from the data, not a dial.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import triangle
from .reference import seasoned
from .synth import TERM

Z_BAND = 1.2816  # two-sided 80% band


def dev_factors(tri: pd.DataFrame, term: int = TERM) -> np.ndarray:
    """Volume-weighted age-to-age factors; f[m] takes MOB m to m+1 (index 0 unused)."""
    f = np.ones(term + 1)
    a = tri.reindex(columns=range(1, term + 1)).to_numpy(dtype=float)
    for m in range(1, term):
        both = ~np.isnan(a[:, m - 1]) & ~np.isnan(a[:, m])
        den = a[both, m - 1].sum()
        f[m] = a[both, m].sum() / den if den > 0 else 1.0
    return f


def chain_ladder(c_n: float, n: int, f: np.ndarray, term: int = TERM) -> float:
    return float(c_n * np.prod(f[n:term]))


def _bs_parts(x: np.ndarray, w: np.ndarray) -> dict:
    ok = ~np.isnan(x) & (w > 0)
    w = np.where(ok, w, 0.0)
    x = np.where(ok, x, 0.0)
    wi = w.sum(1)
    keep = wi > 0
    w, x, wi = w[keep], x[keep], wi[keep]
    xbar_i = (w * x).sum(1) / wi
    wt = wi.sum()
    xbar = (wi * xbar_i).sum() / wt
    n_i = (w > 0).sum(1)
    sigma2 = (w * (x - xbar_i[:, None]) ** 2).sum() / (n_i - 1).sum()
    return {"mu": xbar, "sigma2": sigma2,
            "num": (wi * (xbar_i - xbar) ** 2).sum() - (len(wi) - 1) * sigma2,
            "den": wt - (wi ** 2).sum() / wt}


def buhlmann_straub(x: np.ndarray, w: np.ndarray) -> dict:
    """Non-parametric Buhlmann-Straub estimates for a groups x periods table (NaN or w=0 = missing).

    sigma2 (expected process variance) = sum_ij w_ij (x_ij - xbar_i)^2 / sum_i (n_i - 1)
    tau2   (variance of hypothetical means) = [sum_i w_i (xbar_i - xbar)^2 - (I - 1) sigma2] / (w - sum_i w_i^2 / w)
    k = sigma2 / tau2.  If tau2 <= 0 the groups are indistinguishable: k = inf and Z = 0 everywhere.
    """
    p = _bs_parts(x, w)
    tau2 = p["num"] / p["den"]
    k = p["sigma2"] / tau2 if tau2 > 0 else np.inf
    return {"k": k, "mu": p["mu"], "sigma2": p["sigma2"], "tau2": max(tau2, 0.0)}


def pooled_tau2(parts: list[dict]) -> float:
    """Between-vintage variance of the pace multiplier, pooled across segments.

    theta is relative (1 = on the reference), so its spread is comparable across segments, and 13 seasoned
    vintages per segment are too few to estimate it alone (one segment can come out negative by chance).
    Each segment's moment equation is in its own dollar units, so they are pooled as unweighted ratios."""
    t = [p["num"] / p["den"] for p in parts]
    return max(float(np.mean(t)), 1e-6)


def credibility_weight(exposure, k):
    """Z = e / (e + k): 0 with no exposure, tends to 1 as exposure grows."""
    e = np.asarray(exposure, dtype=float)
    return np.zeros_like(e) if np.isinf(k) else e / (e + k)


@dataclass
class SegmentModel:
    ref: pd.DataFrame          # mob 0..T with p25/p50/p75
    f: np.ndarray              # development factors
    k: float
    mu: float
    tau2: float
    sigma2: float = 0.0        # process variance per $ of expected default

    def r(self, m, col="p50"):
        return np.interp(m, self.ref.mob, self.ref[col])

    def project(self, c_n: float, n: int, orig: float, term: int = TERM) -> dict:
        r_n, r_t = self.r(n), self.r(term)
        remaining = max(r_t - r_n, 0.0)
        pace = c_n / r_n if r_n > 0 else self.mu
        z = float(credibility_weight(orig * r_n / 100, self.k))
        theta = z * pace + (1 - z) * self.mu
        # band: parameter uncertainty in theta, (1 - Z) tau2, plus process noise on the defaults still to come
        w_future = orig * remaining / 100
        var = (1 - z) * self.tau2 + (self.sigma2 / w_future if w_future > 0 else 0.0)
        half = Z_BAND * np.sqrt(var)
        cl = chain_ladder(c_n, n, self.f, term) if n < term else c_n
        return {"pace": pace, "z": z, "theta": theta,
                "proj_credibility": c_n + theta * remaining,
                "proj_low": c_n + max(theta - half, 0) * remaining,
                "proj_high": c_n + (theta + half) * remaining,
                "proj_chain_ladder": max(cl, c_n),
                "proj_reference": c_n + remaining,
                "proj_naive": c_n}

    def path(self, c_n: float, n: int, theta: float, term: int = TERM) -> tuple[np.ndarray, np.ndarray]:
        """Projected cumulative curve from MOB n to the term for a given pace multiplier theta."""
        m = np.arange(n, term + 1)
        return m, c_n + theta * (self.r(m) - self.r(n))


def pace_table(seg_curves: pd.DataFrame, ref: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Monthly pace observations x_ij = actual / reference-expected default $, weights w_ij = expected $."""
    tri_bal = triangle(seg_curves, "default_bal").reindex(columns=range(1, TERM + 1))
    orig = seg_curves.groupby("vintage").orig_balance.first().reindex(tri_bal.index).to_numpy()
    d_ref = np.diff(np.interp(np.arange(0, TERM + 1), ref.mob, ref.p50)) / 100
    w = orig[:, None] * d_ref[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = tri_bal.to_numpy(dtype=float) / w
    return x, w


def fit(curves: pd.DataFrame, ref: pd.DataFrame, exclude: set | None = None,
        tau2: float | None = None) -> dict[str, SegmentModel]:
    """One model per segment: reference, chain-ladder factors (all vintages) and credibility k (seasoned only).

    k = sigma2_segment / tau2_pooled. Pass ``tau2`` to reuse a pooled value fitted elsewhere."""
    is_seasoned = seasoned(curves)
    parts, base = {}, {}
    for seg, g in curves.groupby("segment"):
        mask = is_seasoned.loc[g.index]
        if exclude:
            drop = g.vintage.isin([v for s, v in exclude if s == seg])
            g, mask = g[~drop], mask[~drop]
        r = ref[ref.segment == seg].reset_index(drop=True)
        base[seg] = (r, dev_factors(triangle(g, "cum_default_bal")))
        parts[seg] = _bs_parts(*pace_table(g[mask], r))
    t2 = tau2 if tau2 is not None else pooled_tau2(list(parts.values()))
    return {seg: SegmentModel(r, f, parts[seg]["sigma2"] / t2, parts[seg]["mu"], t2, parts[seg]["sigma2"])
            for seg, (r, f) in base.items()}


def project_latest(curves: pd.DataFrame, models: dict[str, SegmentModel]) -> pd.DataFrame:
    """Projection for every vintage at its latest observed MOB."""
    last = curves.sort_values("mob").groupby(["segment", "vintage"]).tail(1)
    rows = []
    for r in last.itertuples():
        p = models[r.segment].project(r.cum_default_pct, int(r.mob), r.orig_balance)
        rows.append({"segment": r.segment, "vintage": r.vintage, "vintage_month": r.vintage_month, "mob": r.mob,
                     "orig_balance": r.orig_balance, "balance": r.balance, "balance_pct": r.balance_pct,
                     "cum_prepay_pct": r.cum_prepay_pct, "cum_default_pct": r.cum_default_pct,
                     "k": models[r.segment].k, **p})
    return pd.DataFrame(rows).sort_values(["segment", "vintage"]).reset_index(drop=True)
