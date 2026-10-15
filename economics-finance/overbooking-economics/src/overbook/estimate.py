"""Estimating show-up rates from history, with empirical-Bayes shrinkage, and what errors cost.

Synthetic history: 12 clinics x 2 session types (new-patient, follow-up). Each clinic-type has a true
show rate drawn around its type mean; each session's rate wobbles around that (beta-binomial, rho).
Some clinics have years of sessions, some only a handful.

Fit, per session type:
 1. rho (within-session over-dispersion): beta-binomial maximum likelihood on session counts, with each
    clinic's mean held at its raw rate (profile likelihood over a single rho).
 2. Design effect deff = 1 + (n_bar - 1) rho: correlated show-ups carry less information, so each
    clinic's totals are divided by deff to give an effective sample size.
 3. Between-clinic prior Beta(m * k, (1 - m) * k): beta-binomial marginal likelihood of the effective
    clinic totals, maximised over (m, k).
 4. Posterior mean per clinic  p_i = (shows_i/deff + k m) / (booked_i/deff + k): small clinics lean on m.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize as so
from scipy.special import betaln, gammaln

from .profit import optimal_limit, profit_curve
from .showup import sample_shows

TYPES = {"new patient": 0.80, "follow-up": 0.90}
CLINIC_SESSIONS = [260, 220, 180, 150, 120, 96, 72, 48, 30, 18, 10, 5]
KAPPA_TRUE, RHO_TRUE, BOOKED = 120.0, 0.03, 22


def make_history(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session-level history (clinic, type, booked, shows) and the true rates behind it."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for t, mean in TYPES.items():
        for i, ns in enumerate(CLINIC_SESSIONS):
            clinic = f"Clinic {chr(65 + i)}"
            p = rng.beta(mean * KAPPA_TRUE, (1 - mean) * KAPPA_TRUE)
            n = np.full(ns, BOOKED)
            s = sample_shows(rng, n, p, RHO_TRUE)
            rows.append(pd.DataFrame({"clinic": clinic, "type": t, "booked": n, "shows": s}))
            truth.append({"clinic": clinic, "type": t, "true_p": p})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(truth)


def _betabin_ll(k, n, a, b):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + betaln(k + a, n - k + b) - betaln(a, b)


def fit_rho(sessions: pd.DataFrame) -> float:
    """Profile MLE of the within-session correlation, clinic means fixed at their raw rates."""
    p = sessions.groupby("clinic").shows.transform("sum") / sessions.groupby("clinic").booked.transform("sum")
    p = p.clip(1e-3, 1 - 1e-3).to_numpy()
    k, n = sessions.shows.to_numpy(float), sessions.booked.to_numpy(float)

    def nll(lr):
        rho = 1 / (1 + np.exp(-lr))
        c = (1 - rho) / rho
        return -_betabin_ll(k, n, p * c, (1 - p) * c).sum()
    res = so.minimize_scalar(nll, bounds=(-10, 0), method="bounded")
    return float(1 / (1 + np.exp(-res.x)))


def fit_prior(shows: np.ndarray, booked: np.ndarray) -> tuple[float, float]:
    """Between-clinic Beta prior (mean m, strength k) by marginal beta-binomial likelihood."""
    def nll(x):
        m, k = 1 / (1 + np.exp(-x[0])), np.exp(x[1])
        return -_betabin_ll(shows, booked, m * k, (1 - m) * k).sum()
    m0 = shows.sum() / booked.sum()
    res = so.minimize(nll, [np.log(m0 / (1 - m0)), np.log(50)], method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    return float(1 / (1 + np.exp(-res.x[0]))), float(min(np.exp(res.x[1]), 1e6))


def estimate(hist: pd.DataFrame) -> pd.DataFrame:
    """Raw and shrunk show rate per clinic and session type."""
    out = []
    for t, g in hist.groupby("type", sort=False):
        rho = fit_rho(g)
        deff = 1 + (g.booked.mean() - 1) * rho
        tot = g.groupby("clinic", sort=False).agg(sessions=("booked", "size"), booked=("booked", "sum"),
                                                  shows=("shows", "sum")).reset_index()
        se, ne = tot.shows / deff, tot.booked / deff
        m, k = fit_prior(se.to_numpy(), ne.to_numpy())
        tot["type"] = t
        tot["raw_p"] = tot.shows / tot.booked
        tot["weight_on_own_data"] = ne / (ne + k)
        tot["shrunk_p"] = (se + k * m) / (ne + k)
        tot["prior_mean"], tot["prior_strength"], tot["rho_hat"], tot["design_effect"] = m, k, rho, deff
        out.append(tot)
    return pd.concat(out, ignore_index=True)


def regret_by_clinic(est: pd.DataFrame, truth: pd.DataFrame, C: int, r: float, b: float) -> pd.DataFrame:
    """Annual profit lost by planning each clinic's limit with raw vs shrunk vs binomial-only estimates."""
    d = est.merge(truth, on=["clinic", "type"])
    rows = []
    for _, x in d.iterrows():
        cur = profit_curve(C, x.true_p, r, b, RHO_TRUE).profit.to_numpy()
        best = optimal_limit(C, x.true_p, r, b, RHO_TRUE)["limit"]
        plans = {"true": best,
                 "raw": optimal_limit(C, x.raw_p, r, b, x.rho_hat)["limit"],
                 "shrunk": optimal_limit(C, x.shrunk_p, r, b, x.rho_hat)["limit"],
                 "shrunk_no_dispersion": optimal_limit(C, x.shrunk_p, r, b, 0.0)["limit"]}
        row = {"clinic": x.clinic, "type": x["type"], "sessions": x.sessions}
        for k, L in plans.items():
            row[f"limit_{k}"] = L
            row[f"regret_{k}"] = (cur[best] - cur[min(L, len(cur) - 1)]) * x.sessions
        rows.append(row)
    return pd.DataFrame(rows)


def regret_curve(C: int, p_true: float, r: float, b: float, p_hats: np.ndarray, rho: float = 0.0) -> pd.DataFrame:
    """Per-session profit lost when the limit is planned with p_hat but patients show at p_true."""
    cur = profit_curve(C, p_true, r, b, rho)
    prof = cur.profit.to_numpy()
    best = optimal_limit(C, p_true, r, b, rho)["limit"]
    rows = []
    for ph in p_hats:
        L = min(optimal_limit(C, ph, r, b, rho)["limit"], len(prof) - 1)
        rows.append({"p_hat": ph, "limit": L, "regret": prof[best] - prof[L]})
    return pd.DataFrame(rows)
