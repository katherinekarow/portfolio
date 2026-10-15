"""Decisions under uncertainty: expected value, decision trees, value of information,
the newsvendor critical ratio, risk aversion and minimax regret."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------- payoff-table criteria ----------
def criteria(payoff: pd.DataFrame, probs) -> pd.DataFrame:
    """Rows = options, columns = states of the world. Scores every option under five rules."""
    p = np.asarray(probs, float)
    P = payoff.to_numpy(float)
    regret = P.max(axis=0) - P
    out = pd.DataFrame({
        "expected_value": P @ p,
        "worst_case": P.min(axis=1),          # maximin (pessimist)
        "best_case": P.max(axis=1),           # maximax (optimist)
        "max_regret": regret.max(axis=1),     # minimax regret
        "std_dev": np.sqrt(((P - (P @ p)[:, None]) ** 2) @ p),
    }, index=payoff.index)
    return out


def regret_table(payoff: pd.DataFrame) -> pd.DataFrame:
    return payoff.max(axis=0) - payoff


def evpi(payoff: pd.DataFrame, probs) -> dict:
    """EVPI = E[best payoff if you knew the state first] − best expected payoff.

    It is the most a decision-maker should pay for any forecast, and always >= 0.
    """
    p = np.asarray(probs, float)
    P = payoff.to_numpy(float)
    ev = P @ p
    ev_pi = float(P.max(axis=0) @ p)
    return {"best_option": payoff.index[int(np.argmax(ev))], "ev_best": float(ev.max()),
            "ev_with_perfect_info": ev_pi, "evpi": ev_pi - float(ev.max())}


# ---------- utility / risk aversion ----------
def certainty_equivalent(values, probs, risk_tolerance):
    """Exponential utility u(x) = 1 − e^(−x/R). CE = −R·ln(Σ p·e^(−x/R)).

    The CE is the sure amount the decision-maker would swap the gamble for. Smaller R
    means more risk-averse; as R → ∞ the CE becomes the expected value.
    """
    v = np.asarray(values, float)
    p = np.asarray(probs, float)
    if math.isinf(risk_tolerance):
        return float(v @ p)
    R = float(risk_tolerance)
    shift = v.min()                       # for numerical stability
    return float(shift - R * math.log(p @ np.exp(-(v - shift) / R)))


# ---------- decision trees ----------
def end(value):
    return {"kind": "end", "value": float(value)}


def chance(branches):
    """branches = [(label, probability, node), ...]"""
    return {"kind": "chance", "branches": list(branches)}


def decide(options):
    """options = [(label, node), ...]"""
    return {"kind": "decision", "options": list(options)}


def rollback(node):
    """Fold the tree back from the leaves: chance node = probability-weighted mean of its
    branches; decision node = the best branch. Stores 'ev' (and 'best' on decisions)."""
    if node["kind"] == "end":
        node["ev"] = node["value"]
    elif node["kind"] == "chance":
        total = sum(p for _, p, _ in node["branches"])
        if abs(total - 1) > 1e-9:
            raise ValueError(f"chance probabilities sum to {total}")
        node["ev"] = sum(p * rollback(child) for _, p, child in node["branches"])
    else:
        evs = [rollback(child) for _, child in node["options"]]
        k = int(np.argmax(evs))
        node["ev"], node["best"] = evs[k], node["options"][k][0]
    return node["ev"]


def policy(node, prefix=""):
    """The chosen action at every decision node, as {path: action}."""
    out = {}
    if node["kind"] == "decision":
        out[prefix or "start"] = node["best"]
        for label, child in node["options"]:
            if label == node["best"]:
                out.update(policy(child, f"{prefix}{label} > "))
    elif node["kind"] == "chance":
        for label, _, child in node["branches"]:
            out.update(policy(child, f"{prefix}{label} > "))
    return out


def posterior(prior, likelihood):
    """Bayes: P(state | signal) ∝ P(signal | state)·P(state). likelihood[k, s] = P(signal k | state s)."""
    prior = np.asarray(prior, float)
    L = np.asarray(likelihood, float)
    joint = L * prior[None, :]
    p_signal = joint.sum(axis=1)
    return p_signal, joint / p_signal[:, None]


def offer_tree(payoff: pd.DataFrame, probs, forecast_likelihood=None, forecast_cost=0.0,
               signal_names=("forecast weak", "forecast strong")):
    """Choose an offer now, or first buy a market forecast and then choose."""
    states = list(payoff.columns)

    def offers(p, minus=0.0):
        return decide([(opt, chance([(s, float(p[k]), end(payoff.loc[opt, s] - minus))
                                     for k, s in enumerate(states)])) for opt in payoff.index])

    if forecast_likelihood is None:
        return offers(probs)
    p_sig, post = posterior(probs, forecast_likelihood)
    survey = chance([(signal_names[k], float(p_sig[k]), offers(post[k], forecast_cost))
                     for k in range(len(p_sig))])
    return decide([("decide now", offers(probs)), ("buy forecast", survey)])


# ---------- newsvendor ----------
def critical_ratio(price, cost, salvage=0.0):
    """Stock until P(demand <= Q) reaches Cu/(Cu + Co).

    Cu = price − cost  (margin lost per unit short)
    Co = cost − salvage (loss per unit left over)
    """
    cu, co = price - cost, cost - salvage
    return cu / (cu + co)


def poisson_cdf(k, lam):
    k = np.atleast_1d(np.asarray(k))
    out = []
    for kk in k:
        term, total = math.exp(-lam), 0.0
        for i in range(int(kk) + 1):
            if i > 0:
                term *= lam / i
            total += term
        out.append(total)
    return np.array(out)


def newsvendor_poisson(lam, price, cost, salvage=0.0):
    """Smallest Q with F(Q) >= critical ratio."""
    cr = critical_ratio(price, cost, salvage)
    q = 0
    while poisson_cdf(q, lam)[0] < cr:
        q += 1
    return q, cr


def newsvendor_profit(q, demand, price, cost, salvage=0.0):
    """Profit for stocking q against each demand value (vectorised over demand)."""
    d = np.asarray(demand, float)
    sold = np.minimum(q, d)
    return price * sold + salvage * (q - sold) - cost * q


def newsvendor_exact_curve(lam, price, cost, salvage, q_values, kmax=None):
    kmax = kmax or int(lam + 12 * math.sqrt(lam) + 20)
    k = np.arange(kmax + 1)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in k])
    pmf = np.exp(logp)
    return np.array([newsvendor_profit(q, k, price, cost, salvage) @ pmf for q in q_values])


def newsvendor_simulated(lam, price, cost, salvage, q_values, n=200_000, seed=7):
    d = np.random.default_rng(seed).poisson(lam, n)
    return np.array([newsvendor_profit(q, d, price, cost, salvage).mean() for q in q_values])


# ---------- sensitivity ----------
def with_p_high(probs, p_high):
    """Set P(last state) and rescale the others proportionally so the total stays 1."""
    p = np.asarray(probs, float).copy()
    rest = p[:-1] / p[:-1].sum() * (1 - p_high)
    return np.append(rest, p_high)


def one_way(payoff: pd.DataFrame, probs, changes: dict) -> pd.DataFrame:
    """changes = {name: (low_fn, high_fn)} where each fn returns (payoff, probs).
    Records the expected value of each option and the best option at the low and high setting."""
    rows = []
    base = criteria(payoff, probs).expected_value
    for name, fns in changes.items():
        for level, fn in zip(("low", "high"), fns):
            P, p = fn(payoff.copy(), np.asarray(probs, float).copy())
            ev = criteria(P, p).expected_value
            rows.append({"input": name, "level": level, **{f"ev_{k}": round(v, 3) for k, v in ev.items()},
                         "best_option": ev.idxmax(), "best_ev": round(ev.max(), 3),
                         "swing_vs_base": round(ev.max() - base.max(), 3)})
    return pd.DataFrame(rows)


def two_way(payoff: pd.DataFrame, probs, p_high_grid, fixed_grid, fixed_option):
    """Best option over a grid of (P(high state), fixed-offer amount)."""
    rows = []
    for ph in p_high_grid:
        for fx in fixed_grid:
            P = payoff.copy()
            P.loc[fixed_option, :] = fx
            ev = criteria(P, with_p_high(probs, ph)).expected_value
            rows.append({"p_high": round(float(ph), 4), "fixed_offer": round(float(fx), 4),
                         "best_option": ev.idxmax(), "best_ev": round(float(ev.max()), 3)})
    return pd.DataFrame(rows)
