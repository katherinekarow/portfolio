"""Synthetic clinic sessions with PLANTED behaviour segments.

Each session is a clinic day with a fixed number of appointment slots. Appointments
arrive over the 120 days before the session; some later cancel. Four true segments
differ in the *shape* of their demand curve, final fill and cancellation, plus a few
planted outliers. Each session also carries a "service line" label that is only
loosely related to how it books - the point of the project is that behaviour, not the
label, defines the segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 120                       # days before the session that booking opens
GRID_DAYS = np.array([120, 105, 90, 75, 60, 45, 30, 21, 14, 7, 3, 1, 0])
FEATURES = ["share_90", "share_60", "share_30", "share_7",
            "fill_rate", "cancel_rate", "volatility"]
SERVICE_LINES = ["check-up", "dental", "physio", "vaccination"]

# true segment -> (curve exponent p, final fill, cancel probability, extra noise)
# cumulative share booked by d days out: F(d) = (1 - d/120)^p
#   p < 1 books early (concave), p = 1 is linear, p > 1 books late (convex)
SEGMENTS = {
    0: dict(name="early planners", p=0.35, fill=0.96, cancel=0.10, noise=0.15),
    1: dict(name="steady", p=1.00, fill=0.84, cancel=0.06, noise=0.15),
    2: dict(name="late rush", p=3.00, fill=0.80, cancel=0.03, noise=0.15),
    3: dict(name="under-filled, erratic", p=1.10, fill=0.52, cancel=0.22, noise=0.40),
}
# planted outliers: (description, p, fill, cancel, noise, block_day)
# block_day: 45% of demand arrives as one group booking on that day
OUTLIERS = [
    ("block booking three days out", 1.0, 0.95, 0.02, 0.15, 3),
    ("block booking three days out", 1.0, 0.92, 0.02, 0.15, 3),
    ("mass cancellation", 0.45, 0.70, 0.40, 0.15, None),
    ("mass cancellation", 0.45, 0.65, 0.42, 0.15, None),
    ("near-empty", 1.3, 0.10, 0.12, 1.30, None),
    ("near-empty", 1.3, 0.09, 0.12, 1.30, None),
    ("overbooked early", 0.30, 1.50, 0.08, 0.15, None),
    ("overbooked early", 0.30, 1.55, 0.08, 0.15, None),
]


def _simulate(rng, capacity, p, fill, cancel, noise, block_day=None):
    """Daily simulation -> net cumulative bookings at every day-out 120..0, gross, cancels."""
    d = np.arange(HORIZON, -1, -1)                         # 120, 119, ..., 0
    F = (1 - d / HORIZON) ** p
    inc = np.diff(np.concatenate([[0.0], F]))              # expected share arriving each day
    if block_day is not None:                              # almost everything lands at once
        inc = 0.55 * inc
        inc[d == block_day] += 0.45
    gross_target = fill * capacity / (1 - cancel)
    mult = rng.gamma(1 / noise ** 2, noise ** 2, size=d.size)   # mean 1, sd = noise
    gross = rng.poisson(gross_target * inc * mult)
    # each booking cancels with prob `cancel`, on a uniform later day (index >= booking index)
    cancels = np.zeros_like(gross)
    for i in np.nonzero(gross)[0]:
        k = rng.binomial(gross[i], cancel)
        if k:
            when = rng.integers(i, d.size, size=k)
            np.add.at(cancels, when, 1)
    net_daily = gross - cancels
    return d, np.cumsum(net_daily), gross, cancels


def _features(d, cum, gross, cancels, capacity):
    final = cum[-1]
    share = {f"share_{x}": cum[d == x][0] / final for x in (90, 60, 30, 7)}
    # volatility: weekly net pickup vs its 3-week centred moving average, relative to the
    # mean weekly pickup (how bumpy the curve is, independent of its overall shape)
    net = (gross - cancels)[-119:].reshape(17, 7).sum(1)       # last 17 weeks
    resid = net[1:-1] - (net[:-2] + net[1:-1] + net[2:]) / 3
    vol = float(np.std(resid) / max(np.mean(np.abs(net)), 1e-9))
    return {**share, "fill_rate": final / capacity,
            "cancel_rate": cancels.sum() / max(gross.sum(), 1), "volatility": vol}


def make_sessions(n_per_segment: int = 73, seed: int = 7):
    """Return (sessions, curves).

    sessions: one row per session - id, capacity, service_line, true_segment (-1 = planted
              outlier), is_outlier, plus the 7 behaviour FEATURES.
    curves:   cumulative net bookings normalised by final, on GRID_DAYS (one column per day).
    """
    rng = np.random.default_rng(seed)
    rows, curves = [], []
    specs = []
    for seg, s in SEGMENTS.items():
        for _ in range(n_per_segment):
            specs.append((seg, s["p"] * np.exp(rng.normal(0, 0.06)),
                          np.clip(s["fill"] + rng.normal(0, 0.04), 0.3, 1.05),
                          np.clip(s["cancel"] + rng.normal(0, 0.015), 0.0, 0.5),
                          s["noise"], None))
    for desc, p, fill, cancel, noise, block in OUTLIERS:
        specs.append((-1, p, fill, cancel, noise, block))
    order = rng.permutation(len(specs))                     # shuffle so ids carry no signal
    for sid, j in enumerate(order):
        seg, p, fill, cancel, noise, block = specs[j]
        capacity = int(rng.integers(120, 321))
        d, cum, gross, canc = _simulate(rng, capacity, p, fill, cancel, noise, block)
        f = _features(d, cum, gross, canc, capacity)
        # service line: matches the segment 35% of the time, otherwise random
        line = SERVICE_LINES[seg] if (seg >= 0 and rng.random() < 0.35) else SERVICE_LINES[rng.integers(4)]
        rows.append({"session_id": f"S{sid + 1:03d}", "capacity": capacity, "service_line": line,
                     "true_segment": seg, "is_outlier": seg < 0, **f})
        curves.append(np.interp(GRID_DAYS[::-1], d[::-1], cum[::-1])[::-1] / cum[-1])
    sessions = pd.DataFrame(rows)
    curves = pd.DataFrame(np.array(curves), columns=[f"d{x}" for x in GRID_DAYS])
    curves.insert(0, "session_id", sessions.session_id)
    return sessions, curves
