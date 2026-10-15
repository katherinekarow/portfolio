"""The worked examples. All numbers are invented; random ones are seeded."""
from __future__ import annotations

import numpy as np
import pandas as pd

# 1. Farm co-op planting plan: acres of grain (x1) and vegetables (x2)
FARM = dict(
    variables=["grain_acres", "vegetable_acres"],
    profit=[300.0, 500.0],                       # $ per acre
    constraints=["land (acres)", "labour (hours)", "water (acre-ft)"],
    A=[[1, 1], [1, 3], [3, 2]],
    b=[100.0, 240.0, 270.0],
)

# 2. Transportation: three farms send produce (tonnes) to four markets, cost $ per tonne
TRANSPORT = dict(
    farms=["North farm", "Valley farm", "Ridge farm"],
    markets=["City market", "Wholesale hub", "Grocer chain", "Port packer"],
    supply=[120.0, 80.0, 100.0],
    demand=[70.0, 90.0, 60.0, 80.0],
    cost=[[8, 6, 10, 9],
          [9, 12, 13, 7],
          [14, 9, 16, 5]],
)

# 3a. Cold-storage space: which orders to accept (pallets, margin $)
def storage_orders(seed=6, n=12):
    rng = np.random.default_rng(seed)
    w = rng.integers(2, 13, n)
    v = np.round(w * rng.uniform(40, 110, n) + rng.uniform(0, 150, n), 0)
    return [f"order {i + 1:02d}" for i in range(n)], v.tolist(), w.tolist(), 40

# The textbook case where greedy-by-ratio goes wrong
GREEDY_TRAP = dict(values=[60.0, 100.0, 120.0], weights=[10, 20, 30], capacity=50)

# 3b. Clinic staffing: two-hour blocks 08-20, shift patterns (1 = on duty in that block)
BLOCKS = ["08-10", "10-12", "12-14", "14-16", "16-18", "18-20"]
SHIFTS = {
    "early (08-16, break 12-14)": [1, 1, 0, 1, 0, 0],
    "late (12-20, break 16-18)":  [0, 0, 1, 1, 0, 1],
    "mid (10-18)":                [0, 1, 1, 1, 1, 0],
    "morning half (08-12)":       [1, 1, 0, 0, 0, 0],
    "evening half (16-20)":       [0, 0, 0, 0, 1, 1],
    "split (08-10 + 16-20)":      [1, 0, 0, 0, 1, 1],
}
SHIFT_COST = [300.0, 300.0, 320.0, 170.0, 170.0, 260.0]
STAFF_NEEDED = [4, 6, 7, 5, 6, 4]

# 4. Assignment: six produce lots, six wholesale buyers, bid ($ per lot)
def lot_bids(seed=5, n=6):
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.8, 1.2, n)
    taste = rng.uniform(0.85, 1.15, (n, n))
    bids = np.round(1000 * quality[:, None] * taste + rng.normal(0, 40, (n, n)), 0)
    return [f"lot {chr(65 + i)}" for i in range(n)], [f"buyer {j + 1}" for j in range(n)], bids

# 5. Pricing a box of produce: demand 1000·e^(-0.25 p), unit cost $2
PRICING = dict(a=1000.0, b=0.25, cost=2.0, p0=3.0, step=0.005)

# Promotion budget split across sales channels: return a·ln(1 + x/s), budget $100k
CHANNELS = dict(names=["farm-stand", "online box", "grocer promo", "radio"],
                a=[120.0, 80.0, 60.0, 20.0], s=[10.0, 5.0, 20.0, 25.0], budget=100.0)

# 6. A buyer's three offers for the season's crop, payoff $k by market state
OFFERS = pd.DataFrame(
    {"low": [50.0, 30.0, 15.0], "medium": [50.0, 55.0, 60.0], "high": [50.0, 80.0, 110.0]},
    index=["fixed contract", "price-indexed", "spot market"])
OFFER_PROBS = [0.3, 0.5, 0.2]
# A market forecast: P(signal | state); rows = (weak, strong), columns = (low, medium, high)
FORECAST = [[0.85, 0.5, 0.15], [0.15, 0.5, 0.85]]
FORECAST_COST = 2.0
RISK_TOLERANCES = [10.0, 25.0, 50.0, 100.0, float("inf")]

# Newsvendor: a farm stand stocks strawberry boxes each morning; demand ~ Poisson(40)
NEWSVENDOR = dict(lam=40.0, price=6.0, cost=2.5, salvage=0.5)
