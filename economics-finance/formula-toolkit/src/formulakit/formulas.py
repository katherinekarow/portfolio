"""The formulas. Plain Python (no NumPy) so each one ports line-for-line to the browser.

Groups:
  finance  - compound growth, present value, NPV, IRR, payback, loan payment & amortisation, CAGR
  macro    - money multiplier, reserve ratio, real vs nominal rates (approximate and exact Fisher)
  demand   - arc and point elasticity, log-log elasticity fit, constant-elasticity demand,
             revenue-maximising and profit-maximising price, cross-price elasticity
  curves   - fill rate, normalisation against a benchmark, capacity-weighted curve blend,
             percentile reference curves, demand velocity, residual-demand allocation
  targets  - cancellation survival, gross target from a net target, expected net from gross
  stats    - credibility weighting, shrinkage, variance tolerance gate, reconciliation match, WAPE
"""
from __future__ import annotations

import math

from .registry import formula

# =============================================================================== finance


@formula("future_value", "finance", "Compound growth (future value)",
         "FV = PV × (1 + r)^n",
         "What an amount grows to after n periods at rate r per period.",
         [("pv", "Present value", 1000), ("rate", "Rate per period", 0.05), ("n", "Periods", 10)], out="money")
def future_value(pv, rate, n):
    return pv * (1 + rate) ** n


@formula("present_value", "finance", "Present value",
         "PV = FV ÷ (1 + r)^n",
         "What a future amount is worth today when money can earn r per period.",
         [("fv", "Future value", 1628.89), ("rate", "Rate per period", 0.05), ("n", "Periods", 10)], out="money")
def present_value(fv, rate, n):
    return fv / (1 + rate) ** n


@formula("growth_rate", "finance", "Growth rate between two values (CAGR)",
         "r = (end ÷ start)^(1/n) − 1",
         "The constant rate per period that turns the start value into the end value.",
         [("start", "Start value", 1000), ("end", "End value", 1628.89), ("n", "Periods", 10)], out="pct")
def growth_rate(start, end, n):
    return (end / start) ** (1 / n) - 1


@formula("npv", "finance", "Net present value",
         "NPV = Σ CFₜ ÷ (1 + r)^t,  t = 0, 1, 2 …",
         "Today's value of a stream of cash flows; the first flow is at t = 0 (usually the investment, negative).",
         [("rate", "Discount rate per period", 0.08), ("flows", "Cash flows (t = 0, 1, …)", "-1000, 300, 400, 500, 200", "list")],
         out="money")
def npv(rate, flows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


@formula("irr", "finance", "Internal rate of return",
         "IRR = the r where NPV(r) = 0",
         "The break-even discount rate. Found by bisection, so it needs one sign change in NPV between −99% and 1,000%.",
         [("flows", "Cash flows (t = 0, 1, …)", "-1000, 300, 400, 500, 200", "list")], out="pct")
def irr(flows):
    lo, hi = -0.99, 10.0
    f_lo, f_hi = npv(lo, flows), npv(hi, flows)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid, flows)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


@formula("payback", "finance", "Payback period",
         "first t where Σ CF₀…CFₜ ≥ 0 (interpolated within the period)",
         "How long until cumulative cash flow turns positive. Ignores the time value of money; read it next to NPV.",
         [("flows", "Cash flows (t = 0, 1, …)", "-1000, 300, 400, 500, 200", "list")])
def payback(flows):
    cum = 0.0
    for t, cf in enumerate(flows):
        if cum + cf >= 0 and t > 0:
            return t - 1 + (-cum / cf if cf else 0)
        cum += cf
    return None


@formula("loan_payment", "finance", "Loan payment (annuity)",
         "PMT = P × r ÷ (1 − (1 + r)^−n)",
         "The level payment that repays principal P over n periods at rate r per period.",
         [("principal", "Principal", 20000), ("rate", "Rate per period", 0.01), ("n", "Number of payments", 36)], out="money")
def loan_payment(principal, rate, n):
    if rate == 0:
        return principal / n
    return principal * rate / (1 - (1 + rate) ** -n)


@formula("amortization", "finance", "Amortisation schedule",
         "interestₜ = balanceₜ₋₁ × r;  principalₜ = PMT − interestₜ;  balanceₜ = balanceₜ₋₁ − principalₜ",
         "Splits every payment into interest and principal. Principal paid always adds up to the amount borrowed.",
         [("principal", "Principal", 20000), ("rate", "Rate per period", 0.01), ("n", "Number of payments", 36, "int")], out="table")
def amortization(principal, rate, n):
    pmt, bal, rows = loan_payment(principal, rate, n), principal, []
    for t in range(1, int(n) + 1):
        interest = bal * rate
        princ = pmt - interest
        bal -= princ
        rows.append({"period": t, "payment": pmt, "interest": interest, "principal": princ, "balance": max(bal, 0.0)})
    return rows


@formula("effective_rate", "finance", "Effective annual rate",
         "EAR = (1 + APR ÷ m)^m − 1",
         "The true yearly rate when a nominal rate is compounded m times a year.",
         [("apr", "Nominal annual rate", 0.12), ("m", "Compounding periods per year", 12)], out="pct")
def effective_rate(apr, m):
    return (1 + apr / m) ** m - 1


# =============================================================================== macro


@formula("money_multiplier", "macro", "Money multiplier",
         "m = 1 ÷ reserve ratio",
         "How much broad money a unit of reserves can support in the simple textbook model.",
         [("reserve_ratio", "Reserve ratio", 0.10)])
def money_multiplier(reserve_ratio):
    return 1 / reserve_ratio


@formula("reserve_ratio", "macro", "Reserve ratio",
         "rr = 1 ÷ money multiplier",
         "The inverse of the money multiplier.",
         [("multiplier", "Money multiplier", 10)], out="pct")
def reserve_ratio(multiplier):
    return 1 / multiplier


@formula("real_rate", "macro", "Real interest rate (exact Fisher)",
         "real = (1 + nominal) ÷ (1 + inflation) − 1   (≈ nominal − inflation)",
         "Purchasing-power return. The subtraction shortcut is close only when rates are small.",
         [("nominal", "Nominal rate", 0.07), ("inflation", "Inflation rate", 0.03)], out="pct")
def real_rate(nominal, inflation):
    return (1 + nominal) / (1 + inflation) - 1


@formula("nominal_rate", "macro", "Nominal interest rate (exact Fisher)",
         "nominal = (1 + real) × (1 + inflation) − 1",
         "The rate needed to earn a given real return when prices rise.",
         [("real", "Real rate", 0.0388), ("inflation", "Inflation rate", 0.03)], out="pct")
def nominal_rate(real, inflation):
    return (1 + real) * (1 + inflation) - 1


# =============================================================================== demand


@formula("arc_elasticity", "demand", "Price elasticity (arc / midpoint)",
         "e = [(Q₂ − Q₁) ÷ ((Q₁ + Q₂)/2)] ÷ [(P₂ − P₁) ÷ ((P₁ + P₂)/2)]",
         "% change in quantity per % change in price between two observed points. The midpoint base gives the same answer in either direction.",
         [("p1", "Price before", 20), ("q1", "Quantity before", 1000), ("p2", "Price after", 22), ("q2", "Quantity after", 880)])
def arc_elasticity(p1, q1, p2, q2):
    return ((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2))


@formula("point_elasticity", "demand", "Price elasticity (point, linear demand)",
         "e = (dQ/dP) × (P ÷ Q)",
         "Elasticity at one point on a demand line with slope dQ/dP.",
         [("slope", "Slope dQ/dP", -60), ("p", "Price", 20), ("q", "Quantity", 1000)])
def point_elasticity(slope, p, q):
    return slope * p / q


@formula("fit_elasticity", "demand", "Elasticity from data (log-log regression)",
         "ln Q = a + e × ln P   →   e = cov(ln P, ln Q) ÷ var(ln P)",
         "Estimates a constant elasticity from several price/quantity observations by least squares on logs.",
         [("prices", "Prices", "18, 19, 20, 21, 22, 23", "list"), ("quantities", "Quantities", "1180, 1090, 1000, 935, 880, 820", "list")])
def fit_elasticity(prices, quantities):
    x = [math.log(p) for p in prices]
    y = [math.log(q) for q in quantities]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    return sxy / sxx


@formula("demand_at_price", "demand", "Demand at a new price (constant elasticity)",
         "Q₁ = Q₀ × (P₁ ÷ P₀)^e",
         "Projects volume after a price change using an elasticity e (negative for normal goods).",
         [("q0", "Current quantity", 1000), ("p0", "Current price", 20), ("p1", "New price", 22), ("e", "Elasticity", -1.3)])
def demand_at_price(q0, p0, p1, e):
    return q0 * (p1 / p0) ** e


@formula("revenue_change", "demand", "Revenue effect of a price change",
         "ΔRevenue % = (1 + Δp)^(1 + e) − 1",
         "With constant elasticity: raising price lifts revenue only when demand is inelastic (|e| < 1).",
         [("price_change", "Price change (e.g. 0.10 = +10%)", 0.10), ("e", "Elasticity", -1.3)], out="pct")
def revenue_change(price_change, e):
    return (1 + price_change) ** (1 + e) - 1


@formula("optimal_price", "demand", "Profit-maximising price (markup rule)",
         "P* = c × e ÷ (1 + e)   (needs e < −1);  Lerner: (P − c) ÷ P = −1 ÷ e",
         "With constant elasticity e and unit cost c, the best price is a fixed markup over cost; the more elastic the demand, the smaller the markup.",
         [("cost", "Unit cost", 12), ("e", "Elasticity", -2.5)], out="money")
def optimal_price(cost, e):
    if e >= -1:
        return None          # inelastic: profit keeps rising with price - no interior optimum
    return cost * e / (1 + e)


@formula("capacity_price", "demand", "Price that just fills capacity",
         "P = P₀ × (capacity ÷ Q₀)^(1/e)",
         "When capacity is fixed and marginal cost is near zero, revenue is maximised at the price where demand meets capacity (if demand is elastic) - the core trade-off in pricing perishable inventory.",
         [("q0", "Demand at current price", 1200), ("p0", "Current price", 20), ("capacity", "Capacity", 1000), ("e", "Elasticity", -1.5)], out="money")
def capacity_price(q0, p0, capacity, e):
    return p0 * (capacity / q0) ** (1 / e)


@formula("cross_elasticity", "demand", "Cross-price elasticity",
         "e_AB = %ΔQ_A ÷ %ΔP_B (midpoint)",
         "How demand for A responds to B's price: positive = substitutes, negative = complements. Used to check that a price move on one grade doesn't just shift volume between grades.",
         [("qa1", "Quantity of A before", 500), ("qa2", "Quantity of A after", 540), ("pb1", "Price of B before", 30), ("pb2", "Price of B after", 33)])
def cross_elasticity(qa1, qa2, pb1, pb2):
    return arc_elasticity(pb1, qa1, pb2, qa2)


# =============================================================================== curves


@formula("fill_rate", "curves", "Fill rate",
         "fill = units sold (or booked) ÷ capacity",
         "The common unit that makes products or sessions of different sizes comparable.",
         [("units", "Units sold", 450), ("capacity", "Capacity", 2000)], out="pct")
def fill_rate(units, capacity):
    return units / capacity


@formula("normalized_position", "curves", "Position against a benchmark",
         "position = fill today ÷ benchmark fill at the same lead time",
         "1.00 means exactly on the benchmark path; 0.80 means 20% behind where comparable items stood at this point.",
         [("fill_today", "Fill today", 0.225), ("benchmark", "Benchmark fill at this lead time", 0.30)])
def normalized_position(fill_today, benchmark):
    return fill_today / benchmark


@formula("blend_curves", "curves", "Capacity-weighted curve blend",
         "blend(t) = Σ wᵢ × curveᵢ(t),  wᵢ = capacityᵢ ÷ Σ capacity",
         "Combines the curves of components that sell differently (e.g. tiers or channels) into one curve for the whole product, weighting each by its share of capacity.",
         [("curves", "Component fills at one lead time", "0.35, 0.28, 0.42", "list"), ("capacities", "Component capacities", "1200, 600, 200", "list")])
def blend_curves(curves, capacities):
    tot = sum(capacities)
    return sum(c * k / tot for c, k in zip(curves, capacities))


@formula("percentile", "curves", "Benchmark percentile",
         "linear-interpolated percentile of comparable items' fill at the same lead time (as SQL PERCENTILE_CONT)",
         "Turns history into strategy choices: P25 = conservative target, P50 = typical, P75 = aggressive.",
         [("values", "Fill of comparable items", "0.22, 0.24, 0.19, 0.28, 0.26, 0.31, 0.21", "list"), ("q", "Percentile (0-1)", 0.6)], out="pct")
def percentile(values, q):
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


@formula("monotone_curve", "curves", "Make a cumulative curve non-decreasing",
         "curve'(t) = max(curve(t), curve'(t−1)) walking toward the event",
         "Percentiles taken lead time by lead time can dip; a cumulative target must never go down, so each point is lifted to the previous maximum.",
         [("curve", "Curve (far → near)", "0.05, 0.12, 0.11, 0.25, 0.24, 0.40, 0.62", "list")], out="list")
def monotone_curve(curve):
    out, m = [], -math.inf
    for v in curve:
        m = max(m, v)
        out.append(m)
    return out


@formula("velocity", "curves", "Demand velocity (share of final)",
         "share(t) = cumulative units at t ÷ final units",
         "What fraction of the eventual total has normally arrived by each lead time. Dividing today's units by it gives a pace-based projection.",
         [("cumulative", "Cumulative units (far → near)", "40, 110, 260, 520, 800, 950, 1000", "list")], out="list")
def velocity(cumulative):
    final = cumulative[-1]
    return [c / final for c in cumulative]


@formula("pace_projection", "curves", "Pace projection",
         "projected final = units today ÷ typical share of final at this lead time",
         "If comparable items normally have 35% of their final demand by now, 350 today projects to 1,000.",
         [("units_today", "Units today", 350), ("share", "Typical share of final by now", 0.35)])
def pace_projection(units_today, share):
    return units_today / share


@formula("residual_allocation", "curves", "Residual demand allocation (glide path)",
         "extraₜ = gap × (ref(t) − ref(t−1)) ÷ (ref(end) − ref(start)), rounded so the parts sum exactly to the gap",
         "Spreads the units still needed to hit target over the remaining days in the same shape demand normally arrives - more near the event, little far out. Integer rounding uses largest remainders.",
         [("gap", "Units still needed", 300), ("reference", "Reference cumulative curve over the remaining window", "0.30, 0.34, 0.40, 0.48, 0.60, 0.76, 1.00", "list")], out="list")
def residual_allocation(gap, reference):
    inc = [b - a for a, b in zip(reference[:-1], reference[1:])]
    tot = sum(inc)
    raw = [gap * i / tot for i in inc]
    base = [math.floor(r) for r in raw]
    short = int(round(gap)) - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:short]:
        base[i] += 1
    return base


# =============================================================================== targets


@formula("survival_rate", "targets", "Survival (show-up / keep) rate",
         "survival = net units ÷ gross units;  cancellation rate = 1 − survival",
         "Of everything taken so far, the share that is still there (not cancelled) at the end.",
         [("net", "Net units", 860), ("gross", "Gross units", 1000)], out="pct")
def survival_rate(net, gross):
    return net / gross


@formula("gross_target", "targets", "Gross target from a net target",
         "gross target = net target ÷ survival",
         "If 14% of what you take will cancel, you must take 1,000 ÷ 0.86 ≈ 1,163 to keep 1,000.",
         [("net_target", "Net target", 1000), ("survival", "Expected survival", 0.86)])
def gross_target(net_target, survival):
    return net_target / survival


@formula("expected_net", "targets", "Expected net from today's gross",
         "expected net = gross today × survival from this lead time",
         "Converts today's gross position into what will remain, using the survival rate observed for this lead time (later units cancel less).",
         [("gross_today", "Gross today", 740), ("survival_from_here", "Survival from this lead time", 0.89)])
def expected_net(gross_today, survival_from_here):
    return gross_today * survival_from_here


@formula("gross_still_needed", "targets", "Gross still needed today",
         "needed = (net target − expected net from current gross) ÷ survival of new units",
         "Bridges the two views: how many more gross units to take from now on to land on the net target.",
         [("net_target", "Net target", 1000), ("gross_today", "Gross today", 740), ("survival_existing", "Survival of current units", 0.89), ("survival_new", "Survival of new units", 0.93)])
def gross_still_needed(net_target, gross_today, survival_existing, survival_new):
    return max(0.0, (net_target - gross_today * survival_existing) / survival_new)


# =============================================================================== stats


@formula("credibility_weight", "stats", "Credibility weight",
         "Z = n ÷ (n + k)",
         "How much to trust an item's own data versus the group benchmark. n = evidence (exposure, observations); k = noise ÷ between-item spread.",
         [("n", "Evidence (n)", 40), ("k", "Credibility constant (k)", 60)], out="pct")
def credibility_weight(n, k):
    return n / (n + k)


@formula("credibility_blend", "stats", "Credibility-weighted estimate",
         "estimate = Z × own + (1 − Z) × benchmark",
         "Blends an item's own pace with its segment benchmark; early on the benchmark dominates, later the item's own data does.",
         [("own", "Own estimate", 1150), ("benchmark", "Benchmark estimate", 1000), ("z", "Credibility weight Z", 0.4)])
def credibility_blend(own, benchmark, z):
    return z * own + (1 - z) * benchmark


@formula("buhlmann_k", "stats", "Credibility constant from data (Bühlmann)",
         "k = expected process variance ÷ variance of hypothetical means",
         "Estimated from groups of observations: noisy items (high within-variance) need more evidence before they are trusted. Each group is entered as values separated by '|'.",
         [("groups", "Groups (values; groups split by |)", "10,12,11,13 | 15,17,16,14 | 9,8,10,9", "groups")])
def buhlmann_k(groups):
    groups = [g for g in groups if len(g) > 1]
    n = [len(g) for g in groups]
    means = [sum(g) / len(g) for g in groups]
    within = sum(sum((x - m) ** 2 for x in g) / (len(g) - 1) for g, m in zip(groups, means)) / len(groups)
    grand = sum(sum(g) for g in groups) / sum(n)
    nbar = sum(n) / len(n)
    between = sum((m - grand) ** 2 for m in means) / (len(groups) - 1) - within / nbar
    return None if between <= 0 else within / between


@formula("shrink_rate", "stats", "Shrinking a small sample's rate",
         "rate = (successes + m × prior) ÷ (trials + m)",
         "Empirical-Bayes style: a 3-out-of-4 record is pulled toward the group rate; a 300-out-of-400 record barely moves. m is the prior's weight in pseudo-observations.",
         [("successes", "Successes", 3), ("trials", "Trials", 4), ("prior", "Group rate", 0.55), ("m", "Prior weight (pseudo-trials)", 20)], out="pct")
def shrink_rate(successes, trials, prior, m):
    return (successes + m * prior) / (trials + m)


@formula("variance_gate", "stats", "Change tolerance gate",
         "|new − old| ÷ max(|old|, 1) ≤ tolerance  →  pass",
         "The check run before overwriting a published table: if row counts or totals move more than the tolerance (e.g. 2%), stop and investigate instead of publishing.",
         [("old", "Previous value", 12450), ("new", "New value", 12610), ("tolerance", "Tolerance", 0.02)], out="gate")
def variance_gate(old, new, tolerance):
    change = abs(new - old) / max(abs(old), 1)
    return {"change": change, "pass": change <= tolerance}


@formula("reconcile_match", "stats", "Reconciliation match within tolerance",
         "match if |a − b| ÷ max(|a|, |b|) ≤ tolerance",
         "Two systems rarely agree to the cent (rounding, currency). A relative tolerance separates real breaks from noise.",
         [("a", "Value in system A", 1049.50), ("b", "Value in system B", 1000.00), ("tolerance", "Tolerance", 0.05)], out="gate")
def reconcile_match(a, b, tolerance):
    diff = abs(a - b) / max(abs(a), abs(b), 1e-12)
    return {"change": diff, "pass": diff <= tolerance}


@formula("wape", "stats", "Forecast error (WAPE)",
         "WAPE = Σ|actual − forecast| ÷ Σ|actual|",
         "Volume-weighted error: unlike MAPE it isn't blown up by tiny actuals.",
         [("actual", "Actuals", "120, 80, 15, 300", "list"), ("forecast", "Forecasts", "110, 95, 5, 310", "list")], out="pct")
def wape(actual, forecast):
    return sum(abs(a - f) for a, f in zip(actual, forecast)) / sum(abs(a) for a in actual)
