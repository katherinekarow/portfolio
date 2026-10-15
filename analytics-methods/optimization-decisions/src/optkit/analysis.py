"""Runs every worked example and returns tables plus a summary of the headline numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import assignment, decisions, integer, lp, nonlinear, transport
from . import scenarios as S


def farm_lp():
    f = S.FARM
    r = lp.simplex(f["profit"], f["A"], f["b"], maximize=True)
    var = pd.DataFrame({
        "variable": f["variables"], "value": r.x, "profit_per_acre": f["profit"],
        "reduced_cost": r.reduced_costs + 0.0,
        "profit_range_low": r.cost_range[:, 0], "profit_range_high": r.cost_range[:, 1]})
    con = pd.DataFrame({
        "constraint": f["constraints"], "available": f["b"],
        "used": np.asarray(f["A"]) @ r.x, "slack": r.slack_ub + 0.0,
        "shadow_price": r.shadow_ub + 0.0,
        "rhs_range_low": r.rhs_range_ub[:, 0], "rhs_range_high": r.rhs_range_ub[:, 1]})
    path = pd.DataFrame([(k, *x, float(np.dot(f["profit"], x))) for k, x in enumerate(r.path)],
                        columns=["pivot", *f["variables"], "profit"])
    # one-way sweep of grain profit per acre: the plan only changes at the ends of the range
    sweep = []
    for c1 in np.arange(100, 701, 25):
        rr = lp.simplex([c1, f["profit"][1]], f["A"], f["b"], maximize=True)
        sweep.append({"grain_profit_per_acre": c1, "grain_acres": round(rr.x[0], 6),
                      "vegetable_acres": round(rr.x[1], 6), "total_profit": round(rr.objective, 6)})
    return r, var.round(6), con.round(6), path.round(6), pd.DataFrame(sweep)


def transport_example():
    t = S.TRANSPORT
    plan, total, u, v, red = transport.solve(t["cost"], t["supply"], t["demand"])
    nw = transport.northwest_corner(t["supply"], t["demand"])
    nw_cost = float((np.asarray(t["cost"]) * nw).sum())
    rows = []
    for i, farm in enumerate(t["farms"]):
        for j, mk in enumerate(t["markets"]):
            rows.append({"farm": farm, "market": mk, "cost_per_tonne": t["cost"][i][j],
                         "tonnes": round(plan[i, j], 6), "reduced_cost": round(red[i, j] + 0.0, 6)})
    return pd.DataFrame(rows), total, nw_cost


def knapsack_examples(n_instances=1000, seed=2024):
    names, v, w, cap = S.storage_orders()
    dp_val, dp_pick = integer.knapsack_dp(v, w, cap)
    gr_val, gr_pick = integer.knapsack_greedy(v, w, cap)
    orders = pd.DataFrame({"order": names, "pallets": w, "margin": v,
                           "margin_per_pallet": np.round(np.array(v) / np.array(w), 2),
                           "in_dp_optimum": [i in dp_pick for i in range(len(v))],
                           "in_greedy": [i in gr_pick for i in range(len(v))]})
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_instances):
        vv, ww, cc = integer.random_knapsack(rng)
        a, _ = integer.knapsack_dp(vv, ww, cc)
        g, _ = integer.knapsack_greedy(vv, ww, cc)
        rows.append({"instance": k, "dp_optimum": a, "greedy": g, "gap_pct": round(100 * (a - g) / a, 6)})
    trap = S.GREEDY_TRAP
    return orders, dict(dp=dp_val, greedy=gr_val, capacity=cap), pd.DataFrame(rows), \
        (integer.knapsack_dp(**trap)[0], integer.knapsack_greedy(**trap)[0])


def staffing():
    A = np.array(list(S.SHIFTS.values()), float).T
    d = np.array(S.STAFF_NEEDED, float)
    c = np.array(S.SHIFT_COST)
    relax = lp.simplex(c, -A, -d)
    down = np.floor(relax.x + 1e-9)
    up = np.ceil(relax.x - 1e-9)
    x, val, nodes, log = integer.branch_and_bound(c, -A, -d)
    table = pd.DataFrame({"shift": list(S.SHIFTS), "cost": c, "lp_relaxation": relax.x.round(6),
                          "rounded_down": down + 0.0, "rounded_up": up + 0.0, "branch_and_bound": x})
    summary = {"lp_cost": relax.objective, "round_down_feasible": bool(np.all(A @ down >= d)),
               "round_down_short_blocks": int((A @ down < d).sum()),
               "round_up_cost": float(c @ up), "integer_cost": val, "nodes": nodes}
    return table, summary, log


def assignment_example():
    lots, buyers, bids = S.lot_bids()
    a_h, v_h = assignment.hungarian_max(bids)
    a_g, v_g = assignment.greedy_max(bids)
    df = pd.DataFrame({"lot": lots, "hungarian_buyer": [buyers[j] for j in a_h],
                       "hungarian_bid": bids[np.arange(len(lots)), a_h],
                       "greedy_buyer": [buyers[j] for j in a_g],
                       "greedy_bid": bids[np.arange(len(lots)), a_g]})
    return df, v_h, v_g, bids


def pricing():
    m = nonlinear.ExpDemandProfit(S.PRICING["a"], S.PRICING["b"], S.PRICING["cost"])
    gd = nonlinear.gradient_ascent(m.grad, S.PRICING["p0"], S.PRICING["step"], n_iter=40)
    nt = nonlinear.newton(m.grad, m.hess, S.PRICING["p0"], n_iter=6)
    gs, widths = nonlinear.golden_section(m.profit, 2.0, 20.0, tol=1e-8)
    rows = []
    for k in range(len(gd)):
        rows.append({"iteration": k, "gradient_price": gd[k], "gradient_error": abs(gd[k] - m.optimum),
                     "newton_price": nt[k] if k < len(nt) else np.nan,
                     "newton_error": abs(nt[k] - m.optimum) if k < len(nt) else np.nan,
                     "golden_width": widths[k] if k < len(widths) else np.nan})
    return m, pd.DataFrame(rows), gs, widths


def channels():
    ch = S.CHANNELS
    x, lam = nonlinear.lagrange_allocation(ch["a"], ch["s"], ch["budget"])
    eq = np.full(len(x), ch["budget"] / len(x))
    df = pd.DataFrame({"channel": ch["names"], "spend": x.round(4),
                       "return": nonlinear.log_response(x, ch["a"], ch["s"]).round(4),
                       "marginal_return": nonlinear.marginal_return(x, ch["a"], ch["s"]).round(4),
                       "even_split_spend": eq,
                       "even_split_return": nonlinear.log_response(eq, ch["a"], ch["s"]).round(4)})
    return df, lam


def offers():
    P, p = S.OFFERS, S.OFFER_PROBS
    crit = decisions.criteria(P, p)
    info = decisions.evpi(P, p)
    tree = decisions.offer_tree(P, p, S.FORECAST, S.FORECAST_COST)
    ev_tree = decisions.rollback(tree)
    pol = decisions.policy(tree)
    ev_now = tree["options"][0][1]["ev"]
    ev_forecast_gross = tree["options"][1][1]["ev"] + S.FORECAST_COST
    ce = pd.DataFrame([{"risk_tolerance": R, **{opt: round(decisions.certainty_equivalent(P.loc[opt], p, R), 4)
                                               for opt in P.index}} for R in S.RISK_TOLERANCES])
    ce["preferred"] = ce[list(P.index)].idxmax(axis=1)
    hi = lambda ph: (lambda P_, p_: (P_, decisions.with_p_high(p_, ph)))
    def set_cell(opt, st, val):
        def fn(P_, p_):
            P_.loc[opt, st] = val
            return P_, p_
        return fn
    def set_row(opt, val):
        def fn(P_, p_):
            P_.loc[opt, :] = val
            return P_, p_
        return fn
    ow = decisions.one_way(P, p, {
        "P(high market) 0.10 / 0.30": (hi(0.10), hi(0.30)),
        "spot payoff, high market 85 / 135": (set_cell("spot market", "high", 85), set_cell("spot market", "high", 135)),
        "spot payoff, low market 5 / 25": (set_cell("spot market", "low", 5), set_cell("spot market", "low", 25)),
        "fixed contract 45 / 60": (set_row("fixed contract", 45), set_row("fixed contract", 60)),
        "price-indexed, medium 50 / 60": (set_cell("price-indexed", "medium", 50), set_cell("price-indexed", "medium", 60)),
    })
    tw = decisions.two_way(P, p, np.round(np.arange(0.0, 0.501, 0.025), 3),
                           np.arange(40.0, 66.0, 1.0), "fixed contract")
    return dict(criteria=crit, info=info, tree=tree, ev_tree=ev_tree, policy=pol, ev_now=ev_now,
                evsi=ev_forecast_gross - ev_now, ce=ce, one_way=ow, two_way=tw)


def newsvendor():
    nv = S.NEWSVENDOR
    q_star, cr = decisions.newsvendor_poisson(**nv)
    qs = np.arange(25, 56)
    exact = decisions.newsvendor_exact_curve(nv["lam"], nv["price"], nv["cost"], nv["salvage"], qs)
    sim = decisions.newsvendor_simulated(nv["lam"], nv["price"], nv["cost"], nv["salvage"], qs)
    df = pd.DataFrame({"stock": qs, "expected_profit_exact": exact.round(6), "expected_profit_simulated": sim.round(6)})
    return df, q_star, cr
