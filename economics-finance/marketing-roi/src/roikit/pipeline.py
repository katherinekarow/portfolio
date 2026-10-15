"""End-to-end run: history -> fit -> allocate -> financials -> Monte Carlo -> A/B plan -> KPI review."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import abtest, allocate, finance, kpi, mmm


def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    chans = list(cfg["channels"])
    labels = {c: v["label"] for c, v in cfg["channels"].items()}
    b, f = cfg["budget"], cfg["finance"]
    weeks, margin = b["weeks"], f["gross_margin"]
    media = b["total"] - sum(v["amount"] for v in b["fixed"].values())

    # 1. response curves: simulate history from truth, fit back
    hist, truth = mmm.make_history(cfg, rng)
    fitres = mmm.fit(hist, chans, rng)
    params = fitres["params"]
    ptab = mmm.param_table(fitres, truth, chans)

    # 2. allocation on the fitted curves
    lo = np.array([cfg["channels"][c]["min_share"] * media for c in chans])
    hi = np.array([cfg["channels"][c]["max_share"] * media for c in chans])
    naive = np.array([b["naive_split"][c] * media for c in chans])
    opt, opt_greedy = allocate.optimise(params, chans, media, lo, hi, weeks, margin, starts=[naive])
    true_opt, _ = allocate.optimise(truth, chans, media, lo, hi, weeks, margin, starts=[naive])
    atab = allocate.table(chans, labels, naive, opt, true_opt, params, truth, lo, hi, weeks, margin)
    k = allocate.kkt(opt, params, chans, lo, hi, weeks, margin)

    # 3. financials (truth = what would actually happen; model = what the plan expects)
    alloc_opt = dict(zip(chans, opt))
    alloc_naive = dict(zip(chans, naive))
    cf_opt = finance.cash_flows(alloc_opt, params, cfg)
    fin = {"optimal (model)": finance.summary(cf_opt, f["annual_discount"]),
           "naive (model)": finance.summary(finance.cash_flows(alloc_naive, params, cfg), f["annual_discount"]),
           "optimal (truth)": finance.summary(finance.cash_flows(alloc_opt, truth, cfg), f["annual_discount"]),
           "naive (truth)": finance.summary(finance.cash_flows(alloc_naive, truth, cfg), f["annual_discount"])}
    draws = mmm.draw_params(fitres, chans, f["mc_draws"], rng)
    mc = finance.monte_carlo(alloc_opt, draws, cfg, rng)
    mc_naive = finance.monte_carlo(alloc_naive, draws, cfg, np.random.default_rng(cfg["seed"] + 1))
    mcs = finance.mc_summary(mc)
    mcs.insert(0, "allocation", "optimal")
    mcn = finance.mc_summary(mc_naive)
    mcn.insert(0, "allocation", "naive")
    mc_tab = pd.concat([mcs, mcn], ignore_index=True)

    # 4. A/B testing
    a = cfg["ab"]
    ab_plan = abtest.plan(a["tests"], a["alpha"], a["power"])
    t0 = a["tests"][0]
    n = int(ab_plan.n_per_arm.iloc[0])
    p1, p2 = t0["baseline"], t0["baseline"] * (1 + a["analysis_true_rel_lift"])
    ab_res = abtest.analyse(rng.binomial(n, p1), n, rng.binomial(n, p2), n, a["alpha"])
    ab_res = {"test": t0["test"], "n_per_arm": n, "true_rel_lift": a["analysis_true_rel_lift"], **ab_res}
    pk = a["peeking"]
    peek = pd.DataFrame([(L, *abtest.peeking_fpr(L, pk["n_per_arm"], pk["baseline"], pk["sims"], rng, a["alpha"]))
                         for L in pk["looks"]], columns=["looks", "fpr_naive", "fpr_bonferroni"])

    # 5. KPI review after 90 days
    kp = kpi.review(alloc_opt, params, truth, cfg, rng)

    return {"history": hist, "truth": truth, "fit": fitres, "params": params, "param_table": ptab,
            "chans": chans, "labels": labels, "media": media, "lo": lo, "hi": hi, "naive": naive, "opt": opt,
            "opt_greedy": opt_greedy, "true_opt": true_opt, "allocation": atab, "kkt": k, "cash_flows": cf_opt,
            "financials": fin, "mc": mc, "mc_naive": mc_naive, "mc_summary": mc_tab, "ab_plan": ab_plan,
            "ab_result": ab_res, "peeking": peek, "kpi": kp, "cfg": cfg}
