"""python -m optkit run --out sample_output --figures figures"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analysis, plots


def run(out: Path, figures: Path | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r, var, con, path, sweep = analysis.farm_lp()
    var.to_csv(out / "lp_variables.csv", index=False)
    con.to_csv(out / "lp_constraints.csv", index=False)
    path.to_csv(out / "lp_simplex_path.csv", index=False)
    sweep.to_csv(out / "lp_sensitivity_grain_profit.csv", index=False)

    tr, tr_cost, nw_cost = analysis.transport_example()
    tr.to_csv(out / "transport_plan.csv", index=False)

    orders, ks, gaps, trap = analysis.knapsack_examples()
    orders.to_csv(out / "knapsack_orders.csv", index=False)
    gaps.to_csv(out / "knapsack_gap_instances.csv", index=False)

    staff, st, _ = analysis.staffing()
    staff.to_csv(out / "staffing_branch_and_bound.csv", index=False)

    asg, v_h, v_g, _ = analysis.assignment_example()
    asg.to_csv(out / "assignment.csv", index=False)

    model, cv, gs, widths = analysis.pricing()
    cv.to_csv(out / "pricing_convergence.csv", index=False)
    ch, lam = analysis.channels()
    ch.to_csv(out / "budget_allocation.csv", index=False)

    of = analysis.offers()
    of["criteria"].round(4).to_csv(out / "offer_criteria.csv", index_label="option")
    of["ce"].to_csv(out / "offer_certainty_equivalents.csv", index=False)
    of["one_way"].to_csv(out / "sensitivity_one_way.csv", index=False)
    of["two_way"].to_csv(out / "sensitivity_two_way.csv", index=False)

    nv, q_star, cr = analysis.newsvendor()
    nv.to_csv(out / "newsvendor.csv", index=False)

    g = gaps.gap_pct
    summary = {
        "lp_plan": [float(v) for v in r.x], "lp_profit": r.objective,
        "shadow_prices": dict(zip(con.constraint, con.shadow_price.astype(float))),
        "simplex_pivots": len(r.path) - 1,
        "transport_cost": tr_cost, "northwest_corner_cost": nw_cost,
        "storage_dp": ks["dp"], "storage_greedy": ks["greedy"],
        "knapsack_greedy_optimal_share": float((g == 0).mean()),
        "knapsack_gap_median_pct": float(g.median()), "knapsack_gap_p90_pct": float(g.quantile(0.9)),
        "knapsack_gap_max_pct": float(g.max()), "greedy_trap": list(trap),
        "staffing": st,
        "assignment_hungarian": v_h, "assignment_greedy": v_g,
        "price_optimum": model.optimum, "golden_section": gs, "golden_steps": len(widths) - 1,
        "newton_steps_to_1e-8": int((cv.newton_error > 1e-8).sum()),
        "gradient_steps_to_1e-3": int((cv.gradient_error > 1e-3).sum()),
        "budget_lambda": lam, "budget_return": float(ch["return"].sum()),
        "even_split_return": float(ch.even_split_return.sum()),
        "offer_ev": of["criteria"].expected_value.round(4).to_dict(),
        "evpi": of["info"]["evpi"], "evsi": of["evsi"], "tree_ev": of["ev_tree"], "tree_policy": of["policy"],
        "newsvendor_q": q_star, "critical_ratio": cr,
        "newsvendor_sim_best": int(nv.stock[nv.expected_profit_simulated.idxmax()]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if figures:
        figures = Path(figures)
        figures.mkdir(parents=True, exist_ok=True)
        plots.feasible_region(figures, r)
        plots.shadow_prices(figures, con)
        plots.knapsack_gap(figures, gaps)
        plots.convergence(figures, cv)
        plots.newsvendor_curve(figures, nv, q_star, cr)
        plots.two_way(figures, of["two_way"])
        plots.decision_tree(figures, of["tree"])
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="solve every worked example and write tables (and figures)")
    r.add_argument("--out", default="outputs")
    r.add_argument("--figures", default=None)
    a = ap.parse_args(argv)
    s = run(Path(a.out), Path(a.figures) if a.figures else None)
    st = s["staffing"]
    print(f"planting LP: grain {s['lp_plan'][0]:.0f} ac, vegetables {s['lp_plan'][1]:.0f} ac, "
          f"profit ${s['lp_profit']:,.0f} after {s['simplex_pivots']} pivots")
    print("  shadow prices: " + ", ".join(f"{k} ${v:,.0f}" for k, v in s["shadow_prices"].items()))
    print(f"transport: ${s['transport_cost']:,.0f} (northwest-corner start ${s['northwest_corner_cost']:,.0f})")
    print(f"cold storage: DP ${s['storage_dp']:,.0f} vs greedy ${s['storage_greedy']:,.0f}; "
          f"1,000 random knapsacks: greedy optimal {s['knapsack_greedy_optimal_share']:.0%}, "
          f"median gap {s['knapsack_gap_median_pct']:.2f}%, max {s['knapsack_gap_max_pct']:.2f}%")
    print(f"staffing: LP ${st['lp_cost']:,.0f}, round-up ${st['round_up_cost']:,.0f}, "
          f"round-down feasible={st['round_down_feasible']}, branch-and-bound ${st['integer_cost']:,.0f} "
          f"({st['nodes']} nodes)")
    print(f"lot assignment: Hungarian ${s['assignment_hungarian']:,.0f} vs greedy ${s['assignment_greedy']:,.0f}")
    print(f"pricing: p* = {s['price_optimum']:.4f}; Newton {s['newton_steps_to_1e-8']} steps to 1e-8, "
          f"gradient {s['gradient_steps_to_1e-3']} steps to 1e-3, golden-section {s['golden_section']:.6f}")
    print(f"budget: lambda = {s['budget_lambda']:.4f}, return {s['budget_return']:.1f} vs even split "
          f"{s['even_split_return']:.1f}")
    print(f"offers: EV {s['offer_ev']}; EVPI {s['evpi']:.2f}, forecast worth {s['evsi']:.3f} "
          f"(tree EV {s['tree_ev']:.3f})")
    print(f"newsvendor: critical ratio {s['critical_ratio']:.4f} -> stock {s['newsvendor_q']} "
          f"(simulated best {s['newsvendor_sim_best']})")


if __name__ == "__main__":
    main()
