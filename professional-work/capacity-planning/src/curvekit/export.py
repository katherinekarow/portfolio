"""Export the planning model to JSON for the interactive pace dashboard (docs/dashboards/fill-pace/).

    python -m curvekit.export --config config.yaml --out ../../docs/dashboards/fill-pace/data.json

Everything the browser shows is computed here by the same functions the pipeline uses,
so the dashboard can never drift from the model. Series are sampled weekly (every 7 days
to session) plus today's point, which keeps the file small.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import yaml

from . import curves as cv
from . import gross_net, strategies, synth


def _weekly(days_out_max: int, extra: int | None = None) -> list[int]:
    pts = set(range(0, days_out_max + 1, 7)) | {days_out_max}
    if extra is not None:
        pts.add(extra)
    return sorted(pts, reverse=True)


def _path(row, crv: pd.DataFrame, goal: float) -> pd.Series:
    """Path from today to `goal`, shaped like the segment's usual arrival pattern."""
    g = cv.glide_path(row, crv, goal)
    return g.set_index("days_out").glide_path


def build(config_path: str = "config.yaml") -> dict:
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    d, tg, ini = cfg["data"], cfg["targets"], cfg["outreach"]
    deps = synth.make_sessions(d["historical_sessions"], d["future_sessions"], d["seed"])
    bk = synth.make_appointments(deps, d["seed"] + 4)
    prof = cv.historical_profiles(bk, deps)
    crv = cv.segment_curves(prof, tg["percentile"])
    pos = cv.current_position(bk, deps)
    plan = cv.project_and_decompose(pos, crv, tg["volatility_z"])
    gn = gross_net.gross_to_net(pos, gross_net.cancellation_profile(crv))
    rules = dict(min_gap_units=ini["min_gap_units"], min_gap_pct=ini["min_gap_pct"], min_history=ini["min_history"])
    elig = strategies.eligible(plan, **rules)
    allocs = {s: strategies.build_outreach_plan(plan, crv, s, **rules) for s in ("baseline", "milestones")}
    recon = {s: bool(strategies.reconcile(plan, a).ok.all()) if len(a) else True for s, a in allocs.items()}

    # historical spread per segment: 10th-90th percentile of fill rate
    band = prof.groupby(["segment", "days_out"]).fill.quantile([0.1, 0.9]).unstack()
    band.columns = ["p10", "p90"]

    out_deps = []
    for _, r in plan.merge(gn[["session_id", "cancel_rate", "expected_net_from_gross"]], on="session_id").iterrows():
        today = int(r.days_out_today)
        cap = int(r.capacity)
        b = bk[bk.session_id == r.session_id].set_index("days_out")
        c = crv[crv.segment == r.segment].set_index("days_out")
        pts_hist = [x for x in _weekly(365, today) if x >= today]
        pts_fut = [x for x in _weekly(today, today)]
        proj = {k: _path(r, crv, r[f"projected_{k}"]) for k in ("low", "final", "high")}
        glide = _path(r, crv, r.target_final)
        weekly_units = {}
        for s, a in allocs.items():
            a = a[a.session_id == r.session_id]
            if len(a):
                wk = (today - a.days_out) // 7                     # weeks from today
                weekly_units[s] = [round(float(v), 1) for v in a.groupby(wk).outreach_units.sum().to_numpy()]
            else:
                weekly_units[s] = []
        sb = band.loc[r.segment]
        out_deps.append({
            "id": r.session_id, "segment": r.segment, "capacity": cap, "days_out_today": today,
            "net": int(r.net_appointments), "gross": int(r.gross_appointments),
            "projected": {"low": int(r.projected_low), "mid": int(r.projected_final), "high": int(r.projected_high)},
            "target_today": int(r.target_today), "target_final": int(r.target_final),
            "pace_gap_today": int(r.pace_gap_today), "gap_to_target": int(r.gap_to_target),
            "gap_to_full": int(r.gap_to_full), "status": r.status, "support": int(r.n),
            "cancel_rate": round(float(r.cancel_rate), 3), "expected_net_from_gross": int(r.expected_net_from_gross),
            "eligible": bool(elig[plan.session_id == r.session_id].iloc[0]),
            "series": {
                "days_out_hist": pts_hist,
                "net": [int(b.net_appointments[x]) for x in pts_hist],
                "gross": [int(b.gross_appointments[x]) for x in pts_hist],
                "days_out_all": _weekly(365, today),
                "target": [round(float(c.target_fill[x]) * cap) for x in _weekly(365, today)],
                "p10": [round(float(sb.p10[x]) * cap) for x in _weekly(365, today)],
                "p90": [round(float(sb.p90[x]) * cap) for x in _weekly(365, today)],
                "days_out_fut": pts_fut,
                "proj_low": [int(proj["low"][x]) for x in pts_fut],
                "proj_mid": [int(proj["final"][x]) for x in pts_fut],
                "proj_high": [int(proj["high"][x]) for x in pts_fut],
                "glide": [int(glide[x]) for x in pts_fut],
            },
            "outreach": weekly_units,
        })

    seg_summary = (plan.groupby("segment")
                   .agg(sessions=("session_id", "size"), capacity=("capacity", "sum"), net=("net_appointments", "sum"),
                        projected=("projected_final", "sum"), target=("target_final", "sum"),
                        gap=("gap_to_target", "sum")).reset_index())
    return {
        "settings": {"target_percentile": tg["percentile"], "volatility_z": tg["volatility_z"], **rules},
        "reconciled": recon,
        "segments": seg_summary.to_dict(orient="records"),
        "sessions": out_deps,
    }


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs/pace_data.json")
    a = ap.parse_args()
    data = build(a.config)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(data, default=_default, separators=(",", ":")))
    print(f"wrote {a.out}: {len(data['sessions'])} sessions")
