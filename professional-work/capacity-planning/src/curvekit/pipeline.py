"""Config-driven orchestrator.

    python -m curvekit.pipeline --config config.yaml

Runs each step in order, keeps going if a non-critical step fails, and prints a run summary.
Outputs are snapshotted to history *before* being overwritten.
"""
from __future__ import annotations

import argparse
import pathlib
import sqlite3
import time
import traceback

import pandas as pd
import yaml

from . import curves, gross_net, quality, strategies, synth


def run(config_path: str = "config.yaml") -> dict:
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    out = pathlib.Path(cfg["output"]["folder"]); out.mkdir(exist_ok=True)
    ctx: dict = {}
    summary = []

    def step(name, fn, critical=True):
        t = time.time()
        try:
            fn()
            summary.append((name, "ok", round(time.time() - t, 2), ""))
        except Exception as e:  # noqa: BLE001 - we want a summary, not a crash
            summary.append((name, "FAILED", round(time.time() - t, 2), str(e)))
            traceback.print_exc()
            if critical:
                raise

    d, tg, ini, q = cfg["data"], cfg["targets"], cfg["outreach"], cfg["quality"]
    step("generate synthetic data", lambda: ctx.update(
        deps=synth.make_sessions(d["historical_sessions"], d["future_sessions"], d["seed"])) or
        ctx.update(bk=synth.make_appointments(ctx["deps"], d["seed"] + 4)))
    step("historical profiles & segment curves", lambda: ctx.update(
        curves=curves.segment_curves(curves.historical_profiles(ctx["bk"], ctx["deps"]), tg["percentile"])))
    step("current position, projection & gap decomposition", lambda: ctx.update(
        plan=curves.project_and_decompose(curves.current_position(ctx["bk"], ctx["deps"]), ctx["curves"], tg["volatility_z"])))
    step("outreach allocation", lambda: ctx.update(alloc=strategies.build_outreach_plan(
        ctx["plan"], ctx["curves"], ini["strategy"], min_gap_units=ini["min_gap_units"],
        min_gap_pct=ini["min_gap_pct"], min_history=ini["min_history"])))
    step("reconciliation (control totals)", lambda: ctx.update(recon=strategies.reconcile(ctx["plan"], ctx["alloc"])))
    step("gross vs net", lambda: ctx.update(gn=gross_net.gross_to_net(
        curves.current_position(ctx["bk"], ctx["deps"]), gross_net.cancellation_profile(ctx["curves"]))), critical=False)
    step("data-quality checks", lambda: ctx.update(dq=quality.run_checks(ctx["bk"], ctx["deps"], ctx["plan"])), critical=False)

    def publish():
        conn = sqlite3.connect(cfg["output"]["history_db"])
        prev_path = out / "session_plan.csv"
        prev = pd.read_csv(prev_path) if prev_path.exists() else None
        ok, breaches = quality.validation_gate(ctx["plan"], prev, "session_id", "projected_final", q["max_change_pct"])
        if not ok:
            breaches.to_csv(out / "blocked_changes.csv", index=False)
            raise RuntimeError(f"validation gate blocked publish: {len(breaches)} sessions moved > {q['max_change_pct']:.0%}")
        if prev is not None:
            ctx["snapshot"] = quality.snapshot_first(conn, "SESSION_PLAN", prev)
        ctx["plan"].to_csv(prev_path, index=False)
        ctx["alloc"].to_csv(out / "outreach_plan.csv", index=False)
        ctx["recon"].to_csv(out / "reconciliation.csv", index=False)
        conn.close()
    step("snapshot-first publish (with validation gate)", publish, critical=False)

    print("\nRun summary")
    print(pd.DataFrame(summary, columns=["step", "status", "seconds", "note"]).to_string(index=False))
    return ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    run(ap.parse_args().config)
