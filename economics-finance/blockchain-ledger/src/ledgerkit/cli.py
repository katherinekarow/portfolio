"""python -m ledgerkit run --out sample_output --figures figures

Build the settlement stream, run both state models, build the chain, attack it, audit it, try the
off-chain erasure, benchmark against SQLite, and look for wash loops. Prints a report and writes files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from . import analytics, benchmark, chain as ch, models, tamper, workload
from .offchain import DocStore, verify_documents

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GOOD, CRIT = "#0ca30c", "#d03b3b"


def run(out: Path, figures: Path | None = None, n_pay: int = 4000, seed: int = 7, quick: bool = False) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = workload.make_stream(n_pay=n_pay, seed=seed)
    ev = s["events"]
    lines: list[str] = []

    # --- logical layer: account vs UTXO on the same stream
    acc, utxo = models.AccountModel(), models.UTXOModel()
    _, rej_a = models.replay(ev, acc)
    _, rej_u = models.replay(ev, utxo)
    same = acc.balances() == utxo.balances()
    victim = next(iter(utxo.by_owner["Alder"]))  # an unspent coin: spend it, then try again
    owner_amt = utxo.coins[victim][1]
    utxo.apply_raw([victim], [("Birch", owner_amt)], "Alder")
    try:
        utxo.apply_raw([victim], [("Cedar", owner_amt)], "Alder")
        double_spend = "ACCEPTED (bug)"
    except models.Rejected as r:
        double_spend = f"rejected: {r.reason}"

    # --- physical layer: permissioned chain (main) + PoW chains for cost
    c, st_poa = benchmark.run_ledger(ev, mode="poa")
    rep_clean = ch.audit(c, checkpoint=c.head)
    pow_bits = [8] if quick else [8, 12]
    pow_runs = {b: benchmark.run_ledger(ev, mode="pow", difficulty=b) for b in pow_bits}
    cp = pow_runs[pow_bits[-1]][0]

    tblk = len(c.blocks) // 3
    orig_amt = c.blocks[tblk].txs[5]["amount"]
    tamper_rows = []
    audit_parts = [ch.audit_text(rep_clean, "clean permissioned chain")]
    for name, chain_ in (("permissioned", c), (f"PoW {pow_bits[-1]} bits", cp)):
        for lvl in ("edit", "rehash", "rewrite"):
            t, info = tamper.tamper(chain_, tblk, 5, orig_amt // 2, lvl)
            r = ch.audit(t, checkpoint=chain_.head)
            ff = r["first_failure"]
            tamper_rows.append({"chain": name, "attack": lvl, "detected": not r["valid"],
                                "first_failure": f"block {ff[0]}: {ff[1]}" if ff and ff[0] is not None else (ff[1] if ff else "-"),
                                "n_failed_checks": len(r["failures"]), "attacker_hashes": info["hashes"]})
            audit_parts.append(ch.audit_text(r, f"{name}: {lvl} tx 5 of block {tblk} ({orig_amt} -> {orig_amt // 2} cents)"))
    tamper_df = pd.DataFrame(tamper_rows)

    # --- off-chain hybrid
    store = DocStore()
    store.load(s["payloads"], s["salts"])
    docs_before = verify_documents(c, store)
    erase_ids = [tx["invoice"] for _, _, tx in c.transactions() if tx["kind"] == "pay" and tx["to"] == "Larch"][:3]
    for inv in erase_ids:
        store.erase(inv, "erasure request (synthetic)")
    corrupt_id = next(tx["invoice"] for _, _, tx in c.transactions() if tx["kind"] == "pay" and tx["invoice"] not in erase_ids)
    store.corrupt(corrupt_id)
    docs_after = verify_documents(c, store)
    chain_after_erase = ch.audit(c, checkpoint=c.head)["valid"]
    erased_hash = next(tx["doc"] for _, _, tx in c.transactions() if tx["invoice"] == erase_ids[0])
    audit_parts.append("\n".join([
        "== off-chain documents ==",
        f"before: {docs_before}", f"erased {erase_ids} and corrupted {corrupt_id} off-chain",
        f"after:  {docs_after}", f"chain still valid after erasure: {chain_after_erase}",
        f"on-chain hash for {erase_ids[0]} is still {erased_hash[:16]}..., its content is gone: {store.get(erase_ids[0])[1] is None}"]))

    # --- database comparison
    db, st_sql = benchmark.run_sqlite(ev)
    db_same = benchmark.sqlite_balances(db) == acc.balances()
    lat = benchmark.query_latency(c, db, workload.FIRMS, n=60 if quick else 200)
    bench = [{"system": "sqlite (CRUD)", "metric": "throughput", "value": st_sql["accepted"] / st_sql["seconds"], "unit": "tx/s"},
             {"system": "ledger permissioned", "metric": "throughput", "value": st_poa["accepted"] / st_poa["seconds"], "unit": "tx/s"}]
    for b, (_, stt) in pow_runs.items():
        bench.append({"system": f"ledger PoW {b} bits", "metric": "throughput", "value": stt["accepted"] / stt["seconds"], "unit": "tx/s"})
    bench += [{"system": "sqlite (CRUD)", "metric": "storage", "value": st_sql["bytes"] / st_sql["accepted"], "unit": "bytes/tx"},
              {"system": "ledger permissioned", "metric": "storage", "value": st_poa["bytes"] / st_poa["accepted"], "unit": "bytes/tx"},
              {"system": "ledger permissioned + off-chain docs", "metric": "storage",
               "value": (st_poa["bytes"] + store.size_bytes()) / st_poa["accepted"], "unit": "bytes/tx"},
              {"system": "sqlite (CRUD)", "metric": "history rows kept", "value": st_sql["rows_now"], "unit": "rows"},
              {"system": "ledger permissioned", "metric": "history rows kept", "value": st_poa["accepted"], "unit": "rows"},
              {"system": "sqlite (CRUD)", "metric": "deleted rows (unrecoverable)", "value": st_sql["delete"], "unit": "rows"}]
    for b, (_, stt) in pow_runs.items():
        bench.append({"system": f"ledger PoW {b} bits", "metric": "hashes per block", "value": stt["hashes"] / stt["blocks"], "unit": "hashes"})
    for r in lat:
        bench.append({"system": r["system"], "metric": f"{r['operation']} | {r['variant']}", "value": r["median_us"], "unit": "us (median)"})
    bench_df = pd.DataFrame(bench)
    bench_df["value"] = bench_df.value.round(2)
    bench_df.to_csv(out / "benchmark.csv", index=False)

    sweep = benchmark.pow_sweep(range(2, 11, 2) if quick else range(2, 17, 2), budget=2 ** 14 if quick else 2 ** 19)
    sweep.round(4).to_csv(out / "pow_sweep.csv", index=False)
    tx_avg = st_poa["bytes"] / st_poa["accepted"]
    proofs = benchmark.proof_sizes(tx_avg)
    proofs.to_csv(out / "merkle_proofs.csv", index=False)

    # --- analytics
    pay = analytics.payments_table(c)
    flows = analytics.flow_matrix(pay)
    flows.round(2).to_csv(out / "flow_matrix.csv")
    top = analytics.top_flows(pay)
    top.round(2).to_csv(out / "top_flows.csv", index=False)
    loops = analytics.find_loops(pay)
    rings = analytics.flag_rings(loops)
    rings.drop(columns="seqs").to_csv(out / "flagged_rings.csv", index=False)
    flagged_seqs = set(x for sq in rings.seqs for x in sq) if len(rings) else set()
    stray_loops = sum(1 for lp in loops if not set(lp["seqs"]) <= flagged_seqs)

    tamper_df.to_csv(out / "tamper_results.csv", index=False)
    (out / "audit_report.txt").write_text("\n\n".join(audit_parts) + "\n")

    summary = {
        "events": len(ev), "accepted": st_poa["accepted"], "rejected": len(rej_a), "blocks": st_poa["blocks"],
        "reject_reasons": sorted({r for _, r in rej_a}), "models_agree": same and len(rej_a) == len(rej_u),
        "sqlite_agrees": db_same, "unspent_coins": len(utxo.coins), "double_spend": double_spend,
        "clean_audit_valid": rep_clean["valid"], "tamper": tamper_rows,
        "docs_before": docs_before, "docs_after": docs_after, "chain_valid_after_erase": chain_after_erase,
        "bench": bench_df, "sweep": sweep, "proofs": proofs, "rings": rings, "stray_loops": stray_loops,
        "head": c.head, "top": top}
    if figures:
        charts(Path(figures), bench_df, sweep, proofs, flows, rings, pay)
    _print(summary, tamper_df)
    return summary


def _print(sm: dict, tamper_df: pd.DataFrame) -> None:
    b = sm["bench"]
    print(f"stream: {sm['events']} events -> {sm['accepted']} on-chain in {sm['blocks']} blocks, "
          f"{sm['rejected']} rejected ({', '.join(sm['reject_reasons'])})")
    print(f"account model == UTXO model == SQLite balances: {sm['models_agree'] and sm['sqlite_agrees']} "
          f"({sm['unspent_coins']} unspent coins); double spend {sm['double_spend']}")
    print(f"clean chain audit: {'VALID' if sm['clean_audit_valid'] else 'INVALID'}")
    print("tamper experiments:")
    print(tamper_df.to_string(index=False))
    print(f"off-chain docs before {sm['docs_before']}")
    print(f"off-chain docs after  {sm['docs_after']}; chain still valid: {sm['chain_valid_after_erase']}")
    print("benchmark:")
    print(b.to_string(index=False))
    print("proof of work, measured vs 2^bits:")
    print(sm["sweep"][["bits", "trials", "theory", "mean_hashes", "ratio"]].round(3).to_string(index=False))
    p = sm["proofs"]
    print(f"merkle proof at 4096 tx: {int(p.proof_hashes.iloc[-1])} hashes = {int(p.proof_bytes.iloc[-1])} bytes "
          f"vs block {int(p.block_bytes.iloc[-1]):,} bytes")
    r = sm["rings"]
    print(f"flagged wash rings: {len(r)}" + (f" -> {r.ring.iloc[0]} x{r.rounds.iloc[0]} rounds" if len(r) else "")
          + f"; loops outside flagged rings: {sm['stray_loops']}")


# ---------------------------------------------------------------- charts
def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def charts(folder: Path, bench, sweep, proofs, flows, rings, pay):
    folder.mkdir(parents=True, exist_ok=True)

    # 1. ledger vs database
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    col = lambda name: S[0] if name.startswith("sqlite") else S[1]
    thr = bench[bench.metric == "throughput"]
    sto = bench[bench.metric == "storage"]
    lat = bench[bench.unit == "us (median)"]
    lat = lat[~lat.metric.str.startswith("build")]
    for ax, d, labels, title, xl, log in (
            (axes[0], thr, thr.system, "Write throughput", "transactions per second (log scale)", True),
            (axes[1], sto, sto.system.str.replace(" + ", "\n+ "), "Storage per transaction", "bytes per transaction", False),
            (axes[2], lat, (lat.system + ": " + lat.metric.str.replace(" | ", "\n")), "Query latency", "median microseconds (log scale)", True)):
        y = np.arange(len(d))[::-1]
        ax.barh(y, d.value, color=[col(x) for x in d.system], height=0.62)
        ax.set_yticks(y, labels, fontsize=7.8 if ax is axes[2] else 8.5)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xl, color=MUTED, fontsize=8.5)
        for yi, v in zip(y, d.value):
            ax.text(v * (1.08 if log else 1.0) + (0 if log else d.value.max() * 0.015), yi,
                    f"{v:,.0f}" if v >= 10 else f"{v:,.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(right=d.value.max() * (8 if log else 1.22))
        if ax is axes[0]:
            ax.set_xlim(left=1e3)
        _style(ax, title)
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=S[0], label="SQLite (CRUD)"), Patch(color=S[1], label="ledger (append-only)")],
                   frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    fig.suptitle("Same settlement workload: database vs ledger", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(folder / "ledger_vs_database.png", dpi=150)
    plt.close(fig)

    # 2. PoW measured vs theory
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bits = np.arange(sweep.bits.min(), sweep.bits.max() + 0.01, 0.1)
    ax.plot(bits, 2 ** bits, color=MUTED, lw=1.4, ls="--", label="theory: 2^bits")
    ax.errorbar(sweep.bits, sweep.mean_hashes, yerr=sweep.ci95, fmt="o", color=S[0], ms=6, capsize=3, lw=1.2,
                label="measured mean ± 95% CI")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("difficulty (leading zero bits required)", color=MUTED, fontsize=9)
    ax.set_ylabel("hashes per block (log2 scale)", color=MUTED, fontsize=9)
    for r in sweep.itertuples():
        ax.annotate(f"{r.ratio:.2f}×", (r.bits, r.mean_hashes), xytext=(8, -12), textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax, "Proof of work: every extra bit doubles the cost")
    fig.tight_layout()
    fig.savefig(folder / "pow_cost.png", dpi=150)
    plt.close(fig)

    # 3. Merkle proof size
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(proofs.n_tx, proofs.block_bytes, "-o", color=S[1], ms=4, lw=1.6, label="whole block (download everything)")
    pp = proofs[proofs.proof_bytes > 0]
    ax.plot(pp.n_tx, pp.proof_bytes, "-o", color=S[0], ms=4, lw=1.6, label="Merkle inclusion proof")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("transactions in block (log2 scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("bytes needed to verify one tx (log scale)", color=MUTED, fontsize=9)
    last = proofs.iloc[-1]
    ax.annotate(f"{int(last.proof_hashes)} hashes = {int(last.proof_bytes)} B", (last.n_tx, last.proof_bytes),
                xytext=(-110, 14), textcoords="offset points", fontsize=8, color=S[0])
    ax.annotate(f"{last.block_bytes / 1e6:.2f} MB", (last.n_tx, last.block_bytes), xytext=(-55, 6),
                textcoords="offset points", fontsize=8, color=S[1])
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    _style(ax, "Proof size grows with log(n), the block with n")
    fig.tight_layout()
    fig.savefig(folder / "merkle_proof_size.png", dpi=150)
    plt.close(fig)

    # 4. payment network with the flagged ring
    ring_set = set(rings.ring.iloc[0].split(" -> ")) if len(rings) else set()
    others = [f for f in workload.FIRMS if f not in ring_set]
    ring_list = [f for f in workload.FIRMS if f in ring_set]
    firms, step = [], max(1, len(workload.FIRMS) // max(1, len(ring_list)))
    for k in range(len(workload.FIRMS)):  # spread ring members around the circle so their arrows don't overlap
        firms.append(ring_list.pop(0) if ring_list and k % step == 0 else others.pop(0))
    ang = np.pi / 2 - 2 * np.pi * np.arange(len(firms)) / len(firms)
    pos = {f: (np.cos(a), np.sin(a)) for f, a in zip(firms, ang)}
    p = pay[pay.kind == "pay"]
    ring_seqs = set(x for sq in rings.seqs for x in sq) if len(rings) else set()
    normal = p[~p.seq.isin(ring_seqs)].groupby(["payer", "payee"]).amount.sum().sort_values(ascending=False)
    wash = p[p.seq.isin(ring_seqs)].groupby(["payer", "payee"]).amount.agg(["sum", "size", "mean"])
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    top = normal.head(24)
    wmax = top.max()
    for (a, b), v in top.items():
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=0.12", arrowstyle="-|>", mutation_scale=9,
                                     lw=0.6 + 3 * v / wmax, color="#b9b8b2", alpha=0.85, shrinkA=16, shrinkB=16, zorder=1))
    for (a, b), v in wash.iterrows():
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=0.12", arrowstyle="-|>", mutation_scale=14,
                                     lw=3.2, color=CRIT, shrinkA=16, shrinkB=16, zorder=3))
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        ax.text(mx * 0.78, my * 0.78, f"{int(v['size'])} × {v['mean'] / 1e3:,.0f}k", color=CRIT, fontsize=8, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ring_firms = set(wash.index.get_level_values(0)) if len(wash) else set()
    for f, (x, y) in pos.items():
        flagged = f in ring_firms
        ax.scatter([x], [y], s=520, color=CRIT if flagged else S[0], zorder=5, edgecolor="white", lw=1.5)
        ax.text(x * 1.2, y * 1.2, f, ha="center", va="center", fontsize=9, color=INK,
                fontweight="bold" if flagged else "normal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    title = "Payment network: the flagged wash ring"
    sub = (f"{rings.ring.iloc[0]} · {rings.rounds.iloc[0]} rounds, each hop keeps 99-100% of the amount"
           if len(rings) else "no ring flagged")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.03, 0.965, title, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.925, sub + "\ngrey: 24 largest ordinary pair flows over the whole period (width ∝ total XTS)",
             fontsize=8.5, color=MUTED, va="top")
    fig.savefig(folder / "wash_ring.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ledgerkit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the whole lab")
    r.add_argument("--out", default="sample_output")
    r.add_argument("--figures", default=None)
    r.add_argument("--n", type=int, default=4000, help="number of invoice payments")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--quick", action="store_true", help="smaller PoW work, for tests")
    a = ap.parse_args(argv)
    run(Path(a.out), Path(a.figures) if a.figures else None, n_pay=a.n, seed=a.seed, quick=a.quick)
