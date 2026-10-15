"""Analytics on the ledger: a flat payments table, a firm-to-firm flow matrix, and wash-loop detection.

Wash / circular-trading rule (temporal cycle search in the payment graph):
  a chain of payments u -> v -> ... -> u of length 3..max_len, where
    * each hop happens after the previous one and within `window` seconds of the first,
    * each hop passes on between `min_keep` and 100% of the previous hop's amount.
  A ring of firms is FLAGGED when it completes at least `min_rounds` such loops.
Two-party round trips are excluded on purpose: a cancelled invoice paid back is exactly that shape.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

import pandas as pd


def payments_table(chain) -> pd.DataFrame:
    rows = [{"block": b, "pos": k, "seq": tx["seq"], "t": tx["t"], "kind": tx["kind"],
             "payer": tx["from"], "payee": tx["to"], "amount": tx["amount"] / 100, "invoice": tx["invoice"]}
            for b, k, tx in chain.transactions()]
    return pd.DataFrame(rows)


def flow_matrix(pay: pd.DataFrame) -> pd.DataFrame:
    p = pay[pay.kind != "mint"]
    return p.pivot_table(index="payer", columns="payee", values="amount", aggfunc="sum", fill_value=0.0)


def top_flows(pay: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    p = pay[pay.kind != "mint"]
    return (p.groupby(["payer", "payee"]).amount.agg(total="sum", count="size")
            .sort_values("total", ascending=False).head(n).reset_index())


def find_loops(pay: pd.DataFrame, window: int = 3600, min_keep: float = 0.99, max_len: int = 4) -> list[dict]:
    p = pay[pay.kind == "pay"].sort_values(["t", "seq"])
    out_edges: dict[str, list[tuple]] = defaultdict(list)
    for r in p.itertuples(index=False):
        out_edges[r.payer].append((r.t, r.seq, r.payee, r.amount))
    times = {k: [e[0] for e in v] for k, v in out_edges.items()}
    loops: list[dict] = []

    def dfs(origin, node, t0, t_last, amt, a0, nodes, seqs):
        lst = out_edges.get(node)
        if not lst:
            return
        for j in range(bisect_right(times[node], t_last), len(lst)):
            t, seq, nxt, a = lst[j]
            if t > t0 + window:
                break
            if not (min_keep * amt <= a <= amt):
                continue
            if nxt == origin:
                if len(nodes) >= 3:  # three or more distinct firms
                    loops.append({"ring": tuple(nodes), "seqs": tuple(seqs) + (seq,), "start": t0, "end": t,
                                  "amount_in": a0, "amount_back": a})
            elif len(nodes) < max_len and nxt not in nodes:
                dfs(origin, nxt, t0, t, a, a0, nodes + [nxt], seqs + [seq])

    for r in p.itertuples(index=False):
        dfs(r.payer, r.payee, r.t, r.t, r.amount, r.amount, [r.payer, r.payee], [r.seq])
    return loops


def flag_rings(loops: list[dict], min_rounds: int = 2) -> pd.DataFrame:
    """Group loops by the set of firms involved and keep rings that repeat."""
    if not loops:
        return pd.DataFrame(columns=["ring", "rounds", "first_seq", "last_seq", "amount_in", "amount_back"])
    g = defaultdict(list)
    for lp in loops:
        g[frozenset(lp["ring"])].append(lp)
    rows = []
    for _, lps in g.items():
        lps.sort(key=lambda d: d["seqs"][0])
        # count non-overlapping rounds (a payment can belong to only one round)
        used, rounds = set(), []
        for lp in lps:
            if used.isdisjoint(lp["seqs"]):
                used.update(lp["seqs"])
                rounds.append(lp)
        rows.append({"ring": " -> ".join(rounds[0]["ring"] + (rounds[0]["ring"][0],)), "rounds": len(rounds),
                     "first_seq": rounds[0]["seqs"][0], "last_seq": rounds[-1]["seqs"][-1],
                     "amount_in": rounds[0]["amount_in"], "amount_back": rounds[-1]["amount_back"],
                     "seqs": sorted(used)})
    df = pd.DataFrame(rows).sort_values("rounds", ascending=False)
    return df[df.rounds >= min_rounds].reset_index(drop=True)
