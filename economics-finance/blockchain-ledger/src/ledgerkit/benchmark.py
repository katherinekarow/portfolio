"""Ledger vs traditional database: the same settlement workload, measured.

SQLite (CRUD)   - accounts table with UPDATE balance, invoices table with INSERT; a cancelled invoice
                  is DELETEd and balances UPDATEd back. Only current state survives.
Ledger          - append-only: every event (including the payback of a cancelled invoice) becomes a
                  transaction in a hash-linked block; state is derived by replay.
Timings depend on the machine; ratios are the point. SQLite runs in memory here, which flatters it
less than you'd think: the ledger also runs in memory.
"""
from __future__ import annotations

import sqlite3
import statistics
import time

import numpy as np
import pandas as pd

from . import chain as ch
from .hashing import canonical, max_proof_len, merkle_proof, merkle_root, verify_proof
from .models import AccountModel


# ---------------------------------------------------------------- SQLite CRUD
def run_sqlite(events: list[dict], commit_every: int = 50) -> tuple[sqlite3.Connection, dict]:
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE accounts (firm TEXT PRIMARY KEY, balance INTEGER NOT NULL CHECK (balance >= 0));
        CREATE TABLE invoices (invoice TEXT PRIMARY KEY, payer TEXT, payee TEXT, amount INTEGER, t INTEGER);
    """)
    stats = {"insert": 0, "update": 0, "delete": 0, "rejected": 0}
    t0 = time.perf_counter()
    for k, e in enumerate(events):
        if e["kind"] == "mint":
            db.execute("INSERT INTO accounts VALUES (?,?)", (e["to"], e["amount"]))
            stats["insert"] += 1
        else:
            (bal,) = db.execute("SELECT balance FROM accounts WHERE firm=?", (e["from"],)).fetchone()
            if bal < e["amount"]:
                stats["rejected"] += 1
                continue
            db.execute("UPDATE accounts SET balance=balance-? WHERE firm=?", (e["amount"], e["from"]))
            db.execute("UPDATE accounts SET balance=balance+? WHERE firm=?", (e["amount"], e["to"]))
            stats["update"] += 2
            if e["kind"] == "pay":
                db.execute("INSERT INTO invoices VALUES (?,?,?,?,?)", (e["invoice"], e["from"], e["to"], e["amount"], e["t"]))
                stats["insert"] += 1
            else:  # reversal: the database simply forgets the invoice
                db.execute("DELETE FROM invoices WHERE invoice=?", (e["invoice"],))
                stats["delete"] += 1
        if k % commit_every == 0:
            db.commit()
    db.commit()
    stats["seconds"] = time.perf_counter() - t0
    stats["accepted"] = len(events) - stats["rejected"]
    pages = db.execute("PRAGMA page_count").fetchone()[0]
    psize = db.execute("PRAGMA page_size").fetchone()[0]
    stats["bytes"] = pages * psize
    stats["rows_now"] = db.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    return db, stats


def sqlite_balances(db) -> dict[str, int]:
    return {f: b for f, b in db.execute("SELECT firm, balance FROM accounts ORDER BY firm") if b}


def run_ledger(events, block_size=50, mode="poa", difficulty=0) -> tuple[ch.Chain, dict]:
    t0 = time.perf_counter()
    c, rej = ch.build(events, block_size, mode, difficulty)
    secs = time.perf_counter() - t0
    n = sum(len(b.txs) for b in c.blocks)
    return c, {"seconds": secs, "accepted": n, "rejected": len(rej), "bytes": c.size_bytes(),
               "hashes": c.hashes_spent, "blocks": len(c.blocks)}


# ---------------------------------------------------------------- queries
def _median_us(fn, keys) -> float:
    ts = []
    for k in keys:
        t = time.perf_counter()
        fn(k)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6


def query_latency(c: ch.Chain, db, firms: list[str], n: int = 200, seed: int = 1) -> list[dict]:
    keys = np.random.default_rng(seed).choice(firms, n).tolist()

    def ledger_scan(f):  # no index: replay the whole chain for one firm
        s = 0
        for b in c.blocks:
            for tx in b.txs:
                if tx["to"] == f:
                    s += tx["amount"]
                if tx["from"] == f:
                    s -= tx["amount"]
        return s

    t = time.perf_counter()
    idx = AccountModel()
    hist: dict[tuple, list] = {}
    for _, _, tx in c.transactions():
        idx.apply(tx)
        if tx["from"]:
            hist.setdefault((tx["from"], tx["to"]), []).append(tx)
    index_build_ms = (time.perf_counter() - t) * 1e3
    bal = idx.balance
    rng = np.random.default_rng(seed + 1)
    pairs = [tuple(rng.choice(firms, 2, replace=False).tolist()) for _ in range(n)]

    def ledger_pair_scan(pq):
        return [tx for b in c.blocks for tx in b.txs if tx["from"] == pq[0] and tx["to"] == pq[1]]

    def sql_bal(f):
        return db.execute("SELECT balance FROM accounts WHERE firm=?", (f,)).fetchone()

    def sql_pair(pq):
        return db.execute("SELECT * FROM invoices WHERE payer=? AND payee=?", pq).fetchall()

    rows = [("ledger", "balance lookup", "full chain scan (no index)", _median_us(ledger_scan, keys)),
            ("ledger", "balance lookup", "off-chain index (materialised)", _median_us(lambda f: bal[f], keys)),
            ("sqlite", "balance lookup", "primary key", _median_us(sql_bal, keys)),
            ("ledger", "payments A->B", "full chain scan (no index)", _median_us(ledger_pair_scan, pairs)),
            ("ledger", "payments A->B", "off-chain index (materialised)", _median_us(lambda pq: hist.get(pq, []), pairs)),
            ("sqlite", "payments A->B", "no index (table scan)", _median_us(sql_pair, pairs))]
    db.execute("CREATE INDEX ix_pair ON invoices(payer, payee)")
    rows.append(("sqlite", "payments A->B", "with index", _median_us(sql_pair, pairs)))
    out = [{"system": s, "operation": op, "variant": v, "median_us": round(us, 2)} for s, op, v, us in rows]
    out.append({"system": "ledger", "operation": "build off-chain index", "variant": "one replay of the chain",
                "median_us": round(index_build_ms * 1e3, 1)})
    return out


# ---------------------------------------------------------------- proof of work
def pow_sweep(bits_list=range(2, 17, 2), budget: int = 2 ** 19, max_trials: int = 400, min_trials: int = 32) -> pd.DataFrame:
    """Mine independent dummy headers; average attempts per success vs the theoretical 2^bits."""
    rows = []
    for bits in bits_list:
        trials = int(max(min_trials, min(max_trials, budget // 2 ** bits)))
        counts = []
        for k in range(trials):
            b = ch.Block({"sweep": bits, "trial": k, "prev": "0" * 64}, [])
            counts.append(ch.mine(b, bits))
            assert ch.meets_difficulty(b.hash, bits)
        arr = np.array(counts, float)
        rows.append({"bits": bits, "trials": trials, "theory": 2.0 ** bits, "mean_hashes": arr.mean(),
                     "ci95": 1.96 * arr.std(ddof=1) / np.sqrt(trials), "ratio": arr.mean() / 2.0 ** bits})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Merkle proofs
def proof_sizes(tx_bytes_avg: float, sizes=(1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 100, 128, 256, 512, 1000, 1024, 2048, 4096)) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for n in sizes:
        leaves = [canonical({"i": i}) for i in range(n)]
        root = merkle_root(leaves)
        k = int(rng.integers(0, n))
        pr = merkle_proof(leaves, k)
        assert verify_proof(leaves[k], pr, root)
        rows.append({"n_tx": n, "proof_hashes": len(pr), "bound_ceil_log2": max_proof_len(n),
                     "proof_bytes": len(pr) * 33, "block_bytes": round(n * tx_bytes_avg)})
    return pd.DataFrame(rows)
