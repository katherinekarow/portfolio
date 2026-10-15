"""Logical layer: the same transfer stream under two state models.

AccountModel - one balance per firm (how a bank core or an Ethereum-style ledger thinks).
UTXOModel    - balances are unspent coins; a payment consumes whole coins and creates new ones
               (receiver output + change back to the sender), as in a Bitcoin-style ledger.
Both reject overdrafts; the UTXO model also rejects double spends of an already-consumed coin.
"""
from __future__ import annotations

from collections import defaultdict


class Rejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AccountModel:
    def __init__(self):
        self.balance: dict[str, int] = defaultdict(int)

    def apply(self, ev: dict) -> None:
        amt = ev["amount"]
        if amt <= 0:
            raise Rejected("non-positive amount")
        if ev["kind"] != "mint":
            if self.balance[ev["from"]] < amt:
                raise Rejected("overdraft")
            self.balance[ev["from"]] -= amt
        self.balance[ev["to"]] += amt

    def balances(self) -> dict[str, int]:
        return {k: v for k, v in sorted(self.balance.items()) if v}  # zero balances omitted, as in UTXO


class UTXOModel:
    def __init__(self):
        self.coins: dict[str, tuple[str, int]] = {}        # coin id -> (owner, amount), unspent only
        self.by_owner: dict[str, dict[str, None]] = defaultdict(dict)  # insertion-ordered set
        self.spent: set[str] = set()
        self._n = 0

    def _create(self, txid: str, outputs: list[tuple[str, int]]) -> None:
        for k, (owner, amt) in enumerate(outputs):
            cid = f"{txid}:{k}"
            self.coins[cid] = (owner, amt)
            self.by_owner[owner][cid] = None

    def apply_raw(self, inputs: list[str], outputs: list[tuple[str, int]], owner: str) -> str:
        """Low-level transaction with explicit inputs: the place a double spend would be caught."""
        if len(set(inputs)) != len(inputs):
            raise Rejected("double spend (same coin twice in one tx)")
        for cid in inputs:
            if cid in self.spent:
                raise Rejected("double spend (coin already spent)")
            if cid not in self.coins:
                raise Rejected("unknown input")
            if self.coins[cid][0] != owner:
                raise Rejected("input not owned by sender")
        if sum(self.coins[c][1] for c in inputs) != sum(a for _, a in outputs):
            raise Rejected("inputs != outputs")
        for cid in inputs:
            o, _ = self.coins.pop(cid)
            del self.by_owner[o][cid]
            self.spent.add(cid)
        txid = f"u{self._n}"
        self._n += 1
        self._create(txid, outputs)
        return txid

    def apply(self, ev: dict) -> None:
        amt = ev["amount"]
        if amt <= 0:
            raise Rejected("non-positive amount")
        if ev["kind"] == "mint":
            self._create(f"u{self._n}", [(ev["to"], amt)])
            self._n += 1
            return
        frm, picked, total = ev["from"], [], 0
        for cid in self.by_owner[frm]:  # oldest coins first
            picked.append(cid)
            total += self.coins[cid][1]
            if total >= amt:
                break
        if total < amt:
            raise Rejected("overdraft")
        outs = [(ev["to"], amt)] + ([(frm, total - amt)] if total > amt else [])
        self.apply_raw(picked, outs, frm)

    def balances(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for owner, amt in self.coins.values():
            out[owner] += amt
        return {k: v for k, v in sorted(out.items())}


def replay(events: list[dict], model) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Apply events in order; return (accepted, [(rejected event, reason)])."""
    ok, bad = [], []
    for ev in events:
        try:
            model.apply(ev)
            ok.append(ev)
        except Rejected as r:
            bad.append((ev, r.reason))
    return ok, bad
