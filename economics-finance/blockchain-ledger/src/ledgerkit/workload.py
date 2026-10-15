"""Synthetic settlement stream for an invented consortium of trading firms.

Every event is a transfer in integer cents:
  mint      - opening settlement balance credited to a firm (genesis)
  pay       - a buyer settles an invoice to a seller
  reversal  - a cancelled invoice is paid back (the database DELETEs the original row instead)
Planted on purpose: a handful of overdraft attempts, and one circular-trading (wash) pattern
in which three firms pass nearly the same amount around a loop several times.
"""
from __future__ import annotations

import numpy as np

from .hashing import canonical, sha256_hex

FIRMS = ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir",
         "Gum", "Hazel", "Ivy", "Juniper", "Kauri", "Larch"]
WASH_RING = ["Hazel", "Ivy", "Juniper"]
SECONDS_PER_EVENT = 60
T0 = 1_700_000_000  # fixed synthetic epoch so runs are reproducible


def invoice_payload(inv: str, seller: str, buyer: str, amount: int, rng) -> dict:
    """Off-chain document: the invoice body. On-chain we keep only a salted hash of it."""
    n = int(rng.integers(1, 4))
    cuts = np.sort(rng.integers(1, amount, n - 1)) if n > 1 and amount > n else np.array([], int)
    parts = np.diff(np.concatenate([[0], cuts, [amount]])).astype(int).tolist()
    return {"invoice": inv, "seller": seller, "buyer": buyer, "currency": "XTS",  # ISO test currency code
            "lines": [{"sku": f"SKU-{int(rng.integers(100, 999))}", "amount": p} for p in parts],
            "contact_ref": f"contact-{int(rng.integers(1000, 9999))}"}


def make_stream(n_pay: int = 4000, seed: int = 7, cancel_rate: float = 0.05,
                n_overdraft: int = 6, wash_rounds: int = 3, wash_amount: int = 48_000_000,
                opening: int = 250_000_000) -> dict:
    """Return {"events": [...], "payloads": {invoice: dict}, "salts": {invoice: hex}, "planted": {...}}."""
    rng = np.random.default_rng(seed)
    events, payloads, salts = [], {}, {}

    def add(kind, frm, to, amount, inv=None, doc=None):
        events.append({"seq": len(events), "kind": kind, "from": frm, "to": to, "amount": int(amount),
                       "invoice": inv, "doc": doc, "t": T0 + SECONDS_PER_EVENT * len(events)})

    def invoice(frm, to, amount):
        inv = f"INV-{len(payloads) + 1:05d}"
        body = invoice_payload(inv, to, frm, amount, rng)
        salt = rng.bytes(16).hex()
        payloads[inv], salts[inv] = body, salt
        return inv, sha256_hex(bytes.fromhex(salt) + canonical(body))

    for f in FIRMS:
        add("mint", None, f, opening)

    pay_slots = set(rng.choice(np.arange(200, n_pay - 200), n_overdraft, replace=False).tolist())
    wash_start = int(n_pay * 0.55)
    paid = []  # invoices eligible for cancellation
    i = 0
    while i < n_pay:
        if i == wash_start:  # planted wash loop: Hazel -> Ivy -> Juniper -> Hazel, small haircut each hop
            amt = wash_amount
            for _ in range(wash_rounds):
                for a, b in zip(WASH_RING, WASH_RING[1:] + WASH_RING[:1]):
                    inv, doc = invoice(a, b, amt)
                    add("pay", a, b, amt, inv, doc)
                    amt = int(round(amt * 0.998))
        frm, to = rng.choice(FIRMS, 2, replace=False).tolist()
        if i in pay_slots:  # planted overdraft attempt: far more than any firm holds
            amt = opening * 40
        else:
            amt = max(100, int(np.exp(rng.normal(np.log(1_200_000), 1.0))))  # median 12,000.00 XTS
        inv, doc = invoice(frm, to, amt)
        add("pay", frm, to, amt, inv, doc)
        if i not in pay_slots:
            paid.append(len(events) - 1)
        if rng.random() < cancel_rate and len(paid) > 20:  # cancel an earlier invoice
            j = paid.pop(int(rng.integers(0, len(paid) - 10)))
            o = events[j]
            add("reversal", o["to"], o["from"], o["amount"], o["invoice"], o["doc"])
        i += 1
    return {"events": events, "payloads": payloads, "salts": salts,
            "planted": {"overdraft_amount": opening * 40, "n_overdraft": n_overdraft,
                        "wash_ring": WASH_RING, "wash_rounds": wash_rounds}}
