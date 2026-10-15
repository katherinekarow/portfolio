"""On-chain / off-chain hybrid: invoice bodies live in SQLite, only a salted hash goes on-chain.

doc hash = SHA-256(salt || canonical(payload)). The per-document random salt matters for erasure:
an unsalted hash of a low-entropy document (a known template plus an amount) can be confirmed by
guessing; once the salt is deleted together with the payload, the on-chain hash is just 32 random-looking bytes.

"Right to be forgotten": erase() deletes payload and salt and leaves a tombstone. The chain is untouched
and still validates; the integrity check reports the document as 'erased' instead of 'ok'.
"""
from __future__ import annotations

import sqlite3

from .hashing import canonical, sha256_hex


class DocStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS docs (
            invoice TEXT PRIMARY KEY, doc_hash TEXT NOT NULL, salt TEXT, payload TEXT,
            erased_reason TEXT)""")

    def put(self, invoice: str, salt_hex: str, payload: dict) -> str:
        body = canonical(payload)
        h = sha256_hex(bytes.fromhex(salt_hex) + body)
        self.db.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,NULL)", (invoice, h, salt_hex, body.decode()))
        return h

    def load(self, payloads: dict, salts: dict) -> None:
        for inv, p in payloads.items():
            self.put(inv, salts[inv], p)
        self.db.commit()

    def erase(self, invoice: str, reason: str = "erasure request") -> None:
        self.db.execute("UPDATE docs SET payload=NULL, salt=NULL, erased_reason=? WHERE invoice=?", (reason, invoice))
        self.db.commit()

    def get(self, invoice: str):
        return self.db.execute("SELECT salt, payload, erased_reason FROM docs WHERE invoice=?", (invoice,)).fetchone()

    def check(self, invoice: str, onchain_hash: str) -> str:
        """'ok' | 'mismatch' | 'erased' | 'missing'."""
        row = self.get(invoice)
        if row is None:
            return "missing"
        salt, payload, erased = row
        if erased is not None:
            return "erased"
        h = sha256_hex(bytes.fromhex(salt) + payload.encode())
        return "ok" if h == onchain_hash else "mismatch"

    def corrupt(self, invoice: str) -> None:
        """Simulate someone editing an off-chain document without touching the chain."""
        salt, payload, _ = self.get(invoice)
        self.db.execute("UPDATE docs SET payload=? WHERE invoice=?", (payload.replace('"amount":', '"amount":1'), invoice))
        self.db.commit()

    def size_bytes(self) -> int:
        return sum(len(r[0] or "") + len(r[1] or "") for r in self.db.execute("SELECT salt, payload FROM docs"))


def verify_documents(chain, store: DocStore) -> dict[str, int]:
    """Check every on-chain document hash against the off-chain store (each invoice once)."""
    seen, counts = set(), {"ok": 0, "mismatch": 0, "erased": 0, "missing": 0}
    for _, _, tx in chain.transactions():
        inv = tx.get("invoice")
        if inv and inv not in seen:
            seen.add(inv)
            counts[store.check(inv, tx["doc"])] += 1
    return counts
