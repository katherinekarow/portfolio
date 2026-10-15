"""Physical layer: blocks, hash links and two consensus modes.

Block hash = SHA-256(canonical header || "|nonce=" || nonce). The header carries the previous
block's hash (the link) and the Merkle root of the block's transactions.

Consensus modes
  pow  - proof of work: find a nonce so the block hash has >= `difficulty` leading zero bits.
         Expected attempts = 2^difficulty (each hash is an independent coin flip).
  poa  - permissioned, BFT-style: a fixed signer set of 3f+1 validators; the proposer rotates
         round-robin and a block needs approvals from a 2f+1 quorum. Signatures are SIMULATED
         with HMAC-SHA256 over the block hash using demo secrets; a real system would use
         public-key signatures so auditors never need the secrets.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
from dataclasses import dataclass, field

from .hashing import canonical, merkle_root, sha256_hex
from .models import AccountModel, Rejected

GENESIS_PREV = "0" * 64


def tx_bytes(tx: dict) -> bytes:
    return canonical(tx)


def meets_difficulty(hex_hash: str, bits: int) -> bool:
    return bits <= 0 or (int(hex_hash, 16) >> (256 - bits)) == 0


@dataclass
class Validators:
    """Simulated permissioned signer set. Secrets are derived from a seed: demo-only, not keys of anything."""
    n: int = 4
    seed: int = 0
    ids: list[str] = field(init=False)
    _secrets: dict[str, bytes] = field(init=False, repr=False)

    def __post_init__(self):
        self.ids = [f"V{i}" for i in range(self.n)]
        self._secrets = {v: hashlib.sha256(f"demo-validator-{v}-{self.seed}".encode()).digest() for v in self.ids}

    @property
    def quorum(self) -> int:
        f = (self.n - 1) // 3
        return 2 * f + 1

    def proposer(self, index: int) -> str:
        return self.ids[index % self.n]

    def sign(self, vid: str, block_hash: str) -> str:
        return hmac.new(self._secrets[vid], block_hash.encode(), "sha256").hexdigest()

    def verify(self, vid: str, block_hash: str, sig: str) -> bool:
        return vid in self._secrets and hmac.compare_digest(self.sign(vid, block_hash), sig)


@dataclass
class Block:
    header: dict
    txs: list[dict]
    nonce: int = 0
    hash: str = ""
    approvals: dict[str, str] = field(default_factory=dict)

    def compute_hash(self, nonce: int | None = None) -> str:
        n = self.nonce if nonce is None else nonce
        return sha256_hex(canonical(self.header) + b"|nonce=" + str(n).encode())

    def size_bytes(self) -> int:
        return len(canonical({"h": self.header, "n": self.nonce, "hash": self.hash,
                              "a": self.approvals, "txs": self.txs}))


def mine(block: Block, bits: int, start: int = 0) -> int:
    """Search nonces from `start`; set nonce/hash. Returns the number of hashes tried."""
    prefix = canonical(block.header) + b"|nonce="
    shift = 256 - bits
    n = start
    while True:
        h = hashlib.sha256(prefix + str(n).encode()).hexdigest()
        if bits <= 0 or (int(h, 16) >> shift) == 0:
            block.nonce, block.hash = n, h
            return n - start + 1
        n += 1


class Chain:
    def __init__(self, mode: str = "poa", difficulty: int = 0, validators: Validators | None = None):
        if mode not in ("pow", "poa"):
            raise ValueError(mode)
        self.mode, self.difficulty = mode, difficulty
        self.validators = validators or Validators()
        self.blocks: list[Block] = []
        self.hashes_spent = 0

    @property
    def head(self) -> str:
        return self.blocks[-1].hash if self.blocks else GENESIS_PREV

    def seal(self, block: Block) -> None:
        """Run consensus on a block whose header is final."""
        if self.mode == "pow":
            self.hashes_spent += mine(block, self.difficulty)
        else:
            block.nonce, block.hash = 0, block.compute_hash(0)
            block.approvals = {v: self.validators.sign(v, block.hash) for v in self.validators.ids}

    def make_header(self, index: int, prev: str, txs: list[dict]) -> dict:
        return {"index": index, "prev": prev, "merkle": merkle_root([tx_bytes(t) for t in txs]),
                "t": txs[-1]["t"] if txs else 0, "n_tx": len(txs), "mode": self.mode,
                "difficulty": self.difficulty if self.mode == "pow" else 0,
                "proposer": self.validators.proposer(index) if self.mode == "poa" else None}

    def append(self, txs: list[dict]) -> Block:
        b = Block(self.make_header(len(self.blocks), self.head, txs), list(txs))
        self.seal(b)
        self.blocks.append(b)
        return b

    def copy(self) -> "Chain":
        return copy.deepcopy(self)

    def size_bytes(self) -> int:
        return sum(b.size_bytes() for b in self.blocks)

    def transactions(self):
        for b in self.blocks:
            for k, tx in enumerate(b.txs):
                yield b.header["index"], k, tx


def build(events: list[dict], block_size: int = 50, mode: str = "poa", difficulty: int = 0,
          validators: Validators | None = None) -> tuple[Chain, list[tuple[dict, str]]]:
    """Validate events against the account state (the node's mempool check) and pack accepted ones into blocks.
    Mints go in the genesis block. Returns (chain, rejected)."""
    chain = Chain(mode, difficulty, validators)
    state, rejected, pending = AccountModel(), [], []
    mints = [e for e in events if e["kind"] == "mint"]
    for e in mints:
        state.apply(e)
    chain.append(mints)
    for e in events:
        if e["kind"] == "mint":
            continue
        try:
            state.apply(e)
        except Rejected as r:
            rejected.append((e, r.reason))
            continue
        pending.append(e)
        if len(pending) == block_size:
            chain.append(pending)
            pending = []
    if pending:
        chain.append(pending)
    return chain, rejected


# ---------------------------------------------------------------- audit
CHECKS = ("merkle", "hash", "link", "consensus", "state")


def audit(chain: Chain, checkpoint: str | None = None) -> dict:
    """Verify every block. Per block:
      merkle    - header Merkle root matches the transactions in the body
      hash      - stored block hash equals the recomputed header hash
      link      - header.prev equals the previous block's stored hash
      consensus - PoW target met, or proposer is the round-robin one and a quorum of valid approvals exists
      state     - replaying the transactions never overdraws an account
    Optional checkpoint: a head hash published earlier by other consortium members.
    """
    rows, state, prev = [], AccountModel(), GENESIS_PREV
    v = chain.validators
    for b in chain.blocks:
        h = b.header
        r = {"block": h["index"],
             "merkle": h["merkle"] == merkle_root([tx_bytes(t) for t in b.txs]) and h["n_tx"] == len(b.txs),
             "hash": b.hash == b.compute_hash(),
             "link": h["prev"] == prev}
        if chain.mode == "pow":
            r["consensus"] = h["difficulty"] >= chain.difficulty and meets_difficulty(b.hash, chain.difficulty)
        else:
            good = sum(v.verify(vid, b.hash, s) for vid, s in b.approvals.items())
            r["consensus"] = h["proposer"] == v.proposer(h["index"]) and good >= v.quorum
        st = True
        for tx in b.txs:
            try:
                state.apply(tx)
            except Rejected:
                st = False
        r["state"] = st
        r["ok"] = all(r[c] for c in CHECKS)
        rows.append(r)
        prev = b.hash
    failures = [(r["block"], c) for r in rows for c in CHECKS if not r[c]]
    cp_ok = None if checkpoint is None else (chain.head == checkpoint)
    if cp_ok is False:
        failures.append((None, "checkpoint"))
    return {"valid": not failures, "blocks": rows, "failures": failures,
            "first_failure": failures[0] if failures else None, "checkpoint_ok": cp_ok,
            "n_blocks": len(rows), "n_tx": sum(len(b.txs) for b in chain.blocks)}


def audit_text(rep: dict, title: str) -> str:
    lines = [f"== {title} ==", f"blocks: {rep['n_blocks']}  transactions: {rep['n_tx']}",
             f"verdict: {'VALID' if rep['valid'] else 'INVALID'}"]
    if rep["checkpoint_ok"] is not None:
        lines.append(f"checkpoint (published head hash): {'match' if rep['checkpoint_ok'] else 'MISMATCH'}")
    if rep["failures"]:
        lines.append("failed checks (block, check):")
        for blk, c in rep["failures"][:12]:
            lines.append(f"  - block {blk if blk is not None else '-'}: {c}")
        if len(rep["failures"]) > 12:
            lines.append(f"  ... {len(rep['failures']) - 12} more")
    return "\n".join(lines)
