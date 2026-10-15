"""Tamper experiments: change one historical transaction and see which check catches it.

Three escalating attackers:
  edit     - edit the transaction body only
  rehash   - edit, then recompute that block's Merkle root and hash (and re-seal it)
  rewrite  - edit, then rebuild every later block too (re-mine under PoW; under the permissioned
             mode the attacker is one insider validator and can only produce its own signature)
"""
from __future__ import annotations

from .chain import Chain


def tamper(chain: Chain, block: int, tx: int, new_amount: int, level: str = "edit") -> tuple[Chain, dict]:
    c = chain.copy()
    b = c.blocks[block]
    b.txs[tx] = dict(b.txs[tx], amount=int(new_amount))
    info = {"level": level, "block": block, "tx": tx, "hashes": 0}
    if level == "edit":
        return c, info
    if level not in ("rehash", "rewrite"):
        raise ValueError(level)
    last = block if level == "rehash" else len(c.blocks) - 1
    before = c.hashes_spent
    insider = c.validators.proposer(block)
    for i in range(block, last + 1):
        bi = c.blocks[i]
        prev = c.blocks[i - 1].hash if i else bi.header["prev"]
        bi.header = c.make_header(i, prev, bi.txs) | {"t": bi.header["t"]}
        if c.mode == "pow":
            c.seal(bi)
        else:  # insider can recompute the hash but only sign with its own demo secret
            bi.nonce, bi.hash = 0, bi.compute_hash(0)
            bi.approvals = dict(bi.approvals) | {insider: c.validators.sign(insider, bi.hash)}
    info["hashes"] = c.hashes_spent - before
    info["blocks_rewritten"] = last - block + 1
    return c, info
