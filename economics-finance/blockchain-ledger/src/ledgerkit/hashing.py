"""SHA-256 helpers and a Merkle tree with inclusion proofs.

The tree follows the RFC 6962 (Certificate Transparency) construction:
  leaf  = H(0x00 || data)
  node  = H(0x01 || left || right)
  an odd node at the end of a level is promoted unchanged (not duplicated).
Domain separation stops a leaf being passed off as an inner node, and promotion avoids the
duplicate-last-leaf ambiguity of the naive "copy the odd one" tree.
"""
from __future__ import annotations

import hashlib
import json
import math


def canonical(obj) -> bytes:
    """Deterministic JSON bytes (sorted keys, no whitespace) so the same record always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_hash(data: bytes) -> bytes:
    return sha256(b"\x00" + data)


def node_hash(left: bytes, right: bytes) -> bytes:
    return sha256(b"\x01" + left + right)


def _levels(leaves: list[bytes]) -> list[list[bytes]]:
    level = [leaf_hash(x) for x in leaves]
    levels = [level]
    while len(level) > 1:
        nxt = [node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])  # promote the odd node
        levels.append(nxt)
        level = nxt
    return levels


def merkle_root(leaves: list[bytes]) -> str:
    """Hex Merkle root of raw leaf byte strings. Empty tree = SHA-256 of the empty string (RFC 6962)."""
    if not leaves:
        return sha256_hex(b"")
    return _levels(leaves)[-1][0].hex()


def merkle_proof(leaves: list[bytes], index: int) -> list[tuple[str, str]]:
    """Audit path for leaf `index`: a list of (sibling_hex, side) where side says where the sibling sits."""
    if not 0 <= index < len(leaves):
        raise IndexError(index)
    proof = []
    for level in _levels(leaves)[:-1]:
        sib = index ^ 1
        if sib < len(level):
            proof.append((level[sib].hex(), "L" if sib < index else "R"))
        index //= 2
    return proof


def verify_proof(leaf: bytes, proof: list[tuple[str, str]], root_hex: str) -> bool:
    h = leaf_hash(leaf)
    for sib_hex, side in proof:
        sib = bytes.fromhex(sib_hex)
        h = node_hash(sib, h) if side == "L" else node_hash(h, sib)
    return h.hex() == root_hex


def max_proof_len(n: int) -> int:
    """Upper bound on audit-path length: ceil(log2 n)."""
    return 0 if n <= 1 else math.ceil(math.log2(n))
