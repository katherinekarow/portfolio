import hashlib

import numpy as np
import pytest

from ledgerkit import analytics, benchmark, chain as ch, models, tamper, workload
from ledgerkit.cli import run
from ledgerkit.hashing import merkle_proof, merkle_root, verify_proof
from ledgerkit.offchain import DocStore, verify_documents


def H(b):
    return hashlib.sha256(b).digest()


@pytest.fixture(scope="module")
def stream():
    return workload.make_stream(n_pay=1200, seed=11)


@pytest.fixture(scope="module")
def built(stream):
    return ch.build(stream["events"], block_size=40)


# ---------------------------------------------------------------- Merkle
def test_merkle_root_known_vectors():
    # RFC 6962: empty tree = SHA-256(""), a well-known constant
    assert merkle_root([]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    # three leaves recomputed by hand: ((a,b),c) with leaf/node prefixes, c promoted unchanged
    la, lb, lc = (H(b"\x00" + x) for x in (b"a", b"b", b"c"))
    expect = H(b"\x01" + H(b"\x01" + la + lb) + lc).hex()
    assert merkle_root([b"a", b"b", b"c"]) == expect
    # fixed regression vector for four leaves
    assert merkle_root([b"a", b"b", b"c", b"d"]) == H(b"\x01" + H(b"\x01" + la + lb) + H(b"\x01" + lc + H(b"\x00d"))).hex()


def test_merkle_proofs_verify_and_detect_tampering():
    leaves = [f"tx-{i}".encode() for i in range(37)]
    root = merkle_root(leaves)
    for i in range(len(leaves)):
        pr = merkle_proof(leaves, i)
        assert len(pr) <= 6  # ceil(log2 37)
        assert verify_proof(leaves[i], pr, root)
    pr = merkle_proof(leaves, 5)
    assert not verify_proof(b"tx-5-edited", pr, root)
    bad = [(("0" * 64), pr[0][1])] + pr[1:]
    assert not verify_proof(leaves[5], bad, root)


# ---------------------------------------------------------------- chain + audit
def test_clean_chain_validates(built):
    c, _ = built
    rep = ch.audit(c, checkpoint=c.head)
    assert rep["valid"] and rep["checkpoint_ok"]


@pytest.mark.parametrize("mode,bits", [("poa", 0), ("pow", 6)])
def test_tamper_detected_at_exact_block(stream, mode, bits):
    c, _ = ch.build(stream["events"], block_size=40, mode=mode, difficulty=bits)
    k = 9
    t, _ = tamper.tamper(c, k, 3, 1, "edit")
    rep = ch.audit(t)
    assert rep["first_failure"] == (k, "merkle")
    assert [b for b, _ in rep["failures"]] == [k]  # nothing before block k is blamed
    t, _ = tamper.tamper(c, k, 3, 1, "rehash")
    rep = ch.audit(t)
    expected = (k, "consensus") if mode == "poa" else (k + 1, "link")
    assert rep["first_failure"] == expected


def test_full_rewrite_needs_checkpoint_under_pow_but_fails_quorum_under_poa(stream):
    cp, _ = ch.build(stream["events"], block_size=40, mode="pow", difficulty=6)
    t, info = tamper.tamper(cp, 5, 0, 1, "rewrite")
    assert ch.audit(t)["valid"]  # internally consistent after re-mining...
    assert ch.audit(t, checkpoint=cp.head)["failures"] == [(None, "checkpoint")]  # ...but not the published head
    assert info["hashes"] > 0
    c, _ = ch.build(stream["events"], block_size=40, mode="poa")
    t, _ = tamper.tamper(c, 5, 0, 1, "rewrite")
    assert ch.audit(t)["first_failure"] == (5, "consensus")


def test_pow_blocks_meet_difficulty(stream):
    c, _ = ch.build(stream["events"][:300], block_size=50, mode="pow", difficulty=10)
    assert all(ch.meets_difficulty(b.hash, 10) and b.hash.startswith("00") for b in c.blocks)
    assert all(int(b.hash, 16) < 2 ** (256 - 10) for b in c.blocks)


def test_measured_hashes_match_two_to_the_bits():
    sw = benchmark.pow_sweep([4, 6, 8], budget=2 ** 16, max_trials=300, min_trials=100)
    # geometric distribution: mean 2^bits; with >=256 trials the sample mean is within ~12% at 2 SE
    assert np.all(np.abs(sw.ratio - 1) < 0.2), sw


# ---------------------------------------------------------------- logical layer
def test_utxo_and_account_models_agree(stream):
    acc, utxo = models.AccountModel(), models.UTXOModel()
    _, ra = models.replay(stream["events"], acc)
    _, ru = models.replay(stream["events"], utxo)
    assert acc.balances() == utxo.balances()
    assert [e["seq"] for e, _ in ra] == [e["seq"] for e, _ in ru]
    total = sum(e["amount"] for e in stream["events"] if e["kind"] == "mint")
    assert sum(acc.balances().values()) == total  # money is conserved
    db, _ = benchmark.run_sqlite(stream["events"])
    assert benchmark.sqlite_balances(db) == acc.balances()


def test_double_spend_rejected():
    u = models.UTXOModel()
    u.apply({"kind": "mint", "from": None, "to": "Alder", "amount": 100})
    coin = next(iter(u.coins))
    u.apply_raw([coin], [("Birch", 100)], "Alder")
    with pytest.raises(models.Rejected, match="already spent"):
        u.apply_raw([coin], [("Cedar", 100)], "Alder")
    u.apply({"kind": "mint", "from": None, "to": "Elm", "amount": 50})
    c2 = next(c for c, (o, _) in u.coins.items() if o == "Elm")
    with pytest.raises(models.Rejected, match="same coin twice"):
        u.apply_raw([c2, c2], [("Fir", 100)], "Elm")


def test_overdraft_rejected_in_both_models(stream):
    for m in (models.AccountModel(), models.UTXOModel()):
        _, bad = models.replay(stream["events"], m)
        assert len(bad) == stream["planted"]["n_overdraft"]
        assert {r for _, r in bad} == {"overdraft"}
        assert all(e["amount"] == stream["planted"]["overdraft_amount"] for e, _ in bad)


# ---------------------------------------------------------------- off-chain
def test_offchain_erasure_keeps_chain_valid(stream, built):
    c, _ = built
    store = DocStore()
    store.load(stream["payloads"], stream["salts"])
    before = verify_documents(c, store)
    assert before["mismatch"] == before["erased"] == before["missing"] == 0
    inv, onchain = next((tx["invoice"], tx["doc"]) for _, _, tx in c.transactions() if tx["kind"] == "pay")
    store.erase(inv)
    assert store.check(inv, onchain) == "erased"
    assert store.get(inv)[1] is None and store.get(inv)[0] is None  # payload and salt gone
    assert any(tx["doc"] == onchain for _, _, tx in c.transactions())  # hash still on-chain
    assert ch.audit(c, checkpoint=c.head)["valid"]
    other = next(tx["invoice"] for _, _, tx in c.transactions() if tx["kind"] == "pay" and tx["invoice"] != inv)
    store.corrupt(other)
    after = verify_documents(c, store)
    assert after["erased"] == 1 and after["mismatch"] == 1


# ---------------------------------------------------------------- analytics
def test_planted_wash_ring_found(built):
    c, _ = built
    rings = analytics.flag_rings(analytics.find_loops(analytics.payments_table(c)))
    assert len(rings) == 1
    assert set(rings.ring[0].split(" -> ")) == set(workload.WASH_RING)
    assert rings.rounds[0] == 3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_false_rings_in_clean_data(seed):
    s = workload.make_stream(n_pay=2000, seed=seed, wash_rounds=0)
    c, _ = ch.build(s["events"])
    loops = analytics.find_loops(analytics.payments_table(c))
    assert len(analytics.flag_rings(loops)) == 0


# ---------------------------------------------------------------- determinism + CLI
def test_deterministic():
    a = workload.make_stream(n_pay=500, seed=5)
    b = workload.make_stream(n_pay=500, seed=5)
    assert a["events"] == b["events"] and a["salts"] == b["salts"]
    assert ch.build(a["events"])[0].head == ch.build(b["events"])[0].head
    assert ch.build(a["events"], mode="pow", difficulty=6)[0].head == ch.build(b["events"], mode="pow", difficulty=6)[0].head


def test_cli_writes_files(tmp_path):
    sm = run(tmp_path / "out", tmp_path / "fig", n_pay=800, quick=True)
    for f in ("benchmark.csv", "audit_report.txt", "pow_sweep.csv", "merkle_proofs.csv", "flagged_rings.csv", "tamper_results.csv"):
        assert (tmp_path / "out" / f).stat().st_size > 0
    assert len(list((tmp_path / "fig").glob("*.png"))) == 4
    assert sm["models_agree"] and sm["sqlite_agrees"] and sm["clean_audit_valid"]
    assert all(r["detected"] for r in sm["tamper"])
    assert "VALID" in (tmp_path / "out" / "audit_report.txt").read_text()
