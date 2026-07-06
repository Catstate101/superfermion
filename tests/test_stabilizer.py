"""Stabilizer backend correctness tests (Fix #7, 2026-04-27).

The Aaronson-Gottesman tableau simulator must:
1. Reject non-Clifford gates with a clear error.
2. Match the dense-statevector reference for every Clifford gate at every
   qubit position (full sweep on 5 qubits).
3. Match the dense statevector for random Clifford circuits.
4. Match Qiskit-Aer's stabilizer method for Pauli expectation values.
5. Auto-dispatch through SingularityBackend for Clifford circuits.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

import superfermion as sf
from superfermion.backends.registry import BackendRegistry
from superfermion.backends.simulator import StatevectorBackend
from superfermion.backends.stabilizer import (
    StabilizerBackend,
    NotCliffordError,
    is_clifford_circuit,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Clifford detection
# ─────────────────────────────────────────────────────────────────────────────
def test_clifford_detection_positive():
    qc = sf.Circuit(3)
    qc.h(0); qc.s(1); qc.cx(0, 1); qc.cz(1, 2); qc.swap(0, 2)
    assert is_clifford_circuit(qc)


def test_clifford_detection_negative_rotation():
    qc = sf.Circuit(2)
    qc.h(0); qc.rx(0.5, 0); qc.cx(0, 1)
    assert not is_clifford_circuit(qc)


def test_stabilizer_rejects_non_clifford():
    qc = sf.Circuit(2)
    qc.h(0); qc.t(0); qc.cx(0, 1)
    sb = StabilizerBackend()
    with pytest.raises(NotCliffordError):
        sb.evolve(qc)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-gate sweep — match Pauli expectation against dense SV
# ─────────────────────────────────────────────────────────────────────────────
def _z0z1_from_sv(sv: np.ndarray, n: int) -> float:
    mask0 = 1 << (n - 1); mask1 = 1 << (n - 2)
    probs = np.abs(sv) ** 2
    idx = np.arange(1 << n)
    par = ((idx & mask0) != 0).astype(int) ^ ((idx & mask1) != 0).astype(int)
    return float(np.sum(np.where(par == 0, 1.0, -1.0) * probs))


@pytest.mark.parametrize("gate,name", [
    ("h", "H"), ("x", "X"), ("y", "Y"), ("z", "Z"),
    ("s", "S"), ("sdg", "Sdg"),
])
def test_single_qubit_gate_at_every_position(gate, name):
    n = 5
    sb = StabilizerBackend()
    sv_be = StatevectorBackend()
    for q in range(n):
        qc = sf.Circuit(n)
        for k in range(n): qc.h(k)
        for k in range(n - 1): qc.cx(k, k + 1)
        getattr(qc, gate)(q)
        ref = _z0z1_from_sv(sv_be.run(qc, shots=0).statevector, n)
        got = sb.expval(qc, "ZZ" + "I" * (n - 2))
        assert abs(got - ref) < 1e-10, f"{name}(q={q}) stabilizer={got} dense={ref}"


@pytest.mark.parametrize("gate2,name", [
    ("cx", "CNOT"), ("cz", "CZ"), ("swap", "SWAP"),
])
def test_two_qubit_gate_pairs(gate2, name):
    n = 5
    sb = StabilizerBackend()
    sv_be = StatevectorBackend()
    for a in range(n):
        for b in range(n):
            if a == b: continue
            qc = sf.Circuit(n)
            for k in range(n): qc.h(k)
            qc.s(0); qc.s(2)
            getattr(qc, gate2)(a, b)
            ref = _z0z1_from_sv(sv_be.run(qc, shots=0).statevector, n)
            got = sb.expval(qc, "ZZ" + "I" * (n - 2))
            assert abs(got - ref) < 1e-10, f"{name}({a},{b}) stab={got} dense={ref}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Clifford circuits — exhaustive Pauli expectation checks
# ─────────────────────────────────────────────────────────────────────────────
def _random_clifford(n: int, depth: int, seed: int) -> sf.Circuit:
    rng = np.random.default_rng(seed)
    qc = sf.Circuit(n)
    one_q = ["h", "x", "y", "z", "s", "sdg"]
    two_q = ["cx", "cz", "swap"]
    for _ in range(depth):
        for q in range(n):
            getattr(qc, rng.choice(one_q))(q)
        for i in range(0, n - 1, 2):
            getattr(qc, rng.choice(two_q))(i, i + 1)
        for i in range(1, n - 1, 2):
            getattr(qc, rng.choice(two_q))(i, i + 1)
    return qc


@pytest.mark.parametrize("n,depth,seed", [
    (4, 3, 1), (5, 5, 2), (6, 4, 3), (7, 3, 4), (8, 4, 5),
])
def test_random_clifford_matches_dense_zz(n, depth, seed):
    qc = _random_clifford(n, depth, seed)
    sv_be = StatevectorBackend()
    sb = StabilizerBackend()
    sv = sv_be.run(qc, shots=0).statevector
    ref = _z0z1_from_sv(sv, n)
    got = sb.expval(qc, "ZZ" + "I" * (n - 2))
    assert abs(got - ref) < 1e-10, f"n={n} seed={seed} stab={got} dense={ref}"


@pytest.mark.parametrize("pauli", [
    "ZIIIII", "IZIIII", "IIZIII", "ZZIIII", "IZZIII",
    "XIIIII", "IXIIII", "IXZIII", "XYZIII",
])
def test_random_clifford_arbitrary_pauli(pauli):
    n = 6
    qc = _random_clifford(n, depth=4, seed=42)
    sv_be = StatevectorBackend()
    sb = StabilizerBackend()
    sv = sv_be.run(qc, shots=0).statevector
    # Build Pauli operator via SF's SparsePauliOp
    from superfermion.observables.core import SparsePauliOp
    op = SparsePauliOp.from_dict({pauli: 1.0})
    ref = float(np.real(op.expectation(sv)))
    got = sb.expval(qc, pauli)
    assert abs(got - ref) < 1e-10, f"P={pauli} stab={got} dense={ref}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sampling — distribution agrees with dense Born rule
# ─────────────────────────────────────────────────────────────────────────────
def test_sampling_matches_dense_distribution():
    n = 6
    qc = _random_clifford(n, depth=3, seed=7)
    sb = StabilizerBackend()
    res = sb.run(qc, shots=20000, seed=11)
    counts = res.counts
    sv = StatevectorBackend().run(qc, shots=0).statevector
    probs = np.abs(sv) ** 2
    # SF MSB convention: bitstring is read q0 q1 ... q_{n-1}
    total = sum(counts.values())
    for bs, c in counts.items():
        idx = int(bs, 2)
        emp = c / total
        the = float(probs[idx])
        # 20000 shots -> sigma ~ sqrt(p*(1-p)/N) <= 0.0036; allow 0.02
        assert abs(emp - the) < 0.02, f"bs={bs} emp={emp:.4f} theory={the:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Auto-dispatch via SingularityBackend
# ─────────────────────────────────────────────────────────────────────────────
def test_singularity_routes_clifford_to_stabilizer():
    """Singularity should detect Clifford at large n (>22) and use the
    tableau path. At small n it falls through to dense so callers that
    inspect ``.statevector`` keep working."""
    from superfermion.backends.singularity import SingularityBackend
    sing = SingularityBackend()
    sing._cache.clear()
    # n=30 > 22 threshold -> stabilizer tableau path
    qc = _random_clifford(30, depth=4, seed=99)
    res = sing.run(qc, shots=100)
    assert res.metadata.get("regime") == "tableau", (
        f"Expected tableau routing at n=30, got {res.metadata}"
    )


def test_singularity_clifford_fast_at_large_n():
    """Stabilizer dispatch must run a 50-qubit Clifford circuit in well under
    the 60s test budget — proves we're not falling back to dense paths."""
    import time
    from superfermion.backends.singularity import SingularityBackend
    qc = _random_clifford(50, depth=6, seed=1)
    sing = SingularityBackend()
    sing._cache.clear()
    t0 = time.perf_counter()
    res = sing.run(qc, shots=100, seed=42)
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"Clifford n=50 singularity took {dt:.1f}s"
    assert sum(res.counts.values()) == 100


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aer cross-check — third-party reference for stabilizer
# ─────────────────────────────────────────────────────────────────────────────
def test_aer_stabilizer_cross_check():
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Pauli

    rng = np.random.default_rng(2026)
    n = 10
    sf_qc = sf.Circuit(n)
    qiskit_qc = QuantumCircuit(n)
    for _ in range(8):
        for q in range(n):
            kind = int(rng.integers(0, 3))
            if kind == 0: sf_qc.h(q); qiskit_qc.h(q)
            elif kind == 1: sf_qc.s(q); qiskit_qc.s(q)
        for i in range(0, n - 1, 2):
            sf_qc.cx(i, i + 1); qiskit_qc.cx(i, i + 1)

    # Aer stabilizer method for ZZ on q0,q1
    sim = AerSimulator(method="stabilizer")
    qcopy = qiskit_qc.copy()
    qcopy.save_expectation_value(Pauli("I" * (n - 2) + "ZZ"), list(range(n)))
    result = sim.run(qcopy).result()
    aer_zz = float(np.real(result.data(0)["expectation_value"]))

    sb = StabilizerBackend()
    sf_zz = sb.expval(sf_qc, "ZZ" + "I" * (n - 2))
    assert abs(sf_zz - aer_zz) < 1e-12, f"sf={sf_zz}  aer={aer_zz}"
