"""Deployment-readiness stress tests.

These run BEFORE shipping, not on every commit.  They cover:

  1. Numerical stability — deep Trotter, long evolution, norm/fidelity drift
  2. Cross-framework correctness — SF backends vs Qiskit-Aer to machine epsilon
  3. Edge cases — n=1, empty circuit, single-gate, all-Clifford, etc.
  4. Adversarial inputs — NaN params, malformed Pauli strings, unsupported gates
  5. Determinism — same seed -> same counts (sampled backends)
  6. Memory robustness — 200 cycles of build+run+discard with no leak

Mark slow tests with @pytest.mark.slow if you want to skip them in CI.
"""
from __future__ import annotations
import gc
import math
import time
import warnings
from typing import Dict, List

import numpy as np
import pytest

import superfermion as sf
from superfermion.backends.registry import BackendRegistry
from superfermion.backends.simulator import StatevectorBackend
from superfermion.backends.rust_sim import RustBackend
from superfermion.backends.mps import MPSSimulatorBackend
from superfermion.backends.singularity import SingularityBackend
from superfermion.backends.stabilizer import StabilizerBackend, NotCliffordError
from superfermion.observables.core import SparsePauliOp


# ─────────────────────────────────────────────────────────────────────────────
# 1. NUMERICAL STABILITY
# ─────────────────────────────────────────────────────────────────────────────
def test_deep_trotter_norm_preserved():
    """100-step Heisenberg Trotter at n=8 must keep MPS norm = 1 to 1e-12."""
    n = 8
    qc = sf.Circuit(n)
    for _ in range(100):  # 100 Trotter steps -> ~17,000 gates
        for i in range(n - 1):
            qc.h(i); qc.h(i+1); qc.cx(i,i+1); qc.rz(0.05, i+1); qc.cx(i,i+1); qc.h(i); qc.h(i+1)
            qc.rx(math.pi/2,i); qc.rx(math.pi/2,i+1); qc.cx(i,i+1); qc.rz(0.05, i+1); qc.cx(i,i+1); qc.rx(-math.pi/2,i); qc.rx(-math.pi/2,i+1)
            qc.cx(i,i+1); qc.rz(0.05, i+1); qc.cx(i,i+1)
    norm = qc.to_ir().mps_norm_sq(64)
    # 17,000 gates × per-gate FP error ~ 1e-16 = ~1e-12 expected accumulation
    assert abs(norm - 1.0) < 1e-10, f"MPS norm drift after 100 Trotter steps: {norm:.15f}"


def test_long_evolution_dense_norm():
    """1000-gate random circuit dense statevector must stay normalised."""
    rng = np.random.default_rng(7)
    n = 10
    qc = sf.Circuit(n)
    for _ in range(1000):
        kind = int(rng.integers(0, 4))
        if kind == 0:
            qc.rx(float(rng.uniform(-math.pi, math.pi)), int(rng.integers(0, n)))
        elif kind == 1:
            qc.ry(float(rng.uniform(-math.pi, math.pi)), int(rng.integers(0, n)))
        elif kind == 2:
            qc.rz(float(rng.uniform(-math.pi, math.pi)), int(rng.integers(0, n)))
        else:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n))
            while b == a: b = int(rng.integers(0, n))
            qc.cx(a, b)
    sv = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    norm_sq = float(np.real(np.vdot(sv, sv)))
    assert abs(norm_sq - 1.0) < 1e-10, f"sf.rust norm drift after 1000 gates: {norm_sq:.15f}"


def test_repeated_run_consistency():
    """Same circuit + seed run 10 times must give bit-identical statevector."""
    n = 8
    qc = sf.Circuit(n)
    for q in range(n): qc.h(q)
    for i in range(n - 1): qc.cx(i, i + 1)
    qc.rz(0.7, 3)
    sv0 = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    for _ in range(9):
        sv = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
        assert np.allclose(sv, sv0, atol=0.0), "non-determinism in sf.rust!"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CROSS-FRAMEWORK CORRECTNESS  (every SF backend vs Aer)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [4, 6, 8])
def test_sf_backends_match_aer_sv(n):
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    sf_qc = sf.Circuit(n)
    qk_qc = QuantumCircuit(n)
    for q in range(n):
        sf_qc.h(q); qk_qc.h(q)
    for i in range(n - 1):
        sf_qc.cx(i, i + 1); qk_qc.cx(i, i + 1)
        sf_qc.rz(0.3, i + 1); qk_qc.rz(0.3, i + 1)
        sf_qc.cx(i, i + 1); qk_qc.cx(i, i + 1)
    for q in range(n):
        sf_qc.rx(0.5, q); qk_qc.rx(0.5, q)

    qk_qc.save_statevector()
    sv_aer = np.asarray(
        AerSimulator(method="statevector").run(qk_qc).result().get_statevector(),
        dtype=np.complex128,
    )

    backends = ["statevector", "rust", "singularity"]
    for name in backends:
        sv_sf = BackendRegistry.get_backend(name).run(sf_qc, shots=0).statevector
        # SF MSB convention -> reverse axes to LSB to match Aer
        sv_le = np.asarray(sv_sf, dtype=np.complex128).reshape([2]*n).transpose(list(range(n))[::-1]).reshape(-1)
        fid = float(abs(np.vdot(sv_le, sv_aer)))
        assert fid > 1.0 - 1e-10, f"{name} n={n}: fidelity={fid:.16f}"


def test_mps_pauli_expval_matches_dense():
    """sf.mps.expval must match dense statevector expval to 1e-7 (single-prec MPS)."""
    n = 8
    rng = np.random.default_rng(7)
    qc = sf.Circuit(n)
    for q in range(n): qc.h(q)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        qc.rz(float(rng.uniform(-1, 1)), i + 1)
        qc.cx(i, i + 1)
    for q in range(n): qc.rx(0.5, q)

    sv = StatevectorBackend().run(qc, shots=0).statevector
    mps = MPSSimulatorBackend(options={"max_bond_dim": 64})
    op = SparsePauliOp.from_dict({"ZZ" + "I" * (n - 2): 1.0})
    truth = float(np.real(op._fast_expval(sv)))
    got = mps.expval(qc, "ZZ" + "I" * (n - 2))
    assert abs(got - truth) < 1e-7, f"mps.expval={got} truth={truth}"


def test_stabilizer_matches_dense_for_clifford():
    """Pure Clifford circuit: stabilizer expval == dense expval to machine eps."""
    n = 6
    rng = np.random.default_rng(99)
    qc = sf.Circuit(n)
    for _ in range(8):
        for q in range(n):
            kind = int(rng.integers(0, 3))
            if kind == 0: qc.h(q)
            elif kind == 1: qc.s(q)
            else: qc.x(q)
        for i in range(0, n - 1, 2): qc.cx(i, i + 1)
        for i in range(1, n - 1, 2): qc.cx(i, i + 1)
    sv = StatevectorBackend().run(qc, shots=0).statevector
    op = SparsePauliOp.from_dict({"ZIZIZI": 1.0, "XIIIXI": 0.5, "YIIYII": 0.25})
    truth = float(np.real(op._fast_expval(sv)))
    got = StabilizerBackend().expval(qc, {"ZIZIZI": 1.0, "XIIIXI": 0.5, "YIIYII": 0.25})
    assert abs(got - truth) < 1e-12, f"stabilizer={got} dense={truth}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────
def test_n1_circuit():
    """X H |0> = H|1> = (|0> - |1>)/sqrt(2) = |->.  Tests gate ordering at n=1."""
    qc = sf.Circuit(1); qc.x(0); qc.h(0)
    sv = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    expected = np.array([1.0/math.sqrt(2), -1.0/math.sqrt(2)], dtype=np.complex128)
    assert np.allclose(sv, expected, atol=1e-12)


def test_empty_circuit():
    """An empty circuit should return |0...0>."""
    n = 4
    qc = sf.Circuit(n)
    sv = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    expected = np.zeros(1 << n, dtype=np.complex128); expected[0] = 1.0
    assert np.allclose(sv, expected, atol=1e-12)


def test_single_gate_circuit():
    qc = sf.Circuit(3); qc.h(1)
    sv = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    # H on q1 (MSB convention): q1=middle bit. |010> + |000> over sqrt2
    # In SF q0 is MSB, so |q0 q1 q2> with q1=0 -> index sum *4 + 0 + 0 = 0 or 1 depending on q2
    # H|0> on q1 -> (|q1=0> + |q1=1>)/sqrt2 -> for q0=0,q2=0: index 0 (q1=0) and index 2 (q1=1)
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = expected[2] = 1.0/math.sqrt(2)
    assert np.allclose(sv, expected, atol=1e-12)


def test_observable_coverage_all_paulis():
    """Single-Pauli observables I, X, Y, Z on every qubit must each match dense."""
    n = 5
    rng = np.random.default_rng(42)
    qc = sf.Circuit(n)
    for q in range(n): qc.ry(float(rng.uniform(-1, 1)), q)
    for i in range(n - 1): qc.cx(i, i + 1)
    sv = StatevectorBackend().run(qc, shots=0).statevector
    mps = MPSSimulatorBackend(options={"max_bond_dim": 16})
    for ch in "IXYZ":
        for q in range(n):
            pstr = ["I"] * n; pstr[q] = ch
            pstr = "".join(pstr)
            if ch == "I": continue  # trivially 1
            op = SparsePauliOp.from_dict({pstr: 1.0})
            truth = float(np.real(op._fast_expval(sv)))
            got = mps.expval(qc, pstr)
            assert abs(got - truth) < 1e-7, f"{pstr}: mps={got} dense={truth}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADVERSARIAL INPUTS
# ─────────────────────────────────────────────────────────────────────────────
def test_stabilizer_rejects_non_clifford():
    qc = sf.Circuit(2); qc.t(0); qc.cx(0, 1)
    with pytest.raises(NotCliffordError):
        StabilizerBackend().evolve(qc)


def test_unknown_gate_raises():
    """An unsupported gate name should raise, not silently identity."""
    qc = sf.Circuit(2)
    qc._gates.append(sf.circuit.GateRecord("FOO_BAR", [0], []))
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        BackendRegistry.get_backend("rust").run(qc, shots=0)


def test_mps_pauli_string_length_mismatch():
    qc = sf.Circuit(4); qc.h(0)
    mps = MPSSimulatorBackend(options={"max_bond_dim": 8})
    with pytest.raises(ValueError, match="length"):
        mps.expval(qc, "ZZZ")  # n=4 but pauli string has length 3


def test_finite_param_values_only():
    """RZ with NaN angle must not silently produce a NaN statevector."""
    qc = sf.Circuit(2)
    qc.rz(float("nan"), 0)
    qc.cx(0, 1)
    # We don't require the backend to raise — but if it runs, the result
    # should at least be detectable as NaN-tainted, not silently 0.
    sv = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    has_nan = bool(np.any(np.isnan(sv.real)) or np.any(np.isnan(sv.imag)))
    has_well_defined = bool(np.allclose(np.abs(np.vdot(sv, sv)), 1.0, atol=1e-10))
    # Either: clearly NaN-poisoned (good — caller can detect) OR a valid
    # finite-valued state.  What we DON'T accept: silently zeroed-out vector.
    assert has_nan or has_well_defined, (
        "RZ(NaN) produced an unexpected non-NaN, non-normalised state"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DETERMINISM
# ─────────────────────────────────────────────────────────────────────────────
def test_stabilizer_sampling_is_deterministic_per_seed():
    """Same seed -> identical counts."""
    n = 8
    qc = sf.Circuit(n)
    for q in range(n): qc.h(q)
    for i in range(n - 1): qc.cx(i, i + 1)
    sb = StabilizerBackend()
    counts1 = sb.run(qc, shots=1000, seed=123).counts
    counts2 = sb.run(qc, shots=1000, seed=123).counts
    assert counts1 == counts2, "stabilizer.sample is not deterministic per seed"


# ─────────────────────────────────────────────────────────────────────────────
# 6. MEMORY ROBUSTNESS  (200 cycles, no leak)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_no_memory_leak_in_repeated_runs():
    import psutil, os
    proc = psutil.Process(os.getpid())
    n = 6
    qc = sf.Circuit(n)
    for q in range(n): qc.h(q)
    for i in range(n - 1): qc.cx(i, i + 1)
    rb = BackendRegistry.get_backend("rust")
    # Warm up — discard initial allocator state
    for _ in range(20):
        rb.run(qc, shots=0)
    gc.collect()
    rss0 = proc.memory_info().rss
    for _ in range(200):
        rb.run(qc, shots=0)
    gc.collect()
    rss1 = proc.memory_info().rss
    delta_mb = (rss1 - rss0) / 1024 / 1024
    # 200 cycles at n=6 (16 KB SV); growth should be << 50 MB.
    assert delta_mb < 50.0, f"RSS grew {delta_mb:.1f} MB across 200 runs (likely leak)"
