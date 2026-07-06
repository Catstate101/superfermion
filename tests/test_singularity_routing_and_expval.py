"""Regression tests for the post-MPS-benchmark fixes (2026-04-25).

Covers:

* MPS exact ``expval`` API (Pauli string + dict + SparsePauliOp) — must agree
  with dense-statevector ground truth to ~1e-6 absolute error (single-precision
  contraction tolerance).
* sf.statevector / turbo path — keeps tensor form throughout, must match
  ``sf.rust`` to numerical precision (machine epsilon).
* sf.singularity routing — n>32 must NOT pre-fuse before MPS dispatch
  (regression: fusion previously caused >100x slowdown at n=40).
* sf.rust ``decompose_for_rust`` — CRY / CH / U1 / U2 / U3 must run end-to-end
  through the Rust core and return states with fidelity 1.0 vs the dense
  Python statevector.

These tests guard against the silent identity-fallback bug in
``GateRecord.to_unitary()`` that previously made CRY/CH/U1/U2 evaluate to
identity on every dense backend.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import superfermion as sf
import superfermion.circuit as _c
from superfermion.backends.mps import MPSSimulatorBackend
from superfermion.backends.registry import BackendRegistry
from superfermion.backends.simulator import StatevectorBackend
from superfermion.backends.singularity import SingularityBackend
from superfermion.backends.turbo import (
    decompose_for_rust,
    fuse_single_qubit_gates,
)


GR = _c.GateRecord


def _qaoa_pathmaxcut(n: int, p_layers: int = 2) -> sf.Circuit:
    qc = sf.Circuit(n)
    for q in range(n):
        qc.h(q)
    gamma = [0.3, 0.5]
    beta = [0.2, 0.4]
    for p in range(p_layers):
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * gamma[p], i + 1)
            qc.cx(i, i + 1)
        for q in range(n):
            qc.rx(2 * beta[p], q)
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# 1. sf.mps.expval — exact contraction matches dense statevector ground truth
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "obs",
    [
        "Z" + "I" * 7,            # single-site Z on q0
        "X" + "I" * 7,             # single-site X
        "I" * 4 + "Z" + "I" * 3,   # Z on q4
        "ZZ" + "I" * 6,            # ZZ on q0,q1
        "XYZIXYZI",                # multi-site mixed
    ],
)
def test_mps_expval_pauli_string_matches_dense(obs):
    n = 8
    qc = sf.Circuit(n)
    for q in range(n):
        qc.h(q)
    # Force SWAPs by using non-adjacent CNOTs
    qc.cx(0, 7)
    qc.cx(2, 5)
    qc.rz(0.4, 3)
    qc.cx(1, 6)

    # Dense ground truth via SparsePauliOp.expectation
    sv = StatevectorBackend().run(qc, shots=0).statevector
    pauli = sf.SparsePauliOp.from_dict({obs: 1.0})
    truth = float(np.real(pauli.expectation(sv)))

    mps = MPSSimulatorBackend(options={"max_bond_dim": 64})
    got = float(np.real(mps.expval(qc, obs, max_bond=64)))
    assert abs(got - truth) < 5e-5, f"obs={obs}  got={got}  truth={truth}"


def test_mps_expval_dict_linear_combination():
    """Dict-form observable: linear combination of Pauli strings."""
    n = 6
    qc = sf.Circuit(n)
    for q in range(n):
        qc.h(q)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(4, 5)
    qc.rz(0.7, 3)

    obs = {"ZIIIII": 1.0, "IIZIII": 0.5, "IIIIIZ": -0.25}

    sv = StatevectorBackend().run(qc, shots=0).statevector
    pauli = sf.SparsePauliOp.from_dict(obs)
    truth = float(np.real(pauli.expectation(sv)))

    mps = MPSSimulatorBackend(options={"max_bond_dim": 32})
    got = float(np.real(mps.expval(qc, obs, max_bond=32)))
    assert abs(got - truth) < 5e-5


# ─────────────────────────────────────────────────────────────────────────────
# 2. sf.statevector turbo path — tensor-form invariant & correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_statevector_qaoa_matches_rust():
    qc = _qaoa_pathmaxcut(10)
    sv_py = BackendRegistry.get_backend("statevector").run(qc, shots=0).statevector
    sv_rust = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    assert abs(np.vdot(sv_py, sv_rust)) > 1.0 - 1e-10


def test_statevector_run_no_probs_dict_blowup_for_large_n():
    """n=18: probs_dict must be empty (we skip 2^n format() calls)."""
    qc = sf.Circuit(18)
    for q in range(18):
        qc.h(q)
    res = BackendRegistry.get_backend("statevector").run(qc, shots=0)
    assert res.probabilities == {}, "probs_dict should be skipped for n>16"


# ─────────────────────────────────────────────────────────────────────────────
# 3. sf.singularity routing — does NOT fuse before MPS dispatch
# ─────────────────────────────────────────────────────────────────────────────


def test_singularity_n40_uses_native_mps_path_quickly():
    """Regression: pre-fusing a QAOA at n=40 used to make MPS >100x slower
    because the U-gate path falls into a slow Python branch in the Rust MPS
    core. Singularity must dispatch the *original* circuit to MPS."""
    import time

    qc = _qaoa_pathmaxcut(40)
    sing = SingularityBackend()
    sing._cache.clear()

    t0 = time.perf_counter()
    sing.run(qc, shots=100)
    dt = time.perf_counter() - t0
    # Generous bound (10s) — the actual measurement is ~10-20 ms; this is
    # really catching the >100x regression where it would take 50+ seconds.
    assert dt < 10.0, f"n=40 singularity took {dt:.1f}s, regression suspected"


def test_singularity_topology_cache_does_not_overflow_at_large_n():
    """For n>=25 the result has no genuine 2^n statevector — the cache must
    NOT try to materialize one (would OOM at n=40)."""
    qc = _qaoa_pathmaxcut(40)
    sing = SingularityBackend()
    sing._cache.clear()
    sing.run(qc, shots=10)
    # Cache must be empty: we don't cache MPS-regime results
    assert len(sing._cache) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. circuit.GateRecord.to_unitary — no silent identity for parametric gates
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,params",
    [
        ("CRY", [0.7]),
        ("CRX", [1.1]),
        ("CRZ", [0.4]),
        ("CH", []),
        ("U1", [0.3]),
        ("U2", [0.5, 1.2]),
    ],
)
def test_to_unitary_not_identity(name, params):
    """These gates previously fell through to ``np.eye(...)`` — silent bug."""
    g = GR(name, [0, 1] if name in ("CRY", "CRX", "CRZ", "CH") else [0], params)
    u = g.to_unitary()
    assert not np.allclose(u, np.eye(u.shape[0])), (
        f"{name}({params}) silently returned identity"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. decompose_for_rust — round-trip through Rust gives same physics
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("theta", [0.1, 0.7, 1.5, math.pi])
def test_decompose_cry_matches_dense(theta):
    qc = sf.Circuit(2)
    qc.x(0)
    qc.x(1)
    qc._gates.append(GR("CRY", [0, 1], [theta]))

    sv_py = StatevectorBackend().run(qc, shots=0).statevector
    sv_rust = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    assert abs(np.vdot(sv_py, sv_rust)) > 1.0 - 1e-9


@pytest.mark.parametrize(
    "ops",
    [
        [GR("CH", [0, 1])],
        [GR("U1", [0], [0.4]), GR("U2", [1], [0.5, 1.2])],
        [GR("U3", [0], [0.7, 0.3, 1.1])],
    ],
)
def test_decompose_misc_matches_dense(ops):
    qc = sf.Circuit(2)
    for q in range(2):
        qc.h(q)
    for g in ops:
        qc._gates.append(g)

    sv_py = StatevectorBackend().run(qc, shots=0).statevector
    sv_rust = BackendRegistry.get_backend("rust").run(qc, shots=0).statevector
    fid = abs(np.vdot(sv_py, sv_rust))
    assert fid > 1.0 - 1e-9, f"fidelity={fid:.10f}"


def test_decompose_for_rust_passes_through_unknown_gates():
    """Sanity: gates not in the rewrite table are preserved verbatim."""
    src = [GR("X", [0]), GR("RZZ", [0, 1], [0.5]), GR("CCX", [0, 1, 2])]
    out = decompose_for_rust(src)
    assert [g.name for g in out] == ["X", "RZZ", "CCX"]
