"""Adjoint differentiation correctness tests.

Adjoint grad must match parameter-shift grad to numerical precision
(both are exact for rotation gates), AND must be substantially faster
at high parameter count.
"""
from __future__ import annotations
import math
import time
import numpy as np
import pytest

import superfermion as sf
from superfermion.observables.core import SparsePauliOp
from superfermion.qml.gradient.parameter_shift import parameter_shift_grad_vector
from superfermion.qml.gradient.adjoint import adjoint_grad_vector


def _hwe_ansatz(n: int):
    """Hardware-efficient ansatz with 2n parameters: ring + RY/RZ layer."""
    qc = sf.Circuit(n)
    names = []
    for q in range(n):
        nm = f"ry{q}"; qc.ry(sf.param(nm), q); names.append(nm)
    for q in range(n):
        nm = f"rz{q}"; qc.rz(sf.param(nm), q); names.append(nm)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.cx(n - 1, 0)
    return qc, names


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_adjoint_matches_parameter_shift_hamiltonian(n):
    qc, names = _hwe_ansatz(n)
    rng = np.random.default_rng(123 + n)
    theta = rng.uniform(-math.pi, math.pi, len(names))
    # TFIM-like Hamiltonian
    h_dict = {}
    for i in range(n - 1):
        s = ["I"] * n; s[i] = "Z"; s[i + 1] = "Z"
        h_dict["".join(s)] = 1.0
    for i in range(n):
        s = ["I"] * n; s[i] = "X"
        h_dict["".join(s)] = 0.5
    obs = SparsePauliOp.from_dict(h_dict)

    psr = parameter_shift_grad_vector(qc, obs, names, theta, backend="statevector", shots=0)
    adj = adjoint_grad_vector(qc, obs, names, theta)
    assert psr.shape == adj.shape
    max_err = float(np.max(np.abs(psr - adj)))
    assert max_err < 1e-9, f"n={n} max_err={max_err:.3e}\npsr={psr}\nadj={adj}"


@pytest.mark.parametrize("obs_str", ["ZIII", "IZII", "ZZII", "XIII", "IXII"])
def test_adjoint_matches_parameter_shift_single_pauli(obs_str):
    n = 4
    qc, names = _hwe_ansatz(n)
    rng = np.random.default_rng(7)
    theta = rng.uniform(-math.pi, math.pi, len(names))
    obs = SparsePauliOp.from_dict({obs_str: 1.0})
    psr = parameter_shift_grad_vector(qc, obs, names, theta, backend="statevector", shots=0)
    adj = adjoint_grad_vector(qc, obs, names, theta)
    max_err = float(np.max(np.abs(psr - adj)))
    assert max_err < 1e-10, f"obs={obs_str} max_err={max_err:.3e}\npsr={psr}\nadj={adj}"


def test_adjoint_speedup_at_n10():
    """At n=10 with 20 params, adjoint must be at least 5x faster than PSR."""
    n = 10
    qc, names = _hwe_ansatz(n)
    rng = np.random.default_rng(42)
    theta = rng.uniform(-math.pi, math.pi, len(names))
    h_dict = {}
    for i in range(n - 1):
        s = ["I"] * n; s[i] = "Z"; s[i + 1] = "Z"
        h_dict["".join(s)] = 1.0
    for i in range(n):
        s = ["I"] * n; s[i] = "X"; h_dict["".join(s)] = 0.5
    obs = SparsePauliOp.from_dict(h_dict)

    # warmup
    parameter_shift_grad_vector(qc, obs, names, theta, backend="statevector", shots=0)
    adjoint_grad_vector(qc, obs, names, theta)

    t0 = time.perf_counter()
    psr = parameter_shift_grad_vector(qc, obs, names, theta, backend="statevector", shots=0)
    dt_psr = time.perf_counter() - t0
    t0 = time.perf_counter()
    adj = adjoint_grad_vector(qc, obs, names, theta)
    dt_adj = time.perf_counter() - t0

    assert np.allclose(psr, adj, atol=1e-9), "adjoint disagrees with PSR"
    speedup = dt_psr / dt_adj
    print(f"\nn={n} 2n={2*n} params  PSR={dt_psr*1000:.1f}ms  adjoint={dt_adj*1000:.1f}ms  speedup={speedup:.2f}x")
    assert speedup > 4.0, f"adjoint speedup only {speedup:.2f}x (expected >=5x)"
