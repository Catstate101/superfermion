"""Regression: MPS fidelity must be non-decreasing in the bond cap.

Guards the block-diagonal fast path's row selection.  Before the repair,
brick12@bond 16 dropped a merge whose live rows sat outside the C-row
prefix and reported eps = 0.987 as an honest truncation, so F(16) = 0.0032
against F(15) = 0.7889.  This test pins the monotone curve.
"""

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit

N, LAYERS, SEED = 16, 12, 16012
BONDS = (15, 16, 17, 20, 24, 32)


def _brickwall(n, layers, seed):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0.05, 2.5, (layers + 1, n, 2))
    c = sf.Circuit(n)
    for d in range(layers):
        for q in range(n):
            c = c.rz(float(ang[d, q, 0]), q).ry(float(ang[d, q, 1]), q)
        for q in range(d % 2, n - 1, 2):
            c = c.cnot(q, q + 1)
    for q in range(n):
        c = c.rz(float(ang[layers, q, 0]), q).ry(float(ang[layers, q, 1]), q)
    return c


def test_fidelity_monotone_in_bond_dim():
    c = _brickwall(N, LAYERS, SEED)
    exact = np.asarray(c.to_ir().simulate_to_state("statevector").numpy(), dtype=complex)

    fid = {}
    for chi in BONDS:
        m = c.to_ir().evolve_mps(chi)
        sv = np.asarray(m.to_statevector(), dtype=complex)
        fid[chi] = abs(np.vdot(exact, sv)) ** 2 / float(np.vdot(sv, sv).real)
        assert m.max_discarded_weight() < 0.5, (
            f"bond {chi}: a single merge discarded {m.max_discarded_weight():.3f} "
            "of the weight; the row selection is losing live rows again"
        )

    for lo, hi in zip(BONDS, BONDS[1:]):
        assert fid[hi] >= fid[lo] - 1e-6, (
            f"fidelity fell from {fid[lo]:.6f} at bond {lo} to {fid[hi]:.6f} "
            f"at bond {hi}"
        )


def test_large_bond_is_exact():
    c = _brickwall(N, LAYERS, SEED)
    exact = np.asarray(c.to_ir().simulate_to_state("statevector").numpy(), dtype=complex)
    m = c.to_ir().evolve_mps(64)
    sv = np.asarray(m.to_statevector(), dtype=complex)
    assert float(np.vdot(sv, sv).real) == pytest.approx(1.0, abs=1e-9)
    assert abs(np.vdot(exact, sv)) ** 2 == pytest.approx(1.0, abs=1e-9)
