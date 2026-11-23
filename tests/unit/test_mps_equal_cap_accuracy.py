"""Regression: equal-bond accuracy must behave like a gauged MPS, not a greedy one.

Guards the mixed-canonical gauge added for report §39.  A two-site SVD is the
Schmidt truncation of *its own cut* only when every site below the pair is
left-isometric and every site above is right-isometric.  Without that gauge each
merge truncated a local matrix instead, so the discarded weight was not the
cut's tail and the error compounded gate by gate.  Measured on the fixture below
(n=16, 12 cycles, chi=128): greedy 8.2500e-02 against gauged 9.8836e-03, an
8.35x gap, while ``discarded_weight`` over-reported the loss by 39.1%
(1.1475e-01 against an achieved 8.2500e-02).  Those are the report's own
numbers: this fixture is the benchmark circuit, and ``to_ir().evolve_mps(128)``
lands on the same 9.88e-03 as ``sf.cat(mode="mps", bond_dim=128)`` does, so the
assertions below pin both entry points.

Two assertions, because the gauge has two separately observable consequences:

* the accuracy itself -- ``test_equal_cap_accuracy``;
* the telemetry -- ``discarded_weight`` only equals the achieved loss while every
  merge's eps is a Schmidt tail, so it is a fidelity bound now and was not
  before (§39.7).

The cap does not bind below n=16 on this shape: at n=12, chi=128 the evolution
is exact (1-fid 0.0e+00, discarded 2.2e-14) with and without the gauge, so a
smaller fixture cannot detect the gauge's removal.  Run this module directly to
print the fixture's calibration under whatever ``SF_MPS_GAUGE`` is set.
"""

import functools

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit

# The benchmark's own seed, so this fixture is the circuit §39 calibrated: its
# chi=128 loss is 9.88e-03 gauged and 8.25e-02 greedy.  A different seed gives a
# different Schmidt tail (seed 16012, used by the monotonicity test, yields
# 5.39e-03) and the numbers below would no longer be the reported ones.
N, LAYERS, SEED = 16, 12, 20261119
CHI = 128

# Set from the calibration below, not chosen in advance: gauged 9.88e-03 and
# greedy 8.25e-02 leave a 5x margin either side of this line.
MAX_LOSS = 2.0e-2
TELEMETRY_RTOL = 0.25


def _brick_circuit(n, cycles, seed=20261119):
    """RY,RZ on every wire, CX on even bonds, RY,RZ, CX on odd bonds.

    Deliberately the benchmark's own generator and draw order, not the lighter
    brickwork of ``test_mps_bond_monotonicity``: that one uses a single CNOT
    ladder per layer and is *exact* at chi=128 on n=16 (1-fid 1.3e-15), so it
    cannot detect a missing gauge at all.  This generator has twice the
    entangling density and a real Schmidt tail -- 8.6e-03 at chi=128 across the
    middle cut.  The RNG is consumed in the same order as the generator so the
    angles, and therefore the calibration below, are the benchmark's.
    """
    rng = np.random.default_rng(seed)
    c = sf.Circuit(n)
    for _ in range(cycles):
        for q in range(n):
            c = c.ry(float(rng.uniform(-np.pi, np.pi)), q)
            c = c.rz(float(rng.uniform(-np.pi, np.pi)), q)
        for q in range(0, n - 1, 2):
            c = c.cx(q, q + 1)
        for q in range(n):
            c = c.ry(float(rng.uniform(-np.pi, np.pi)), q)
            c = c.rz(float(rng.uniform(-np.pi, np.pi)), q)
        for q in range(1, n - 1, 2):
            c = c.cx(q, q + 1)
    return c


@functools.lru_cache(maxsize=1)
def _fixture():
    """(1 - fidelity, discarded_weight) at a cap that binds.

    The state is not renormalised after truncation, so ``<sv|sv>`` is the
    surviving weight and ``1 - |<exact|sv>|^2 / <sv|sv>`` is the loss against
    the exact state.  Both are taken from one evolution, cached because two
    tests read it and the n=16 evolution is the expensive part.
    """
    c = _brick_circuit(N, LAYERS, SEED)
    exact = np.asarray(c.to_ir().simulate_to_state("statevector").numpy(), dtype=complex)
    m = c.to_ir().evolve_mps(CHI)
    sv = np.asarray(m.to_statevector(), dtype=complex)
    surv = float(np.vdot(sv, sv).real)
    fid = abs(np.vdot(exact, sv)) ** 2 / surv
    loss = 1.0 - fid
    # A fixture that stopped binding would make both assertions pass vacuously,
    # which is worse than failing: it would report "gauge fine" forever.
    assert loss > 1e-6, (
        f"the chi={CHI} cap no longer binds on this fixture (1-fid {loss:.3e}); "
        "the calibration is void -- pick a smaller cap before trusting a pass"
    )
    return loss, float(m.discarded_weight())


def test_equal_cap_accuracy():
    loss, _ = _fixture()
    assert loss < MAX_LOSS, (
        f"1-fid at bond {CHI} is {loss:.3e}, above the {MAX_LOSS:.1e} line. "
        "A greedy per-merge truncation lands near 8.3e-2 here because the "
        "prefix is not left-isometric; check that the gauge in "
        "MPSState::apply_2q_gate still runs for a truncating merge"
    )


def test_discarded_weight_is_the_achieved_loss():
    loss, discard = _fixture()
    rel = abs(discard - loss) / loss
    assert rel < TELEMETRY_RTOL, (
        f"discarded_weight {discard:.3e} is {rel:.0%} away from the measured "
        f"loss {loss:.3e}. Above ~39% is the signature of a merge whose eps is "
        "a local truncation rather than the tail of the cut's Schmidt spectrum"
    )


if __name__ == "__main__":
    _loss, _discard = _fixture()
    print("SF_MPS_GAUGE=%r" % __import__("os").environ.get("SF_MPS_GAUGE"))
    print("1-fid        %.4e" % _loss)
    print("discard      %.4e" % _discard)
    print("relative gap %.1f%%" % (100.0 * abs(_discard - _loss) / _loss))
