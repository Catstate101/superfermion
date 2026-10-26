"""Regression: MPS diagnostic contracts documented for 0.1.17.

Each assertion pins one clause that used to be misleading:
- discarded_weight is a sum, not a bounded fraction;
- fidelity_lower_bound is the clamped 1 - sum(eps);
- the densified vector is not renormalised, and mps_norm_sq reports that;
- truncation_events counts steps, not severity;
- mps_todense rejects an IR with a named TypeError.
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion.observables.core import mps_todense

pytestmark = pytest.mark.unit


def _bell_at_bond_one():
    c = sf.Circuit(2).h(0).cnot(0, 1)
    return c, c.to_ir().evolve_mps(1)


def test_discarded_weight_is_a_sum_and_lower_bound_clamps():
    _, m = _bell_at_bond_one()
    dw = float(m.discarded_weight())
    assert 0.0 < dw < 1.0
    assert float(m.fidelity_lower_bound()) == pytest.approx(max(0.0, min(1.0, 1.0 - dw)))
    assert float(m.max_discarded_weight()) <= dw + 1e-12


def test_statevector_is_unnormalised_after_truncation():
    c, m = _bell_at_bond_one()
    sv = np.asarray(m.to_statevector(), dtype=complex)
    norm_sq = float(np.vdot(sv, sv).real)
    assert norm_sq < 1.0
    assert c.to_ir().mps_norm_sq(1) == pytest.approx(norm_sq, abs=1e-9)


def test_truncation_events_counts_steps():
    _, m = _bell_at_bond_one()
    assert int(m.truncation_events()) >= 1
    assert float(m.max_discarded_weight()) <= 1.0


def test_mps_todense_rejects_ir_with_named_error():
    c = sf.Circuit(2).h(0).cnot(0, 1)
    with pytest.raises(TypeError, match="sf.Circuit"):
        mps_todense(c.to_ir())
