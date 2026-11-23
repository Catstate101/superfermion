"""cat: the gradient on the ``mps`` lane.

Before this entry point existed, ``sf.cat(qc, mode="mps").run_gradient(...)``
raised ``cat: plan is not an adjoint plan``: the tensor-network lane had no
gradient on the handle at all, so "cat adjoint" and the MPS lane were different
objects.  ``State.grad`` on an MPS state did work, which is what makes the
claim below checkable -- the handle now answers with the same shift rule.

What has to be true:

* at a bond large enough to be exact, the ``mps`` handle's gradient equals the
  dense ``adjoint`` lane's reverse sweep (they answer the same question by
  different routes);
* it equals ``State.grad`` on the handle's own state *exactly* -- same engine,
  so any difference is a wiring bug, not rounding;
* ``run_gradient_terms`` sums the terms, matching the adjoint lane;
* at a truncating bond it follows the forward the handle reports (finite
  differences of that same chain) rather than the dense lane, because the
  two are different functionals;
* a wrong ``theta`` length is refused by count, and a lane with no gradient
  still refuses to pretend.
"""

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit


def _ladder(n, layers=2):
    circuit = sf.Circuit(n)
    for k in range(layers):
        for q in range(n):
            circuit.ry(sf.param("t%d_%d" % (k, q)), q)
        for q in range(n - 1):
            circuit.cx(q, q + 1)
        for q in range(n):
            circuit.rz(sf.param("u%d_%d" % (k, q)), q)
    return circuit


def _pauli(n, spec):
    """``spec`` maps qubit -> "X"/"Y"/"Z"; identity elsewhere."""
    letters = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    labels = [0] * n
    for q, p in spec.items():
        labels[q] = letters[p]
    return [(labels, 1.0, 0.0)]


def _label(n, spec):
    chars = ["I"] * n
    for q, p in spec.items():
        chars[q] = p
    return "".join(chars)


N = 5
SPEC = {0: "Z", 2: "X", 4: "Z"}
OBS = _label(N, SPEC)
# Two layers of Ry/CNOT/Rz on N qubits: 4*N parameters, in circuit order (the
# same order `parameter_names()` reports on every lane).
THETA = [0.11 * (i + 1) - 0.7 for i in range(4 * N)]


def _grad_dict(state, dag, values):
    return state.grad(_pauli(N, SPEC), dag, values)


def test_mps_gradient_matches_the_dense_adjoint_lane():
    circuit = _ladder(N)
    dense = np.asarray(
        sf.cat(circuit, mode="adjoint").run_gradient(OBS, THETA)
    )
    chain = np.asarray(
        sf.cat(circuit, mode="mps", bond_dim=64).run_gradient(OBS, THETA)
    )

    assert dense.shape == chain.shape
    assert np.max(np.abs(dense)) > 1e-3, "a flat gradient would hide a zero bug"
    np.testing.assert_allclose(chain, dense, atol=1e-9, rtol=0)


def test_mps_gradient_is_the_same_shift_rule_state_grad_runs():
    circuit = _ladder(N)
    names = list(circuit.parameters)
    values = dict(zip(names, THETA))

    handle = sf.cat(circuit, mode="mps", bond_dim=64)
    got = np.asarray(handle.run_gradient(OBS, THETA))

    state = handle.simulate(values)
    shift = _grad_dict(state, circuit.to_ir(), values)
    want = np.asarray([shift[name] for name in names])

    np.testing.assert_array_equal(got, want)


def test_run_gradient_terms_sums_on_the_mps_lane():
    circuit = _ladder(N)
    second = _label(N, {1: "Y", 3: "Y"})
    terms = [(OBS, 0.7), (second, -1.3)]

    chain = np.asarray(
        sf.cat(circuit, mode="mps", bond_dim=64).run_gradient_terms(terms, THETA)
    )
    dense = np.asarray(
        sf.cat(circuit, mode="adjoint").run_gradient_terms(terms, THETA)
    )

    np.testing.assert_allclose(chain, dense, atol=1e-9, rtol=0)


def test_truncated_gradient_follows_the_forward_the_handle_reports():
    circuit = _ladder(N)
    names = list(circuit.parameters)
    values = dict(zip(names, THETA))
    handle = sf.cat(circuit, mode="mps", bond_dim=1)

    state = handle.simulate(values)
    vec = np.asarray(state.numpy()).ravel()
    assert 1.0 - float(np.vdot(vec, vec).real) > 0.1, "bond 1 must truncate here"

    got = np.asarray(handle.run_gradient(OBS, THETA))

    h = 1e-5
    fd = np.empty(len(names))
    for i, name in enumerate(names):
        up = dict(values)
        down = dict(values)
        up[name] += h
        down[name] -= h
        fd[i] = (
            handle.simulate(up).expectation(_pauli(N, SPEC))
            - handle.simulate(down).expectation(_pauli(N, SPEC))
        ) / (2 * h)

    np.testing.assert_allclose(got, fd, atol=1e-6, rtol=0)
    # ... and the dense lane is a *different* answer here, by design.
    dense = np.asarray(sf.cat(circuit, mode="adjoint").run_gradient(OBS, THETA))
    assert np.max(np.abs(got - dense)) > 1e-2


def test_gradient_is_unchanged_by_a_simulate_call_in_between():
    circuit = _ladder(N)
    values = dict(zip(circuit.parameters, THETA))
    handle = sf.cat(circuit, mode="mps", bond_dim=64)

    first = np.asarray(handle.run_gradient(OBS, THETA))
    handle.simulate(values)
    again = np.asarray(handle.run_gradient(OBS, THETA))

    np.testing.assert_array_equal(first, again)


def test_wrong_theta_length_is_named_by_count():
    handle = sf.cat(_ladder(N), mode="mps", bond_dim=8)
    with pytest.raises(ValueError, match=r"4 values but the plan has 20"):
        handle.run_gradient(OBS, THETA[:4])


def test_lanes_without_a_gradient_still_refuse():
    circuit = _ladder(N)
    for mode in ("statevector", "density_matrix"):
        with pytest.raises(ValueError, match="adjoint plan"):
            sf.cat(circuit, mode=mode).run_gradient(OBS, THETA)


def test_mps_gradient_rejects_an_empty_term_list():
    handle = sf.cat(_ladder(3), mode="mps", bond_dim=8)
    with pytest.raises(ValueError, match="at least one"):
        handle.run_gradient_terms([], [0.1] * 12)
