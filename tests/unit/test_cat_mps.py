"""cat: the MPS lane.

``sf.cat(circuit, mode="mps", bond_dim=chi)`` compiles the symbolic circuit
once and re-evolves the tensor network per call.  What has to be true:

* at a bond dimension large enough for the circuit it reproduces the exact
  statevector down to rounding -- it is the same evolution, not a lookalike;
* it is the *same* engine as ``dag.simulate_to_state("mps", ...)``, so the two
  entry points cannot drift apart;
* re-binding parameters in place works on a reused handle (the whole point of
  compiling once), and does not leak the previous values;
* truncation is reported, not absorbed: ``truncation_report()`` carries the
  discarded weight when ``chi`` is too small to be exact;
* a single tensor network is still one pure state, so the statevector lane's
  guards apply -- trailing readout and leading reset are fine, reset after use
  and feed-forward are refused, and the refusal names ``mps``;
* sampling goes through the returned state, and asking the handle for counts
  raises with that route instead of guessing parameter values.
"""

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit


def _ladder(n, layers=2):
    """Ry/Cx/rz ladder with *symbolic* angles, so rebinding can be exercised."""
    circuit = sf.Circuit(n)
    for k in range(layers):
        for q in range(n):
            circuit.ry(sf.param("t%d_%d" % (k, q)), q)
        for q in range(n - 1):
            circuit.cx(q, q + 1)
        for q in range(n):
            circuit.rz(sf.param("u%d_%d" % (k, q)), q)
    return circuit


def _sv(state):
    return np.asarray(state.numpy()).ravel()


def test_mps_matches_the_exact_statevector_at_large_bond_dim():
    circuit = _ladder(6)
    values = {name: 0.31 * (i + 1) for i, name in enumerate(circuit.parameters)}

    exact = sf.cat(circuit, mode="statevector").simulate(values)
    approx = sf.cat(circuit, mode="mps", bond_dim=64).simulate(values)

    assert np.max(np.abs(_sv(approx) - _sv(exact))) < 1e-12


def test_mps_is_the_same_engine_as_the_dag_entry_point():
    circuit = _ladder(5)
    values = {name: 0.23 for name in circuit.parameters}
    dag = circuit.to_ir().bind(values)

    via_dag = np.asarray(dag.simulate_to_state("mps", "cpu", 8).numpy()).ravel()
    via_cat = _sv(sf.cat(circuit, mode="mps", bond_dim=8).simulate(values))

    np.testing.assert_allclose(via_cat, via_dag, atol=1e-15, rtol=0)


def test_handle_rebinds_parameters_in_place():
    circuit = _ladder(4, layers=1)
    handle = sf.cat(circuit, mode="mps", bond_dim=32)
    reference = sf.cat(circuit, mode="statevector")

    first = {name: 0.17 for name in handle.parameter_names}
    second = {name: -0.93 for name in handle.parameter_names}

    a = _sv(handle.simulate(first))
    b = _sv(handle.simulate(second))

    np.testing.assert_allclose(a, _sv(reference.simulate(first)), atol=1e-12)
    np.testing.assert_allclose(b, _sv(reference.simulate(second)), atol=1e-12)
    assert np.max(np.abs(a - b)) > 1e-3, "the second bind reused the first state"


def test_truncation_is_reported_not_absorbed():
    circuit = sf.Circuit(4)
    circuit.h(0)
    for q in range(1, 4):
        circuit.cx(0, q)

    state = sf.cat(circuit, mode="mps", bond_dim=1).simulate({})
    report = state.truncation_report()
    norm2 = float(np.vdot(_sv(state), _sv(state)).real)

    assert report is not None, "bond_dim=1 on a GHZ cut must truncate"
    assert report["truncation_events"] > 0
    assert report["discarded_weight"] > 0.1
    # The lane keeps the surviving weight rather than renormalizing, so the
    # report accounts for exactly the norm the state is missing.
    assert norm2 == pytest.approx(1.0 - report["discarded_weight"], abs=1e-12)
    assert report["fidelity_lower_bound"] == pytest.approx(norm2, abs=1e-12)


def test_handle_refuses_to_sample_and_names_the_route():
    handle = sf.cat(_ladder(3), mode="mps", bond_dim=8)

    with pytest.raises(ValueError, match="simulate"):
        handle.sample(16)


def test_state_samples_after_binding():
    circuit = sf.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    counts = sf.cat(circuit, mode="mps", bond_dim=8).simulate({}).sample(200, seed=7)

    assert set(counts) <= {"00", "11"}


def test_trailing_measurement_and_leading_reset_are_accepted():
    circuit = sf.Circuit(3)
    circuit.reset(0)
    circuit.ry(0.4, 0)
    circuit.cx(0, 1)
    circuit.measure(1, 0)

    values = {name: 0.5 for name in circuit.parameters}
    exact = sf.cat(circuit, mode="statevector").simulate(values)
    mps = sf.cat(circuit, mode="mps", bond_dim=16).simulate(values)

    assert np.max(np.abs(_sv(mps) - _sv(exact))) < 1e-12


def test_reset_after_use_is_refused_with_the_mps_lane_named():
    circuit = sf.Circuit(2)
    circuit.h(0)
    circuit.reset(0)

    with pytest.raises(ValueError, match="mps plan"):
        sf.cat(circuit, mode="mps")


def test_dynamic_constructs_are_refused():
    circuit = sf.Circuit(2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.cx(0, 1)

    with pytest.raises(ValueError, match="mps"):
        sf.cat(circuit, mode="mps")


def test_unbound_parameters_raise_instead_of_panicking():
    handle = sf.cat(_ladder(3), mode="mps", bond_dim=8)

    with pytest.raises(ValueError, match="unbound parameter"):
        handle.simulate({})

    partial = {name: 0.1 for name in handle.parameter_names[:-1]}
    with pytest.raises(ValueError, match="unbound parameter"):
        handle.simulate(partial)


def test_bond_dim_zero_is_rejected():
    with pytest.raises(ValueError, match="bond_dim"):
        sf.cat(sf.Circuit(2), mode="mps", bond_dim=0)


def test_unknown_mode_lists_mps_as_an_option():
    with pytest.raises(ValueError, match="mps"):
        sf.cat(sf.Circuit(2), mode="tensor")


def test_handle_reports_its_mode_and_bond_dim():
    circuit = sf.Circuit(3)
    circuit.h(0)
    handle = sf.cat(circuit, mode="mps", bond_dim=12)

    assert handle.mode == "mps"
    assert handle.bond_dim == 12
    assert handle.n_qubits == 3

    state = handle.simulate({})
    assert state.method == "mps"
    assert state.device == "cat"


def test_other_lanes_are_unchanged_by_the_extra_constructor_argument():
    circuit = sf.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    assert sf.cat(circuit).mode == "statevector"
    assert sf.cat(circuit, mode="dm").simulate({}).method == "density_matrix"
