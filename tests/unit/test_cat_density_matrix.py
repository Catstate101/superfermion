"""cat: the density-matrix lane.

Every other ``cat`` lane carries one pure vector or one tableau, so it has to
*refuse* the shapes that produce a mixed state — a ``Reset`` after use and a
mid-circuit readout.  ``mode="density_matrix"`` carries `rho` instead, so those
shapes are answered exactly:

* ``reset`` is the channel ``rho -> Tr_q(rho) (x) |0><0|``;
* a readout with no later conditioned gate is dephasing;
* feed-forward is still refused — one `rho` cannot be a mixture over branches.

These tests pin the state itself (diagonal, purity, entropy), the bit-exact
agreement with the pre-existing ``sf.run(method="density_matrix")`` path, the
Qiskit reference value for the channel, and the one refusal that survives.
"""

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit


def _ghz(n=2):
    circuit = sf.Circuit(n)
    circuit.h(0)
    for q in range(1, n):
        circuit.cx(0, q)
    return circuit


def _rho(state):
    """Flat engine buffer (``data[ket + bra*dim]``) -> ``(dim, dim)`` matrix."""
    flat = np.asarray(state.numpy())
    dim = int(round(np.sqrt(flat.size)))
    return flat.reshape((dim, dim), order="F")


def test_dm_mode_returns_a_density_matrix_state():
    state = sf.cat(_ghz(2), mode="density_matrix").simulate({})
    assert state.method == "density_matrix"
    assert state.device == "cat"
    assert state.n_qubits == 2
    assert state.shape == [4, 4]
    assert _rho(state).shape == (4, 4)


def test_both_mode_spellings_agree():
    circuit = _ghz(2)
    a = _rho(sf.cat(circuit, mode="dm").simulate({}))
    b = _rho(sf.cat(circuit, mode="density_matrix").simulate({}))
    np.testing.assert_array_equal(a, b)


def test_reset_after_use_is_the_exact_mixed_state():
    """``ghz; reset(0)`` is ``0.5|00><00| + 0.5|10><10|`` (label ``|q1 q0>``).

    Nothing pure can hold this: the statevector lane refuses the circuit and
    Qiskit's pure-state path post-selects instead of returning the channel.
    """
    circuit = _ghz(2)
    circuit.reset(0)
    state = sf.cat(circuit, mode="density_matrix").simulate({})
    diag = np.real(np.diag(_rho(state)))
    np.testing.assert_allclose(diag, [0.5, 0.0, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        state.probabilities(), [0.5, 0.0, 0.5, 0.0], atol=1e-12
    )
    assert state.purity() == pytest.approx(0.5, abs=1e-12)
    assert state.entropy() == pytest.approx(np.log(2.0), abs=1e-12)
    # <Z0> = +1 (re-prepared in |0>), <Z1> = 0 (still uniform), <ZZ> = 0.
    assert state.expectation([([3, 0], 1.0, 0.0)]) == pytest.approx(1.0, abs=1e-12)
    assert state.expectation([([0, 3], 1.0, 0.0)]) == pytest.approx(0.0, abs=1e-12)
    assert state.expectation([([3, 3], 1.0, 0.0)]) == pytest.approx(0.0, abs=1e-12)


def test_without_the_reset_the_same_circuit_stays_pure():
    """The channel is the only difference, so purity must stay 1 without it."""
    state = sf.cat(_ghz(2), mode="density_matrix").simulate({})
    rho = _rho(state)
    assert state.purity() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        np.real(np.diag(rho)), [0.5, 0.0, 0.0, 0.5], atol=1e-12
    )
    assert abs(rho[3, 0]) == pytest.approx(0.5, abs=1e-12)


def test_reset_of_a_basis_state_is_deterministic():
    circuit = sf.Circuit(2)
    circuit.x(0)
    circuit.reset(0)
    state = sf.cat(circuit, mode="density_matrix").simulate({})
    np.testing.assert_allclose(
        np.real(np.diag(_rho(state))), [1.0, 0.0, 0.0, 0.0], atol=1e-12
    )
    assert state.purity() == pytest.approx(1.0, abs=1e-12)


def test_leading_reset_is_the_identity_channel():
    """``reset`` on a qubit nothing touched yet cannot move the state."""
    bare = _ghz(2)
    with_reset = sf.Circuit(2)
    with_reset.reset(0)
    with_reset.h(0)
    with_reset.cx(0, 1)
    a = _rho(sf.cat(bare, mode="density_matrix").simulate({}))
    b = _rho(sf.cat(with_reset, mode="density_matrix").simulate({}))
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_mid_circuit_readout_dephases_without_moving_populations():
    circuit = sf.Circuit(1, n_cbits=1)
    circuit.h(0)
    circuit.measure(0, 0)
    rho = _rho(sf.cat(circuit, mode="density_matrix").simulate({}))
    np.testing.assert_allclose(np.real(np.diag(rho)), [0.5, 0.5], atol=1e-12)
    assert abs(rho[0, 1]) < 1e-12


def test_matches_sf_run_density_matrix_exactly():
    """The lane reuses the vetted DM kernels, so the buffers must be identical."""
    circuit = _ghz(3)
    for q in range(3):
        circuit.ry(0.31 * (q + 1), q)
    circuit.cz(0, 2)
    cat_rho = _rho(sf.cat(circuit, mode="density_matrix").simulate({}))
    run_rho = _rho(sf.run(circuit, method="density_matrix").state)
    np.testing.assert_array_equal(cat_rho, run_rho)


def test_parameter_rebinding_through_one_handle():
    """``ry(t); cx; reset(1)`` leaves q0 in ``cos^2(t/2)|0> + sin^2(t/2)|1>``."""
    t = sf.param("t")
    circuit = sf.Circuit(2)
    circuit.ry(t, 0)
    circuit.cx(0, 1)
    circuit.reset(1)
    handle = sf.cat(circuit, mode="density_matrix")
    assert list(handle.parameter_names) == ["t"]
    for value in (0.0, 0.7, np.pi / 2):
        diag = np.real(np.diag(_rho(handle.simulate({"t": value}))))
        np.testing.assert_allclose(
            diag[:2],
            [np.cos(value / 2) ** 2, np.sin(value / 2) ** 2],
            atol=1e-12,
        )
        np.testing.assert_allclose(diag[2:], [0.0, 0.0], atol=1e-12)


def test_feed_forward_is_still_refused():
    """A conditioned gate makes the state a mixture over branches, not one rho."""
    circuit = sf.Circuit(2, n_cbits=1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.c_if(0, 1).x(1)
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode="density_matrix")
    message = str(excinfo.value)
    assert "conditioned" in message
    assert "simulate_dynamic" in message


def test_a_three_qubit_gate_matches_the_statevector_lane():
    """Toffoli is wider than the shared one-/two-qubit DM kernel.

    ``DensityMatrixState::apply_unitary`` panics above two qubits (so
    ``sf.run(..., method="density_matrix")`` still does), and the lane applies
    the gate axis-by-axis instead of refusing it.
    """
    circuit = sf.Circuit(3)
    circuit.h(0)
    circuit.h(1)
    circuit.ccx(0, 1, 2)
    rho = _rho(sf.cat(circuit, mode="density_matrix").simulate({}))
    psi = sf.cat(circuit, mode="statevector").simulate({}).numpy()
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-12)


def test_matches_qiskit_density_matrix_on_the_reset_case():
    """Qiskit's ``DensityMatrix`` is the reference for the reset channel."""
    qiskit = pytest.importorskip("qiskit")
    quantum_info = pytest.importorskip("qiskit.quantum_info")

    circuit = _ghz(2)
    circuit.reset(0)

    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.reset(0)
    expected = np.asarray(quantum_info.DensityMatrix.from_instruction(qc).data)

    got = _rho(sf.cat(circuit, mode="density_matrix").simulate({}))
    np.testing.assert_allclose(got, expected, atol=1e-12)
