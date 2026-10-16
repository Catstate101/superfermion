"""Numerical and ownership tests for the opt-in performance API."""

import itertools

import numpy as np
import pytest

import superfermion as sf
from superfermion.kernels import angle_fidelity_kernel


def dense_pauli(paulis):
    matrices = [np.eye(2), np.array([[0, 1], [1, 0]]),
                np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    result = np.ones((1, 1), dtype=complex)
    for code in reversed(paulis):
        result = np.kron(result, matrices[code])
    return result


def test_performance_prepared_matches_bound_circuits_and_snapshot():
    x = sf.param("x")
    circuit = sf.Circuit(3).ry(x, 0).cx(0, 2).rz(x, 2).h(1)
    prepared = sf.prepare(circuit)
    rows = [{"x": v} for v in [0.1, -0.8, 2.1, 0.1]]
    actual = prepared.states_batch(rows)
    expected = np.stack([circuit.bind(row).to_ir().simulate() for row in rows])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    circuit.x(1)
    np.testing.assert_allclose(prepared.states_batch(rows), expected, atol=1e-12)
    actual[0, 0] = 100
    np.testing.assert_allclose(prepared.simulate(rows[0]).numpy(), expected[0], atol=1e-12)
    assert prepared.states_batch([]).shape == (0, 8)


def test_performance_native_readouts_and_all_three_qubit_paulis():
    rng = np.random.default_rng(73)
    vector = rng.normal(size=8) + 1j * rng.normal(size=8)
    vector /= np.linalg.norm(vector)
    state = sf.State.from_numpy(vector, 3)
    for paulis in itertools.product(range(4), repeat=3):
        expected = np.vdot(vector, dense_pauli(paulis) @ vector).real
        actual = state.expectation([(list(paulis), 1.0, 0.0)])
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    weights = np.array([(3 - 2 * i.bit_count()) / 3 for i in range(8)])
    expected = np.dot(np.abs(vector)**2, weights)
    observable = sf.DiagonalObservable(weights.tolist())
    np.testing.assert_allclose(state.mean_z(), expected, atol=1e-12)
    np.testing.assert_allclose(state.diagonal_expectation(observable), expected, atol=1e-12)
    np.testing.assert_allclose(state.norm_squared(), 1, atol=1e-12)


def test_performance_batch_readouts_and_errors():
    circuit = sf.Circuit(2).ry(sf.param("x"), 0).cx(0, 1)
    prepared = sf.prepare(circuit)
    rows = [{"x": x} for x in [0.0, 0.5, 1.7]]
    vectors = prepared.states_batch(rows)
    weights = np.array([1., 0., 0., -1.])
    expected = np.abs(vectors)**2 @ weights
    np.testing.assert_allclose(prepared.mean_z_batch(rows), expected, atol=1e-12)
    np.testing.assert_allclose(
        prepared.diagonal_batch(rows, sf.DiagonalObservable(weights.tolist())),
        expected, atol=1e-12,
    )
    for invalid in [{}, {"x": float("nan")}, {"x": 0.1, "extra": 2}]:
        with pytest.raises(ValueError):
            prepared.simulate(invalid)
    with pytest.raises(ValueError):
        prepared.diagonal_batch(rows, sf.DiagonalObservable([0., 1.]))
    with pytest.raises(ValueError):
        sf.DiagonalObservable([1., 2., 3.])
    with pytest.raises(ValueError):
        sf.prepare(sf.Circuit(1).h(0).measure(0))


def test_performance_angle_kernel_matches_statevector_overlap():
    rng = np.random.default_rng(21)
    for n in [1, 3, 5]:
        x = rng.normal(size=(5, n))
        y = rng.normal(size=(3, n))
        parameters = [sf.param(f"x{i}") for i in range(n)]
        circuit = sf.Circuit(n)
        for i, parameter in enumerate(parameters):
            circuit.ry(parameter, i)
        for i in range(n-1):
            circuit.cx(i, i+1)
        prepared = sf.prepare(circuit)
        def encode(a):
            return prepared.states_batch([
                {f"x{i}": float(value) for i, value in enumerate(row)} for row in a
            ])
        sx, sy = encode(x), encode(y)
        expected = np.abs(sx.conj() @ sy.T)**2
        np.testing.assert_allclose(angle_fidelity_kernel(x, y, block_rows=2), expected, atol=1e-12)


def test_performance_run_can_return_state_without_numpy_export():
    result = sf.run(sf.Circuit(2).h(0).cx(0, 1), device="cpu", shots=0,
                    method="statevector", return_statevector=False)
    assert result.statevector is None
    assert result.state is not None
    np.testing.assert_allclose(result.state.mean_z(), 0.0, atol=1e-12)
    assert set(result.probabilities) == {"00", "11"}


def test_performance_adjoint_matches_finite_difference():
    circuit = sf.Circuit(2).ry(sf.param("x"), 0).cx(0, 1).rz(sf.param("z"), 1)
    dag = circuit.to_ir()
    observable = [([1, 1], 0.7, 0.0), ([3, 0], -0.2, 0.0)]
    values = {"x": 0.37, "z": -0.81}
    gradient = dag.adjoint_grad(observable, values)
    for name in values:
        plus, minus = dict(values), dict(values)
        plus[name] += 1e-6
        minus[name] -= 1e-6
        expected = (sf.prepare(circuit).simulate(plus).expectation(observable)
                    - sf.prepare(circuit).simulate(minus).expectation(observable)) / 2e-6
        np.testing.assert_allclose(gradient[name], expected, atol=1e-7, rtol=1e-6)
