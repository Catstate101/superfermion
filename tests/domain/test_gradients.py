"""Quantum gradient domain tests."""

import math

import numpy as np
import pytest

import superfermion as sf
from superfermion.observables.core import SparsePauliOp


pytestmark = pytest.mark.domain

parameter_shift = pytest.importorskip(
    "superfermion.qml.gradient.parameter_shift",
    reason="parameter-shift gradient module unavailable",
)
adjoint = pytest.importorskip(
    "superfermion.qml.gradient.adjoint",
    reason="adjoint gradient module unavailable",
)


class TestParameterShiftGradient:
    def test_basic_gradient_on_ry_circuit(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})
        params = {"theta": 0.5}

        grad = parameter_shift.parameter_shift_grad(
            circuit, observable, params, backend="statevector"
        )
        assert "theta" in grad
        expected = -math.sin(0.5)
        assert abs(grad["theta"] - expected) < 1e-5

    def test_gradient_vector_shape(self):
        theta = sf.param("theta")
        phi = sf.param("phi")
        circuit = sf.Circuit(2).ry(theta, 0).ry(phi, 1).cnot(0, 1)
        observable = SparsePauliOp.from_dict({"ZZ": 1.0})
        names = ["theta", "phi"]
        values = np.array([0.3, 0.7])

        grad_vec = parameter_shift.parameter_shift_grad_vector(
            circuit, observable, names, values, backend="statevector"
        )
        assert grad_vec.shape == (2,)
        assert np.all(np.isfinite(grad_vec))


class TestAdjointGradient:
    def test_adjoint_matches_parameter_shift(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})
        names = ["theta"]
        values = np.array([0.5])

        ps_grad = parameter_shift.parameter_shift_grad_vector(
            circuit, observable, names, values, backend="statevector"
        )
        adj_grad = adjoint.adjoint_grad_vector(
            circuit, observable, names, values
        )
        assert adj_grad.shape == (1,)
        assert abs(ps_grad[0] - adj_grad[0]) < 1e-4

    def test_adjoint_grad_dict_form(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).rx(theta, 0)
        observable = SparsePauliOp.from_dict({"X": 1.0})
        params = {"theta": math.pi / 4}

        grad = adjoint.adjoint_grad(circuit, observable, params)
        assert "theta" in grad
        assert np.isfinite(grad["theta"])
