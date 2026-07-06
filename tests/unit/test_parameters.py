"""Unit tests for symbolic parameters."""

import pytest

import superfermion as sf
from superfermion.parameters import SymbolicParameter, param


pytestmark = pytest.mark.unit


class TestSymbolicParameter:
    def test_param_creates_symbolic_parameter(self):
        theta = param("theta")
        assert isinstance(theta, SymbolicParameter)
        assert theta.name == "theta"

    def test_sf_param_alias(self):
        phi = sf.param("phi")
        assert isinstance(phi, SymbolicParameter)
        assert phi.name == "phi"

    def test_binding_substitutes_in_circuit(self):
        theta = param("theta")
        circuit = sf.Circuit(1).rx(theta, 0)
        bound = circuit.bind({"theta": 1.57})
        gate = bound.to_gate_list()[0]
        assert gate["params"] == [1.57]
        assert bound.n_parameters == 0

    def test_equality_by_name(self):
        a = param("theta")
        b = param("theta")
        c = param("phi")
        assert a == b
        assert a != c

    def test_hashable_by_name(self):
        a = param("theta")
        b = param("theta")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_multiple_parameters_in_one_circuit(self, parametric_circuit):
        assert parametric_circuit.n_parameters == 2
        assert set(parametric_circuit.parameters) == {"theta", "phi"}
        bound = parametric_circuit.bind({"theta": 0.5, "phi": 1.0})
        params = [p for g in bound.to_gate_list() for p in g.get("params", [])]
        assert params == [0.5, 1.0]
