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


class TestRustStorageBinding:
    """Regression: bind() must substitute symbols on use_rust_storage circuits.

    Previously the Rust GateSequence stored a placeholder 0.0 for symbolic
    params and kept no names, so bind() produced an all-zero circuit while
    still reporting n_parameters == 0.
    """

    def _circuit(self, rust):
        theta, phi = param("theta"), param("phi")
        circuit = sf.Circuit(2, use_rust_storage=rust)
        circuit.ry(theta, 0)
        circuit.ry(phi, 1)
        return circuit

    def test_rust_bind_matches_default_storage(self):
        import numpy as np

        values = {"theta": 0.5, "phi": -1.2}
        default = np.asarray(sf.simulate(self._circuit(False).bind(values)).numpy())
        rust = np.asarray(sf.simulate(self._circuit(True).bind(values)).numpy())
        np.testing.assert_allclose(rust, default)

    def test_rust_bind_records_values(self):
        bound = self._circuit(True).bind({"theta": 0.5, "phi": -1.2})
        params = [p for g in bound.to_gate_list() for p in g.get("params", [])]
        assert params == [0.5, -1.2]

    def test_rust_partial_then_chained_bind(self):
        import numpy as np

        default = self._circuit(False).bind({"theta": 0.5, "phi": -1.2})
        full = np.asarray(sf.simulate(default).numpy())
        partial = self._circuit(True).bind({"theta": 0.5})
        assert partial.n_parameters == 1
        chained = partial.bind({"phi": -1.2})
        np.testing.assert_allclose(np.asarray(sf.simulate(chained).numpy()), full)
