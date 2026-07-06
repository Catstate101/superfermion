"""Unit tests for the Circuit fluent API."""

import json
from unittest.mock import patch

import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.parameters import param


pytestmark = pytest.mark.unit


class TestCircuitConstruction:
    def test_default_qubit_and_cbit_counts(self):
        c = Circuit(3)
        assert c.n_qubits == 3
        assert c.n_cbits == 3

    def test_custom_cbit_count(self):
        c = Circuit(2, n_cbits=4)
        assert c.n_cbits == 4

    def test_optional_name(self):
        c = Circuit(1, name="bell")
        assert "name='bell'" in repr(c)

    def test_requires_at_least_one_qubit(self):
        with pytest.raises(ValueError, match="at least 1 qubit"):
            Circuit(0)


class TestFluentGateAPI:
    def test_single_qubit_gates(self):
        c = Circuit(1).h(0).x(0).y(0).z(0).s(0).t(0).id(0)
        gates = c.to_gate_list()
        names = [g["name"] for g in gates]
        assert names == ["H", "X", "Y", "Z", "S", "T", "ID"]

    def test_parameterized_rotations(self):
        c = Circuit(1).rx(0.5, 0).ry(1.0, 0).rz(1.5, 0)
        gates = c.to_gate_list()
        assert gates[0]["name"] == "RX"
        assert gates[0]["params"] == [0.5]

    def test_two_qubit_gates(self):
        c = Circuit(2).cnot(0, 1).cz(0, 1).swap(0, 1)
        gates = c.to_gate_list()
        assert [g["name"] for g in gates] == ["CNOT", "CZ", "SWAP"]

    def test_three_qubit_gates(self):
        c = Circuit(3).ccx(0, 1, 2).cswap(0, 1, 2)
        gates = c.to_gate_list()
        assert [g["name"] for g in gates] == ["CCX", "CSWAP"]

    def test_cx_is_alias_for_cnot(self, bell_circuit):
        assert bell_circuit.to_gate_list()[1]["name"] == "CNOT"

    def test_invalid_qubit_raises(self):
        with pytest.raises(IndexError):
            Circuit(2).h(5)

    def test_duplicate_qubits_in_gate_raises(self):
        with pytest.raises(ValueError, match="Duplicate qubit"):
            Circuit(2).cnot(0, 0)


class TestParameterizedCircuits:
    def test_param_registration(self, parametric_circuit):
        assert parametric_circuit.n_parameters == 2
        assert set(parametric_circuit.parameters) == {"theta", "phi"}

    def test_bind_substitutes_values(self, parametric_circuit):
        bound = parametric_circuit.bind({"theta": 0.1, "phi": 0.2})
        assert bound.n_parameters == 0
        params = [p for g in bound.to_gate_list() for p in g.get("params", [])]
        assert params == [0.1, 0.2]

    def test_partial_bind_leaves_unbound(self, parametric_circuit):
        bound = parametric_circuit.bind({"theta": 0.1})
        assert bound.n_parameters == 1
        assert bound.parameters == ["phi"]

    def test_build_raises_on_unbound(self, parametric_circuit):
        with pytest.raises(RuntimeError, match="unbound parameter"):
            parametric_circuit.build()

    def test_build_returns_self_when_bound(self, parametric_circuit):
        bound = parametric_circuit.bind({"theta": 0.1, "phi": 0.2})
        assert bound.build() is bound


class TestCircuitProperties:
    def test_depth_bell_state(self, bell_circuit):
        assert bell_circuit.depth == 2

    def test_gate_count(self, bell_circuit):
        assert bell_circuit.gate_count == 2

    def test_empty_circuit_depth_zero(self):
        assert Circuit(1).depth == 0
        assert Circuit(1).gate_count == 0

    def test_ghz_depth(self, ghz_circuit):
        c = ghz_circuit(4)
        assert c.n_qubits == 4
        assert c.gate_count == 4


class TestCircuitRun:
    def test_run_delegates_to_sf_run(self, bell_circuit):
        mock_result = object()
        with patch("superfermion.runner.run", return_value=mock_result) as mock_run:
            result = bell_circuit.run(device="cpu", shots=500)
        mock_run.assert_called_once_with(
            bell_circuit, device="cpu", shots=500,
        )
        assert result is mock_result


class TestSerialization:
    def test_to_qasm3_bell(self, bell_circuit):
        qasm = bell_circuit.to_qasm3()
        assert qasm.startswith("OPENQASM 3.0;")
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm

    def test_to_json_roundtrip(self, bell_circuit):
        data = json.loads(bell_circuit.to_json())
        assert data["n_qubits"] == 2
        assert len(data["gates"]) == 2
        restored = Circuit.from_json(bell_circuit.to_json())
        assert restored.to_gate_list() == bell_circuit.to_gate_list()

    def test_to_gate_list_structure(self, bell_circuit):
        gates = bell_circuit.to_gate_list()
        assert gates[0] == {"name": "H", "qubits": [0]}
        assert gates[1] == {"name": "CNOT", "qubits": [0, 1]}


class TestEdgeCases:
    def test_single_qubit_circuit(self):
        c = Circuit(1).h(0)
        assert c.n_qubits == 1
        assert c.gate_count == 1

    def test_duplicate_gate_names_normalized_uppercase(self):
        c = Circuit(1).h(0).h(0)
        gates = c.to_gate_list()
        assert all(g["name"] == "H" for g in gates)
        assert len(gates) == 2

    def test_measure_adds_classical_bit(self):
        c = Circuit(1).h(0).measure(0)
        gates = c.to_gate_list()
        assert gates[-1]["name"] == "MEASURE"
        assert gates[-1]["classical_bits"] == [0]
