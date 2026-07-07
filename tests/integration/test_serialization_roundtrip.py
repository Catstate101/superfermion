"""Integration tests for circuit and result serialization round-trips."""

import json

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.results import RunResult


pytestmark = pytest.mark.integration


class TestCircuitSerialization:
    def test_json_roundtrip_preserves_gate_sequence(self, bell_circuit):
        json_str = bell_circuit.to_json()
        data = json.loads(json_str)

        assert data["n_qubits"] == 2
        assert len(data["gates"]) == 2

        restored = Circuit.from_json(json_str)
        assert restored.to_gate_list() == bell_circuit.to_gate_list()
        assert restored.n_qubits == bell_circuit.n_qubits
        assert restored.depth == bell_circuit.depth

    def test_json_roundtrip_parametric_circuit(self, parametric_circuit):
        json_str = parametric_circuit.to_json()
        data = json.loads(json_str)
        restored = Circuit.from_json(json_str)

        assert restored.n_qubits == parametric_circuit.n_qubits
        assert restored.gate_count == parametric_circuit.gate_count
        assert [g["name"] for g in data["gates"]] == ["RX", "RY", "CNOT"]
        assert data["gates"][0]["params"] == ["theta"]
        assert data["gates"][1]["params"] == ["phi"]

        parametric_circuit._ensure_gates()
        restored._ensure_gates()
        for orig, rest in zip(parametric_circuit._gates, restored._gates):
            assert orig.name == rest.name
            assert orig.qubits == rest.qubits

    def test_qasm3_export_has_valid_header(self, bell_circuit):
        qasm = bell_circuit.to_qasm3()

        assert isinstance(qasm, str)
        assert qasm.startswith("OPENQASM 3.0;")
        assert "qubit[2] q;" in qasm
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm

    def test_qasm3_parametric_circuit_includes_parameter_names(self, parametric_circuit):
        qasm = parametric_circuit.to_qasm3()

        assert qasm.startswith("OPENQASM 3.0;")
        assert "theta" in qasm
        assert "phi" in qasm

    def test_large_circuit_json_roundtrip(self):
        n = 20
        circuit = Circuit(n)
        for q in range(n):
            circuit.h(q)
        for q in range(n - 1):
            circuit.cnot(q, q + 1)
        for q in range(0, n, 2):
            circuit.rz(0.25, q)

        json_str = circuit.to_json()
        restored = Circuit.from_json(json_str)

        assert restored.n_qubits == n
        assert restored.gate_count == circuit.gate_count
        assert restored.to_gate_list() == circuit.to_gate_list()


class TestRunResultSerialization:
    def test_dict_roundtrip_preserves_all_fields(self, bell_circuit):
        sv = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        original = RunResult(
            counts={"00": 512, "11": 488},
            probabilities={"00": 0.512, "11": 0.488},
            statevector=sv,
            shots=1000,
            circuit=bell_circuit,
            metadata={"device": "cpu", "backend": "statevector"},
        )

        restored = RunResult.from_dict(original.to_dict())

        assert restored.counts == original.counts
        assert restored.probabilities == original.probabilities
        assert restored.shots == original.shots
        assert restored.metadata == original.metadata

    def test_statevector_roundtrips_through_dict(self, bell_circuit):
        sv = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        original = RunResult(
            counts={"00": 500, "11": 500},
            statevector=sv,
            shots=1000,
            circuit=bell_circuit,
        )

        d = original.to_dict()
        assert "statevector_real" in d
        assert "statevector_imag" in d

        restored = RunResult.from_dict(d)
        np.testing.assert_allclose(restored.statevector, sv, rtol=1e-10, atol=1e-10)

    def test_execution_result_roundtrip_via_run(self, bell_circuit):
        """End-to-end: run a circuit, serialize the result, restore it."""
        executed = sf.run(bell_circuit, device="cpu", shots=500)
        assert executed.statevector is not None

        d = executed.to_dict()
        restored = RunResult.from_dict(d)

        assert restored.counts == executed.counts
        assert restored.shots == executed.shots
        np.testing.assert_allclose(
            restored.statevector,
            np.asarray(executed.statevector),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_large_statevector_roundtrip(self):
        n = 8
        dim = 2 ** n
        sv = np.zeros(dim, dtype=np.complex128)
        sv[0] = 1.0

        original = RunResult(
            counts={"00000000": 1000},
            statevector=sv,
            shots=1000,
            circuit=Circuit(n),
            metadata={"note": "large sv"},
        )

        restored = RunResult.from_dict(original.to_dict())
        assert restored.statevector.shape == (dim,)
        np.testing.assert_allclose(restored.statevector, sv)
        assert restored.metadata == original.metadata
