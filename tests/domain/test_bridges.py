"""Cross-framework bridge domain tests."""

import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.domain


class TestNativeQASM:
    def test_to_qasm3_produces_valid_string(self, bell_circuit):
        qasm = bell_circuit.to_qasm3()
        assert isinstance(qasm, str)
        assert qasm.startswith("OPENQASM 3.0;")
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm


class TestQASMRoundTrip:
    def test_sf_qasm_sf_roundtrip(self, bell_circuit):
        to_qasm = pytest.importorskip(
            "superfermion.bridge",
            reason="bridge module unavailable",
        ).to_qasm
        from_qasm = pytest.importorskip(
            "superfermion.bridge",
            reason="bridge module unavailable",
        ).from_qasm

        qasm_str = to_qasm(bell_circuit)
        assert "OPENQASM" in qasm_str
        restored = from_qasm(qasm_str)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert restored.gate_count == bell_circuit.gate_count


class TestQiskitBridge:
    def test_qiskit_roundtrip(self, bell_circuit):
        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")
        qiskit_circ = bridge.to_qiskit(bell_circuit)
        restored = bridge.from_qiskit(qiskit_circ)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert len(restored.to_gate_list()) == len(bell_circuit.to_gate_list())


class TestCirqBridge:
    def test_cirq_roundtrip(self, bell_circuit):
        pytest.importorskip("cirq")
        bridge = pytest.importorskip("superfermion.bridge")
        cirq_circ = bridge.to_cirq(bell_circuit)
        restored = bridge.from_cirq(cirq_circ)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert restored.gate_count >= 2
