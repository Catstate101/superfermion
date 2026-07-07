"""Comprehensive cross-framework bridge domain tests."""

import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.domain

bridge = pytest.importorskip("superfermion.bridge", reason="bridge module unavailable")


class TestQiskitRoundTrip:
    def test_to_qiskit_from_qiskit_roundtrip(self, bell_circuit):
        pytest.importorskip("qiskit")
        qiskit_circ = bridge.to_qiskit(bell_circuit)
        restored = bridge.from_qiskit(qiskit_circ)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert restored.gate_count == bell_circuit.gate_count

    def test_from_qiskit_preserves_gate_types(self, bell_circuit):
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        restored = bridge.from_qiskit(qc)
        names = {g["name"] for g in restored.to_gate_list()}
        assert "H" in names
        assert names & {"CX", "CNOT"}


class TestIonQFormat:
    def test_to_ionq_bell_circuit_format(self, bell_circuit):
        ionq_gates = bridge.to_ionq(bell_circuit)
        assert isinstance(ionq_gates, list)
        assert len(ionq_gates) == 2

        h_gate = ionq_gates[0]
        assert h_gate["gate"] == "h"
        assert "target" in h_gate

        cnot_gate = ionq_gates[1]
        assert cnot_gate["gate"] == "cnot"
        assert "control" in cnot_gate
        assert "target" in cnot_gate

    def test_to_ionq_parametric_rotation(self):
        c = sf.Circuit(1).rx(1.57, 0)
        ionq_gates = bridge.to_ionq(c)
        assert len(ionq_gates) == 1
        assert ionq_gates[0]["gate"] == "rx"
        assert "rotation" in ionq_gates[0]
        assert abs(ionq_gates[0]["rotation"] - 1.57) < 1e-6


class TestBraketFormat:
    def test_to_braket_bell_circuit(self, bell_circuit):
        braket = pytest.importorskip("braket.circuits")
        braket_circ = bridge.to_braket(bell_circuit)
        assert isinstance(braket_circ, braket.Circuit)
        assert braket_circ.qubit_count == bell_circuit.n_qubits
        assert len(braket_circ.instructions) >= 2

    def test_to_braket_gate_instructions(self, bell_circuit):
        pytest.importorskip("braket.circuits")
        braket_circ = bridge.to_braket(bell_circuit)
        instr_names = [instr.operator.name for instr in braket_circ.instructions]
        assert "H" in instr_names
        assert "CNot" in instr_names
