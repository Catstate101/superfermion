"""Quantum error correction domain tests."""

import pytest

from superfermion.circuit import Circuit


pytestmark = pytest.mark.domain

SurfaceCode2D = pytest.importorskip(
    "superfermion.qec",
    reason="qec module unavailable",
).SurfaceCode2D


class TestSurfaceCode:
    def test_surface_code_construction(self):
        code = SurfaceCode2D(distance=3)
        assert code.d == 3
        assert code.n_data == 9
        assert code.n_ancilla == 8

    def test_surface_code_builds_circuit(self):
        code = SurfaceCode2D(distance=3)
        circuit = code.build()
        assert isinstance(circuit, Circuit)
        assert circuit.n_qubits == code.n_data + code.n_ancilla
        assert circuit.gate_count > 0

    def test_surface_code_uses_clifford_gates(self):
        code = SurfaceCode2D(distance=3)
        circuit = code.build()
        gate_names = {g["name"] for g in circuit.to_gate_list()}
        assert gate_names.issubset({"H", "CNOT", "CX", "MEASURE"})
