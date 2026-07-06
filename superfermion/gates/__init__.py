"""Gates — single source of truth for gate matrices, mappings, and endianness."""

from superfermion.gates.endianness import (
    sf_to_qiskit_qubit,
    qiskit_to_sf_qubit,
    sf_to_qiskit_qubits,
    qiskit_to_sf_qubits,
    sf_to_qiskit_statevector,
    qiskit_to_sf_statevector,
)
from superfermion.gates.matrices import gate_unitary_matrix
from superfermion.gates.registry import GateRegistry

__all__ = [
    "sf_to_qiskit_qubit",
    "qiskit_to_sf_qubit",
    "sf_to_qiskit_qubits",
    "qiskit_to_sf_qubits",
    "sf_to_qiskit_statevector",
    "qiskit_to_sf_statevector",
    "gate_unitary_matrix",
    "GateRegistry",
]
