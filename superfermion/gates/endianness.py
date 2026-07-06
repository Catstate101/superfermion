"""Endianness conversion utilities for interoperability.

SF uses q0=MSB convention. Qiskit/Rust core use q0=LSB.
These functions are the single source of truth for conversions.
"""

from __future__ import annotations

from typing import List

import numpy as np


def sf_to_qiskit_qubit(q: int, n_qubits: int) -> int:
    """Convert a single qubit index from SF (MSB) to Qiskit (LSB)."""
    return n_qubits - 1 - q


def qiskit_to_sf_qubit(q: int, n_qubits: int) -> int:
    """Convert a single qubit index from Qiskit (LSB) to SF (MSB)."""
    return n_qubits - 1 - q


def sf_to_qiskit_qubits(qubits: List[int], n_qubits: int) -> List[int]:
    """Convert a list of qubit indices from SF (MSB) to Qiskit (LSB)."""
    return [n_qubits - 1 - q for q in qubits]


def qiskit_to_sf_qubits(qubits: List[int], n_qubits: int) -> List[int]:
    """Convert a list of qubit indices from Qiskit (LSB) to SF (MSB)."""
    return [n_qubits - 1 - q for q in qubits]


def sf_to_qiskit_statevector(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a statevector from SF (MSB) ordering to Qiskit (LSB) ordering."""
    return sv.reshape([2] * n_qubits).transpose(list(range(n_qubits))[::-1]).flatten()


def qiskit_to_sf_statevector(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a statevector from Qiskit (LSB) ordering to SF (MSB) ordering."""
    return sv.reshape([2] * n_qubits).transpose(list(range(n_qubits))[::-1]).flatten()
