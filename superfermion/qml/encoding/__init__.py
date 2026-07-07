"""
Quantum Data Encoding — Map classical data to quantum states.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np
from superfermion.circuit import Circuit


def angle_encoding(n_qubits: int, data: np.ndarray, rotation: str = "RY") -> Circuit:
    """Map data to single-qubit rotations.

    Args:
        n_qubits: Number of qubits.
        data: Classical data vector of length n_qubits.
        rotation: Rotation gate to use ("RX", "RY", or "RZ").

    Returns:
        Circuit with encoding applied.
    """
    if len(data) > n_qubits:
        raise ValueError(f"Data length {len(data)} exceeds n_qubits {n_qubits}")

    c = Circuit(n_qubits)
    gate_fn = getattr(c, rotation.lower())

    for i, val in enumerate(data):
        gate_fn(val, i)

    return c


def basis_encoding(n_qubits: int, value: int) -> Circuit:
    """Map an integer to its binary basis state |bin(value)>.

    Example: value=3 (binary 11) -> |11>.
    """
    c = Circuit(n_qubits)
    binary = format(value, f'0{n_qubits}b')
    for i, bit in enumerate(binary):
        if bit == '1':
            c.x(i)
    return c


def amplitude_encoding(n_qubits: int, data: np.ndarray) -> Circuit:
    """Encode a normalized vector into the amplitudes of a quantum state.

    Requires data length == 2^n_qubits.
    """
    dim = 2**n_qubits
    data = np.asarray(data, dtype=np.float64)
    if len(data) != dim:
        padded = np.zeros(dim)
        padded[:len(data)] = data
        data = padded

    norm = np.linalg.norm(data)
    if norm > 1e-10:
        data = data / norm

    c = Circuit(n_qubits)
    c._metadata["initial_state"] = data
    return c


def iqp_encoding(n_qubits: int, data: np.ndarray, reps: int = 1) -> Circuit:
    """Instantaneous Quantum Polynomial (IQP) encoding.

    Hadamards followed by RZ(x_i) and CZ(x_i * x_j).
    Commonly used in Quantum Kernels.
    """
    c = Circuit(n_qubits)

    for _ in range(reps):
        for i in range(n_qubits):
            c.h(i)

        for i in range(min(n_qubits, len(data))):
            c.rz(data[i], i)

        idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if idx < len(data):
                    phi = data[i] * data[j]
                    c.cx(i, j).rz(phi, j).cx(i, j)
                idx += 1

    return c
