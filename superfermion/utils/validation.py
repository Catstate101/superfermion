"""
Validation — Input validation utilities for Superfermion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def validate_n_qubits(n: int, max_qubits: int = 30) -> None:
    """Validate qubit count."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n}")
    if n > max_qubits:
        raise ValueError(
            f"n_qubits={n} exceeds maximum {max_qubits}. "
            f"Large circuits may require tensor network backends."
        )


def validate_qubit_index(qubit: int, n_qubits: int, gate: str = "") -> None:
    """Validate a qubit index is in range."""
    from superfermion.utils.exceptions import QubitIndexError
    if not isinstance(qubit, int) or qubit < 0 or qubit >= n_qubits:
        raise QubitIndexError(qubit, n_qubits, gate)


def validate_statevector(sv: np.ndarray, n_qubits: int = None) -> None:
    """Validate a statevector."""
    if sv.ndim != 1:
        raise ValueError(f"Statevector must be 1D, got shape {sv.shape}")
    
    dim = sv.shape[0]
    import math
    expected_n = int(math.log2(dim))
    if 2**expected_n != dim:
        raise ValueError(f"Statevector dimension {dim} is not a power of 2")
    
    if n_qubits is not None and expected_n != n_qubits:
        raise ValueError(
            f"Statevector dimension {dim} does not match {n_qubits} qubits "
            f"(expected {2**n_qubits})"
        )


def validate_probability(p: float, name: str = "probability") -> None:
    """Validate a probability value is in [0, 1]."""
    if not (0 <= p <= 1):
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def validate_shots(shots: int) -> None:
    """Validate measurement shot count."""
    if not isinstance(shots, int) or shots < 1:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    if shots > 1_000_000:
        raise ValueError(f"shots={shots} exceeds maximum 1,000,000")


def validate_angle(theta: float, name: str = "angle") -> None:
    """Validate a rotation angle (warn if outside typical range)."""
    import math
    if abs(theta) > 100 * math.pi:
        import warnings
        warnings.warn(
            f"{name}={theta:.2f} is very large. "
            f"Did you forget to wrap to [-pi, pi]?",
            stacklevel=3
        )
