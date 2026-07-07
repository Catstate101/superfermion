"""
Quantum Boltzmann Machine (QBM) — Generative Quantum Model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

import superfermion as sf


class QBM:
    """A simple Quantum Boltzmann Machine implementation.

    The energy of a state is modeled by a transverse-field Ising-like Hamiltonian
    whose coefficients are trained.
    """

    def __init__(self, n_qubits: int, device: str = "cpu", method: str = "statevector"):
        self.n_qubits = n_qubits
        self.device = device
        self.method = method
        self.h = np.zeros(n_qubits)
        self.J = np.random.normal(0, 0.1, (n_qubits, n_qubits))
        self.J = (self.J + self.J.T) / 2.0

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Compute the energy of data x.

        Args:
            x: Binary data in {0, 1}^n or batched (batch, n).

        Returns:
            Energy values.
        """
        x = np.asarray(x, dtype=np.float64)
        x_mapped = 2.0 * x - 1.0

        energies = -x_mapped @ self.h - np.sum(x_mapped * (self.J @ x_mapped.T).T, axis=-1)
        return energies

    def partition_function(self) -> float:
        """Exhaustive partition function calculation for small qubit counts."""
        states = np.array([
            [int(b) for b in format(i, f'0{self.n_qubits}b')]
            for i in range(2**self.n_qubits)
        ], dtype=np.float64)

        unnorm_probs = np.exp(-self.energy(states))
        return float(np.sum(unnorm_probs))
