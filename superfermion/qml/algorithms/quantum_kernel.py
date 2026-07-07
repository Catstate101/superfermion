"""
Quantum Kernels — Kernel methods for Quantum Machine Learning.

Implements the Quantum Kernel Estimator (QKE).
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import superfermion as sf


class QuantumKernel:
    """Estimates the kernel matrix K_ij = |<phi(x_i)|phi(x_j)>|^2.

    Args:
        encoding_fn: A function that takes a data vector and returns an sf.Circuit.
        device: Device to use for execution.
        method: Simulation method.
    """
    def __init__(
        self,
        encoding_fn: Callable[[np.ndarray], sf.Circuit],
        device: str = "cpu",
        method: str = "statevector",
    ):
        self.encoding_fn = encoding_fn
        self.device = device
        self.method = method

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate the kernel for two data points."""
        c1 = self.encoding_fn(x1)
        c2 = self.encoding_fn(x2)

        s1 = sf.simulate(c1, device=self.device, method=self.method)
        s2 = sf.simulate(c2, device=self.device, method=self.method)

        sv1 = s1.numpy()
        sv2 = s2.numpy()

        fidelity = float(np.abs(np.vdot(sv1, sv2))**2)
        return fidelity

    def calculate_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate the kernel matrix between datasets X and Y.

        If Y is None, calculates the square matrix for X.
        """
        if Y is None:
            Y = X

        n_x = X.shape[0]
        n_y = Y.shape[0]

        matrix = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                matrix[i, j] = self.evaluate(X[i], Y[j])

        return matrix
