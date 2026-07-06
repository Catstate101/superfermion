"""
Quantum Kernels — Kernel methods for Quantum Machine Learning.

Implements the Quantum Kernel Estimator (QKE).
"""

from __future__ import annotations

from typing import Callable, Optional
import jax
import jax.numpy as jnp
import superfermion as sf


class QuantumKernel:
    """Estimates the kernel matrix K_ij = |<phi(x_i)|phi(x_j)>|^2.
    
    Args:
        encoding_fn: A function that takes a data vector and returns an sf.Circuit.
        backend: Backend to use for execution.
    """
    def __init__(self, encoding_fn: Callable[[jnp.ndarray], sf.Circuit], backend: str = "jax"):
        self.encoding_fn = encoding_fn
        self.backend = backend
        
    def evaluate(self, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
        """Evaluate the kernel for two data points."""
        c1 = self.encoding_fn(x1)
        c2 = self.encoding_fn(x2)
        
        f1 = sf.qml.circuit_to_jax(c1, backend=self.backend)
        f2 = sf.qml.circuit_to_jax(c2, backend=self.backend)
        
        sv1 = f1()
        sv2 = f2()
        
        # Fidelity |<sv1|sv2>|^2
        fidelity = jnp.abs(jnp.vdot(sv1, sv2))**2
        return float(fidelity)
    
    def calculate_matrix(self, X: jnp.ndarray, Y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Calculate the kernel matrix between datasets X and Y.
        
        If Y is None, calculates the square matrix for X.
        """
        if Y is None:
            Y = X
            
        n_x = X.shape[0]
        n_y = Y.shape[0]
        
        # JAX-vectorized implementation for performance
        # We need a function that maps (x, y) -> fidelity
        def kernel_fn(x, y):
            c_x = self.encoding_fn(x)
            c_y = self.encoding_fn(y)
            
            # Using the fact that <phi(x)|phi(y)> = <0| U_x^\dagger U_y |0>
            # We can build a single circuit and get the prob of |0...0>
            # But with statevector it's easier to just compute dot product.
            fx = sf.qml.circuit_to_jax(c_x, backend=self.backend)
            fy = sf.qml.circuit_to_jax(c_y, backend=self.backend)
            return jnp.abs(jnp.vdot(fx(), fy()))**2
            
        # Vmap over rows and columns
        # Note: This might be slow if calling circuit_to_jax inside vmap.
        # Ideally we'd have a parameterized circuit.
        
        matrix = jnp.zeros((n_x, n_y))
        
        # For now, use a simple loop (can be optimized later)
        # In production this would be JIT-ed
        for i in range(n_x):
            for j in range(n_y):
                matrix = matrix.at[i, j].set(self.evaluate(X[i], Y[j]))
        
        return matrix
