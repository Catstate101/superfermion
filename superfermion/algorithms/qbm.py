"""
Quantum Boltzmann Machine (QBM) — Generative Quantum Model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

import superfermion as sf


class QBM(nn.Module):
    """A simple Quantum Boltzmann Machine implementation.
    
    The energy of a state is modeled by a transverse-field Ising-like Hamiltonian
    whose coefficients are trained.
    """
    n_qubits: int
    backend: str = "jax"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the unnormalized probability (energy) of data x.
        
        Note: True QBMs involve sampling. Here we implement the 
        energy-based kernel that can be used for training.
        """
        # 1. Learnable Hamiltonian parameters (Weights & Biases)
        # J_ij Z_i Z_j + h_i Z_i
        
        # Biases
        h = self.param(
            "h", 
            jax.nn.initializers.zeros, 
            (self.n_qubits,)
        )
        
        # Weights (Interaction matrix)
        J = self.param(
            "J", 
            jax.nn.initializers.normal(stddev=0.1), 
            (self.n_qubits, self.n_qubits)
        )
        
        # Ensure J is symmetric for a physical Ising model
        J_sym = (J + J.T) / 2.0
        
        # Energy calculation: E(x) = -sum h_i x_i - sum J_ij x_i x_j
        # Assuming x is in {-1, 1} or {0, 1} mapped to {-1, 1}
        x_mapped = 2.0 * x - 1.0
        
        energies = -jnp.dot(x_mapped, h) - jnp.sum(x_mapped * (J_sym @ x_mapped.T).T, axis=-1)
        
        return energies

    def get_partition_function(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Exhaustive partition function calculation for small qubit counts."""
        # Generating all 2^n states
        states = jnp.array([
            [int(b) for b in format(i, f'0{self.n_qubits}b')]
            for i in range(2**self.n_qubits)
        ])
        
        unnorm_probs = jnp.exp(-self.apply(params, states))
        return jnp.sum(unnorm_probs)
