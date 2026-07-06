"""
Linear — A fully connected hybrid quantum-classical layer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer

class Linear(nn.Module):
    """A hybrid Linear layer.
    
    Equivalent to a QuantumLayer with a linear feature map.
    """
    n_qubits: int
    features: int
    ansatz: Optional[sf.Circuit] = None
    backend: str = "jax"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Classical projection to match qubit count
        x_proj = nn.Dense(self.n_qubits, name="input_projection")(x)
        
        # Quantum execution
        q_out = QuantumLayer(
            n_qubits=self.n_qubits,
            ansatz=self.ansatz,
            backend=self.backend
        )(x_proj)
        
        # Final projection to desired feature dimension
        # q_out is statevector (2^n)
        return nn.Dense(self.features, name="output_projection")(q_out)
