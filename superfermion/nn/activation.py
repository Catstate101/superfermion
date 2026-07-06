"""
Activation Functions — Classical and Quantum non-linearities.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer

def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.relu(x)

def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x)

def silu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.silu(x)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)

def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    return jax.nn.softmax(x, axis=axis)

class QAct(nn.Module):
    """Quantum Activation Layer.
    
    Uses 1-qubit rotation circuits as a trainable non-linearity.
    """
    backend: str = "jax"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # We apply a 1-qubit circuit to each element of the input
        # This is quite expensive if not batched correctly.
        original_shape = x.shape
        x_flat = x.reshape(-1, 1) # Treat each element as an input to a 1-qubit gate
        
        # 1-qubit circuit: RY(x) -> RY(theta) -> Measure Z
        def circuit_factory(n):
            c = sf.Circuit(1)
            c.ry(sf.param("in"), 0)
            c.ry(sf.param("weight"), 0)
            return c
        
        # We can't easily use QuantumLayer here because it expects features.
        # Let's use it by projecting 1 -> 1
        q_out = QuantumLayer(
            n_qubits=1,
            ansatz=circuit_factory,
            backend=self.backend
        )(x_flat)
        
        # q_out is shape (N, 2) [statevector]
        # We take the probability of |1> as the "activated" value
        res = jnp.abs(q_out[:, 1])**2
        
        return res.reshape(original_shape)
