"""
Recurrent Layers — Pure Flax LSTM/GRU convenience wrappers with no quantum integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

class LSTM(nn.Module):
    """Hybrid-ready Long Short-Term Memory."""
    features: int

    @nn.compact
    def __call__(self, carry, x: jnp.ndarray) -> jnp.ndarray:
        # LSTMCell returns (new_carry, output)
        return nn.LSTMCell(features=self.features)(carry, x)

class GRU(nn.Module):
    """Hybrid-ready Gated Recurrent Unit."""
    features: int

    @nn.compact
    def __call__(self, carry, x: jnp.ndarray) -> jnp.ndarray:
        return nn.GRUCell(features=self.features)(carry, x)

class S4(nn.Module):
    """Hybrid-ready Structured State Space (S4) Layer."""
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Simplistic placeholder for S4 logic using standard Flax Dense for now
        # until a custom S4 kernel is integrated.
        return nn.Dense(features=self.features)(x)
