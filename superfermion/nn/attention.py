"""
Attention Module — Pure Flax Multi-Head Attention convenience wrapper with no quantum integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with hybrid capability."""
    num_heads: int
    qkv_features: int | None = None
    out_features: int | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=self.out_features
        )(x)

class FlashAttention(nn.Module):
    """Placeholder for Quantum-Enhanced Flash Attention."""
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Standard MHA for now
        return nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x)
