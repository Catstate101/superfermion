"""
Transformer Block — Pure Flax Transformer convenience wrapper with no quantum integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from superfermion.nn.attention import MultiHeadAttention
from superfermion.nn.norm import LayerNorm

class TransformerBlock(nn.Module):
    """Transformer block combining MHA, Norm, and MLP."""
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Residual 1: Attention
        y = LayerNorm()(x)
        y = MultiHeadAttention(num_heads=self.num_heads)(y)
        x = x + y
        
        # Residual 2: MLP
        y = LayerNorm()(x)
        y = nn.Dense(features=self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(features=x.shape[-1])(y)
        x = x + y
        
        return x
