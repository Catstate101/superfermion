"""
Embedding Module — Pure Flax convenience wrapper with no quantum integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

class Embedding(nn.Module):
    """Flax Embed convenience wrapper (no quantum-specific logic)."""
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features
        )(inputs)
