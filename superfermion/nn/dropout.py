"""
Dropout Module — Pure Flax convenience wrapper with no quantum integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

class Dropout(nn.Module):
    """Flax Dropout convenience wrapper (no quantum-specific logic)."""
    rate: float
    deterministic: bool = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = None) -> jnp.ndarray:
        deterministic = self.deterministic if deterministic is None else deterministic
        return nn.Dropout(rate=self.rate, deterministic=deterministic)(x)
