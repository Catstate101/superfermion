"""
Normalization Layers — Pure Flax LayerNorm/BatchNorm convenience wrappers with no quantum integration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

class LayerNorm(nn.Module):
    """Hybrid-ready Layer Normalization."""
    use_scale: bool = True
    use_bias: bool = True
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.LayerNorm(
            use_scale=self.use_scale,
            use_bias=self.use_bias,
            epsilon=self.epsilon
        )(x)

class BatchNorm(nn.Module):
    """Hybrid-ready Batch Normalization."""
    use_running_average: bool = True
    momentum: float = 0.99
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.BatchNorm(
            use_running_average=self.use_running_average,
            momentum=self.momentum,
            epsilon=self.epsilon
        )(x)

class GroupNorm(nn.Module):
    """Hybrid-ready Group Normalization."""
    num_groups: int = 32
    use_scale: bool = True
    use_bias: bool = True
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.GroupNorm(
            num_groups=self.num_groups,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
            epsilon=self.epsilon
        )(x)
