"""
Riemannian Optimizer — Optimizing on the natural geometry of quantum states.
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict

def riemannian_gradient(grad: jnp.ndarray, metric_tensor: jnp.ndarray) -> jnp.ndarray:
    """Compute the Riemannian gradient given a Euclidean gradient and a metric tensor.
    
    In QML, this is often the Natural Gradient where the metric tensor 
    is the Fubini-Study metric (or Quantum Fisher Information Matrix).
    
    Args:
        grad: Euclidean gradient.
        metric_tensor: Metric tensor (e.g. QFIM).
        
    Returns:
        Riemannian (natural) gradient.
    """
    # G_inv @ grad
    # Using pseudo-inverse for stability if the metric is singular
    metric_inv = jnp.linalg.pinv(metric_tensor)
    return jnp.dot(metric_inv, grad)
