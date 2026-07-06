"""
SPSA Optimizer — Simultaneous Perturbation Stochastic Approximation.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple

def spsa_grad(fun: Callable, params: jnp.ndarray, key: jax.random.PRNGKey, 
              delta: float = 0.01) -> jnp.ndarray:
    """Approximate gradient using SPSA.
    
    Args:
        fun: Function to differentiate.
        params: Current parameters.
        key: PRNG key for generating perturbations.
        delta: Perturbation magnitude.
        
    Returns:
        Approximated gradient.
    """
    # 1. Generate random perturbation direction (Bernoulli +/- 1)
    perturbation = jax.random.bernoulli(key, 0.5, shape=params.shape).astype(jnp.float32) * 2 - 1
    
    # 2. Evaluate function at perturbed points
    params_plus = params + delta * perturbation
    params_minus = params - delta * perturbation
    
    y_plus = fun(params_plus)
    y_minus = fun(params_minus)
    
    # 3. Compute gradient approximation
    grad = (y_plus - y_minus) / (2 * delta * perturbation)
    
    return grad
