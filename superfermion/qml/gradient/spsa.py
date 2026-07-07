"""
SPSA Optimizer — Simultaneous Perturbation Stochastic Approximation.
"""

import numpy as np
from typing import Callable


def spsa_grad(fun: Callable, params: np.ndarray, seed: int = 42,
              delta: float = 0.01) -> np.ndarray:
    """Approximate gradient using SPSA.

    Args:
        fun: Function to differentiate (params -> scalar).
        params: Current parameters as numpy array.
        seed: Random seed for generating perturbations.
        delta: Perturbation magnitude.

    Returns:
        Approximated gradient as numpy array.
    """
    seed_int = int(seed) if np.ndim(seed) == 0 else int(np.asarray(seed).flat[0])
    rng = np.random.default_rng(seed_int)
    perturbation = rng.choice([-1.0, 1.0], size=params.shape)

    params_plus = params + delta * perturbation
    params_minus = params - delta * perturbation

    y_plus = fun(params_plus)
    y_minus = fun(params_minus)

    grad = (y_plus - y_minus) / (2 * delta * perturbation)
    return grad
