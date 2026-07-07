"""
Riemannian Optimizer — Optimizing on the natural geometry of quantum states.

Uses sf.State.qfim() for the metric tensor. No JAX dependency.
"""

import numpy as np


def riemannian_gradient(grad: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
    """Compute the Riemannian gradient given a Euclidean gradient and metric tensor.

    In QML, this is the Natural Gradient where the metric tensor is the
    Fubini-Study metric (Quantum Fisher Information Matrix).

    Args:
        grad: Euclidean gradient (1-D array).
        metric_tensor: Metric tensor / QFIM (2-D array).

    Returns:
        Riemannian (natural) gradient.
    """
    metric_inv = np.linalg.pinv(metric_tensor)
    return metric_inv @ grad
