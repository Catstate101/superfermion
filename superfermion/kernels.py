"""Analytical kernels for explicitly supported circuit structures."""

from __future__ import annotations

import numpy as np


def angle_fidelity_kernel(x, y=None, *, block_rows: int = 64) -> np.ndarray:
    """Exact fidelity kernel for product RY encoding plus a fixed unitary.

    The fixed unitary cancels in the overlap, leaving the product of
    cos((x_i-y_i)/2)**2. This shortcut is not valid for data-dependent
    entanglers or arbitrary multi-layer feature maps.
    """
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("inputs must be two matrices with the same feature count")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    if not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
    kernel = np.empty((len(x), len(y)), dtype=np.float64)
    for start in range(0, len(x), block_rows):
        angles = (x[start:start + block_rows, None, :] - y[None, :, :]) * 0.5
        np.cos(angles, out=angles)
        np.square(angles, out=angles)
        kernel[start:start + block_rows] = np.prod(angles, axis=2)
    return kernel
