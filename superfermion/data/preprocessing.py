"""
Preprocessing — Data normalization and quantum encoding transforms.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def min_max_scale(
    X: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Scale features to a given range.

    Args:
        X: Input array of shape (n_samples, n_features).
        feature_range: Target (min, max) range.

    Returns:
        Scaled array.
    """
    X = np.asarray(X, dtype=np.float64)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scale = x_max - x_min
    scale = np.where(scale == 0, 1.0, scale)  # Avoid division by zero

    X_std = (X - x_min) / scale
    lo, hi = feature_range
    return X_std * (hi - lo) + lo


def normalize_for_encoding(
    X: np.ndarray,
    encoding: str = "angle",
) -> np.ndarray:
    """Normalize data for quantum encoding.

    Args:
        X: Input feature array.
        encoding: Target encoding ('angle', 'amplitude', 'basis').

    Returns:
        Normalized array ready for quantum encoding.
    """
    X = np.asarray(X, dtype=np.float64)

    if encoding == "angle":
        # Scale to [-pi, pi]
        return min_max_scale(X, feature_range=(-math.pi, math.pi))

    elif encoding == "amplitude":
        # Normalize each sample to unit norm (for amplitude encoding)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        # Pad to power of 2 if needed
        n_features = X.shape[1]
        n_qubits = int(math.ceil(math.log2(max(n_features, 2))))
        target_dim = 2 ** n_qubits
        if n_features < target_dim:
            padding = np.zeros((X.shape[0], target_dim - n_features))
            X = np.hstack([X, padding])
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
        return X / norms

    elif encoding == "basis":
        # Binary encoding: threshold at mean
        threshold = X.mean(axis=0)
        return (X > threshold).astype(np.float64)

    else:
        raise ValueError(f"Unknown encoding: '{encoding}'. Use 'angle', 'amplitude', or 'basis'.")


def angle_encoding_transform(X: np.ndarray) -> np.ndarray:
    """Transform features to angles for Ry/Rz gate encoding.

    Maps each feature to [0, π] range for direct use as rotation angles.

    Args:
        X: Feature array (n_samples, n_features).

    Returns:
        Array of angles in [0, π].
    """
    return min_max_scale(X, feature_range=(0.0, math.pi))


def amplitude_encoding_transform(X: np.ndarray) -> np.ndarray:
    """Transform features to amplitudes for amplitude encoding.

    Normalizes each sample to unit L2 norm and pads to power-of-2 dimension.

    Args:
        X: Feature array (n_samples, n_features).

    Returns:
        Normalized, padded array.
    """
    return normalize_for_encoding(X, encoding="amplitude")
