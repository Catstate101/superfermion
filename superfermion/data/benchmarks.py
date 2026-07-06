"""
Benchmark Datasets — Built-in datasets for quantum ML experimentation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from superfermion.data.dataset import QuantumDataset
from superfermion.data.preprocessing import normalize_for_encoding


def load_mnist_q(
    n_samples: int = 200,
    n_features: int = 4,
    classes: Tuple[int, int] = (0, 1),
    seed: int = 42,
) -> QuantumDataset:
    """Load a simulated MNIST-like binary classification dataset.

    Generates PCA-reduced synthetic digit features suitable for
    quantum classifiers. Uses synthetic data to avoid external deps.

    Args:
        n_samples: Total samples (split equally between classes).
        n_features: Number of features (PCA components).
        classes: Two class labels.
        seed: Random seed.

    Returns:
        QuantumDataset with angle encoding metadata.
    """
    rng = np.random.RandomState(seed)
    half = n_samples // 2

    # Class 0: cluster around origin
    X0 = rng.randn(half, n_features) * 0.8 + np.array([1.0] * n_features)
    # Class 1: cluster offset
    X1 = rng.randn(n_samples - half, n_features) * 0.8 + np.array([-1.0] * n_features)

    X = np.vstack([X0, X1])
    y = np.array([classes[0]] * half + [classes[1]] * (n_samples - half), dtype=np.float64)

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    # Normalize for angle encoding
    X = normalize_for_encoding(X, "angle")

    return QuantumDataset(X, y, name="MNIST-Q", encoding="angle", n_qubits=n_features)


def load_iris_q(
    n_features: int = 4,
    binary: bool = True,
    seed: int = 42,
) -> QuantumDataset:
    """Load a synthetic Iris-like dataset for quantum classification.

    Args:
        n_features: Number of features.
        binary: If True, only use 2 classes.
        seed: Random seed.

    Returns:
        QuantumDataset.
    """
    rng = np.random.RandomState(seed)

    # 3 clusters
    means = [
        np.array([0.5, 0.3, 0.2, 0.1])[:n_features],
        np.array([-0.3, 0.8, -0.1, 0.5])[:n_features],
        np.array([0.1, -0.5, 0.9, -0.3])[:n_features],
    ]

    n_per_class = 50
    n_classes = 2 if binary else 3
    X_list, y_list = [], []

    for i in range(n_classes):
        Xi = rng.randn(n_per_class, n_features) * 0.3 + means[i]
        X_list.append(Xi)
        y_list.extend([i] * n_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float64)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X = normalize_for_encoding(X, "angle")

    return QuantumDataset(X, y, name="Iris-Q", encoding="angle", n_qubits=n_features)


def load_xor_q(
    n_samples: int = 100,
    noise: float = 0.1,
    seed: int = 42,
) -> QuantumDataset:
    """Load an XOR dataset — a quantum advantage benchmark.

    XOR is linearly non-separable, making it useful for testing
    quantum classifiers with entanglement.

    Args:
        n_samples: Number of samples.
        noise: Gaussian noise level.
        seed: Random seed.

    Returns:
        QuantumDataset with 2 features.
    """
    rng = np.random.RandomState(seed)
    n = n_samples // 4

    # Four quadrants
    X = np.vstack([
        rng.randn(n, 2) * noise + [1, 1],
        rng.randn(n, 2) * noise + [-1, -1],
        rng.randn(n, 2) * noise + [1, -1],
        rng.randn(n, 2) * noise + [-1, 1],
    ])
    y = np.array(
        [0] * n + [0] * n + [1] * n + [1] * n,
        dtype=np.float64,
    )

    # Ensure we have exactly n_samples
    X = X[:n_samples]
    y = y[:n_samples]

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X = normalize_for_encoding(X, "angle")

    return QuantumDataset(X, y, name="XOR-Q", encoding="angle", n_qubits=2)


def load_circles_q(
    n_samples: int = 200,
    noise: float = 0.1,
    factor: float = 0.5,
    seed: int = 42,
) -> QuantumDataset:
    """Load concentric circles dataset — tests nonlinear classification.

    Args:
        n_samples: Number of samples.
        noise: Gaussian noise.
        factor: Ratio of inner to outer circle radius.
        seed: Random seed.

    Returns:
        QuantumDataset with 2 features.
    """
    rng = np.random.RandomState(seed)
    half = n_samples // 2

    # Outer circle
    theta_outer = rng.uniform(0, 2 * math.pi, half)
    X_outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    X_outer += rng.randn(half, 2) * noise

    # Inner circle
    theta_inner = rng.uniform(0, 2 * math.pi, n_samples - half)
    X_inner = factor * np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1)
    X_inner += rng.randn(n_samples - half, 2) * noise

    X = np.vstack([X_outer, X_inner])
    y = np.array([0] * half + [1] * (n_samples - half), dtype=np.float64)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X = normalize_for_encoding(X, "angle")

    return QuantumDataset(X, y, name="Circles-Q", encoding="angle", n_qubits=2)
