"""
Superfermion Data Pipeline — Quantum-aware dataset management.

Provides unified dataset loading, quantum feature map preprocessing,
and integration with NumPy, PyTorch, and built-in benchmarks.

Usage:
    >>> from superfermion.data import Dataset, load_mnist_q
    >>> dataset = Dataset.from_numpy(X, y)
    >>> train, test = dataset.split(0.8)
"""

from __future__ import annotations

from superfermion.data.dataset import (
    Dataset, QuantumDataset, DataLoader,
)
from superfermion.data.preprocessing import (
    normalize_for_encoding, angle_encoding_transform,
    amplitude_encoding_transform, min_max_scale,
)
from superfermion.data.benchmarks import (
    load_mnist_q, load_iris_q, load_xor_q, load_circles_q,
)

__all__ = [
    "Dataset", "QuantumDataset", "DataLoader",
    "normalize_for_encoding", "angle_encoding_transform",
    "amplitude_encoding_transform", "min_max_scale",
    "load_mnist_q", "load_iris_q", "load_xor_q", "load_circles_q",
]
