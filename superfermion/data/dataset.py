"""
Dataset — Unified dataset class with quantum-aware batching and splitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np


class Dataset:
    """Unified dataset for quantum machine learning.

    Supports feature arrays, labels, batching, splitting, and shuffling.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Optional label array of shape (n_samples,) or (n_samples, n_classes).
        name: Dataset name.

    Examples:
        >>> X = np.random.randn(100, 4)
        >>> y = np.random.randint(0, 2, 100)
        >>> ds = Dataset(X, y, name="my_data")
        >>> train, test = ds.split(0.8)
        >>> for batch_x, batch_y in ds.batches(32):
        ...     pass
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        name: str = "",
    ) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64) if y is not None else None
        self.name = name

        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {len(self.X)}, y: {len(self.y)}"
            )

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1

    @property
    def n_classes(self) -> int:
        if self.y is None:
            return 0
        return len(np.unique(self.y))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.X.shape

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset]:
        """Split into train and test sets.

        Args:
            train_ratio: Fraction of data for training.
            shuffle: Whether to shuffle before splitting.
            seed: Random seed.

        Returns:
            (train_dataset, test_dataset) tuple.
        """
        n = self.n_samples
        indices = np.arange(n)

        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

        split_idx = int(n * train_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        train_y = self.y[train_idx] if self.y is not None else None
        test_y = self.y[test_idx] if self.y is not None else None

        return (
            Dataset(self.X[train_idx], train_y, name=f"{self.name}_train"),
            Dataset(self.X[test_idx], test_y, name=f"{self.name}_test"),
        )

    def batches(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Iterate over mini-batches.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Shuffle before batching.
            seed: Random seed.

        Yields:
            (X_batch, y_batch) tuples.
        """
        indices = np.arange(self.n_samples)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

        for start in range(0, self.n_samples, batch_size):
            end = min(start + batch_size, self.n_samples)
            batch_idx = indices[start:end]
            batch_y = self.y[batch_idx] if self.y is not None else None
            yield self.X[batch_idx], batch_y

    def shuffle(self, seed: Optional[int] = None) -> Dataset:
        """Return a new shuffled dataset."""
        rng = np.random.RandomState(seed)
        indices = np.arange(self.n_samples)
        rng.shuffle(indices)
        new_y = self.y[indices] if self.y is not None else None
        return Dataset(self.X[indices], new_y, name=self.name)

    def subset(self, indices: Union[List[int], np.ndarray]) -> Dataset:
        """Get a subset of the dataset."""
        new_y = self.y[indices] if self.y is not None else None
        return Dataset(self.X[indices], new_y, name=f"{self.name}_subset")

    @classmethod
    def from_numpy(cls, X: np.ndarray, y: Optional[np.ndarray] = None, name: str = "") -> Dataset:
        """Create from numpy arrays."""
        return cls(X, y, name=name)

    def to_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Export as numpy arrays."""
        return self.X, self.y

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        y_val = self.y[idx] if self.y is not None else None
        return self.X[idx], y_val

    def __repr__(self) -> str:
        name_part = f"'{self.name}', " if self.name else ""
        label_part = f", classes={self.n_classes}" if self.y is not None else ""
        return f"Dataset({name_part}samples={self.n_samples}, features={self.n_features}{label_part})"


class QuantumDataset(Dataset):
    """Dataset with pre-computed quantum feature maps.

    Extends Dataset with quantum-specific properties like
    required qubits for encoding and encoding type metadata.

    Args:
        X: Feature array.
        y: Label array.
        encoding: Encoding type used ('angle', 'amplitude', 'iqp').
        n_qubits: Number of qubits needed for encoding.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        name: str = "",
        encoding: str = "angle",
        n_qubits: Optional[int] = None,
    ) -> None:
        super().__init__(X, y, name=name)
        self.encoding = encoding
        self._n_qubits = n_qubits

    @property
    def n_qubits_required(self) -> int:
        """Number of qubits needed to encode this dataset."""
        if self._n_qubits:
            return self._n_qubits
        if self.encoding == "amplitude":
            return int(math.ceil(math.log2(max(self.n_features, 2))))
        return self.n_features  # angle encoding: 1 qubit per feature

    def __repr__(self) -> str:
        return (
            f"QuantumDataset('{self.name}', samples={self.n_samples}, "
            f"features={self.n_features}, encoding='{self.encoding}', "
            f"qubits_required={self.n_qubits_required})"
        )


class DataLoader:
    """Iterable data loader with batching and multi-epoch support.

    Args:
        dataset: The dataset to load from.
        batch_size: Samples per batch.
        shuffle: Shuffle each epoch.
        seed: Base random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        epoch_seed = self.seed + self._epoch if self.seed is not None else None
        self._epoch += 1
        return self.dataset.batches(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=epoch_seed,
        )

    @property
    def n_batches(self) -> int:
        return int(math.ceil(self.dataset.n_samples / self.batch_size))

    def __len__(self) -> int:
        return self.n_batches

    def __repr__(self) -> str:
        return (
            f"DataLoader(samples={self.dataset.n_samples}, "
            f"batch_size={self.batch_size}, batches={self.n_batches})"
        )
