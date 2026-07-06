"""TopologyCache with TTL and size limits for Singularity."""

from __future__ import annotations

import time
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from superfermion.circuit import Circuit


class TopologyCache:
    """Thread-safe cache for pre-computed statevectors with TTL and size bounds."""

    def __init__(self, max_entries: int = 64, ttl_seconds: float = 300.0):
        self._cache: Dict[str, Tuple[np.ndarray, dict, float]] = {}
        self._max_entries = max_entries
        self._ttl = ttl_seconds

    @staticmethod
    def fingerprint(circuit: Circuit) -> str:
        """Fast structural + parameter hash for unique circuit identification."""
        parts = [str(circuit.n_qubits)]
        for g in circuit._gates:
            parts.append(f"{g.name}:{g.qubits}:{g.params}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def get(self, key: str) -> Optional[Tuple[np.ndarray, dict]]:
        """Return (statevector, metadata) if cached and fresh, else None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        sv, meta, timestamp = entry
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        return (sv, meta)

    def put(self, key: str, statevector: np.ndarray, metadata: dict) -> None:
        """Store a statevector in the cache, evicting oldest if full."""
        if len(self._cache) >= self._max_entries:
            oldest = min(self._cache, key=lambda k: self._cache[k][2])
            del self._cache[oldest]
        self._cache[key] = (np.asarray(statevector, dtype=np.complex128), metadata, time.time())

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
