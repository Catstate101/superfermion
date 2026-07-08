"""
TopologyFactory — generic hardware topology creation.

Pattern: Factory
Problem: Topologies must be created without hardcoding provider names.
Solution: Generic factory delegates to Rust CouplingMap for any topology shape.
         Named presets ("eagle", "heron") are convenience aliases.

Usage::

    backend = TopologyFactory.create("heavy_hex", n_qubits=127,
                                     basis_gates=["rz", "sx", "x", "ecr"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GenericBackend:
    """Concrete BenchmarkBackend created by TopologyFactory."""

    _n_qubits: int
    _basis_gates: List[str]
    _coupling_map: List[Tuple[int, int]]

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def basis_gates(self) -> List[str]:
        return self._basis_gates

    @property
    def coupling_map(self) -> List[Tuple[int, int]]:
        return self._coupling_map


_PRESETS = {
    "eagle": ("heavy_hex", 127, ["id", "rz", "sx", "x", "ecr"]),
    "heron": ("heavy_hex", 133, ["id", "rz", "sx", "x", "ecr"]),
    "ankaa": ("grid", 84, ["rx", "rz", "cz"]),
    "aria": ("all_to_all", 25, ["gpi", "gpi2", "ms"]),
    "forte": ("all_to_all", 36, ["gpi", "gpi2", "ms"]),
    "garnet": ("grid", 20, ["rx", "rz", "cz"]),
}


class TopologyFactory:
    """Create hardware topologies by shape, not by vendor name."""

    @staticmethod
    def create(shape: str, n_qubits: int = 0,
               basis_gates: List[str] | None = None,
               **kwargs) -> GenericBackend:
        """Create a topology backend.

        Args:
            shape: Topology shape — "heavy_hex", "linear", "grid",
                   "all_to_all", or a preset alias ("eagle", "heron", etc.)
            n_qubits: Number of qubits (required for non-preset shapes).
            basis_gates: Native gate set of the target.
            **kwargs: Extra args passed to the topology generator (e.g. rows, cols for grid).
        """
        if shape.lower() in _PRESETS:
            preset_shape, preset_n, preset_basis = _PRESETS[shape.lower()]
            n_qubits = n_qubits or preset_n
            basis_gates = basis_gates or preset_basis
            shape = preset_shape

        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        basis_gates = basis_gates or ["rz", "sx", "x", "cx"]

        edges = _build_edges(shape, n_qubits, **kwargs)
        return GenericBackend(
            _n_qubits=n_qubits,
            _basis_gates=basis_gates,
            _coupling_map=edges,
        )

    @staticmethod
    def from_edges(n_qubits: int, edges: List[Tuple[int, int]],
                   basis_gates: List[str] | None = None) -> GenericBackend:
        return GenericBackend(
            _n_qubits=n_qubits,
            _basis_gates=basis_gates or ["rz", "sx", "x", "cx"],
            _coupling_map=list(edges),
        )

    @staticmethod
    def list_presets() -> list[str]:
        return sorted(_PRESETS)


def _build_edges(shape: str, n: int, **kwargs) -> List[Tuple[int, int]]:
    try:
        import _sf_core
        CM = _sf_core.CouplingMap
        if shape == "heavy_hex":
            return CM.heavy_hex(n).edges()
        elif shape == "linear":
            return CM.linear(n).edges()
        elif shape == "grid":
            rows = kwargs.get("rows", int(n ** 0.5))
            cols = kwargs.get("cols", (n + rows - 1) // rows)
            return CM.grid(rows, cols).edges()
        elif shape == "all_to_all":
            edges = CM.all_to_all(n).edges()
            return edges + [(b, a) for a, b in edges]
    except ImportError:
        pass

    if shape == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    elif shape == "all_to_all":
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    elif shape == "grid":
        rows = kwargs.get("rows", int(n ** 0.5))
        cols = kwargs.get("cols", (n + rows - 1) // rows)
        edges = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= n:
                    continue
                if c + 1 < cols and idx + 1 < n:
                    edges.append((idx, idx + 1))
                if r + 1 < rows and (r + 1) * cols + c < n:
                    edges.append((idx, (r + 1) * cols + c))
        return edges
    return []
