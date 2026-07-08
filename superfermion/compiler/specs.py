"""
Hardware Specifications — Metadata for quantum hardware targets.

Provides connectivity graphs and native gate sets for compilation targets.
Topologies are generated from Rust ``CouplingMap`` — no hardcoded edge lists.
Named presets (``"ibm_eagle"``, ``"ionq_aria"``) are convenience aliases for
generic topology shapes (heavy-hex, all-to-all, grid, linear).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class HardwareSpec:
    """Metadata for a specific quantum device or compilation target."""
    name: str
    n_qubits: int
    native_gates: List[str]
    coupling_map: List[Tuple[int, int]]
    basis_gates: List[str] = field(default_factory=lambda: ["id", "rz", "sx", "x", "cx"])

    @property
    def is_fully_connected(self) -> bool:
        """Returns True if every qubit can talk to every other qubit."""
        expected = self.n_qubits * (self.n_qubits - 1)
        return len(self.coupling_map) == expected


def _coupling_edges(shape: str, n_qubits: int, **kwargs) -> List[Tuple[int, int]]:
    """Generate coupling map edges from Rust ``CouplingMap`` by topology shape."""
    from superfermion._sf_core import CouplingMap as CM

    if shape == "heavy_hex":
        return CM.heavy_hex(n_qubits).edges()
    elif shape == "linear":
        return CM.linear(n_qubits).edges()
    elif shape == "grid":
        rows = kwargs.get("rows", int(n_qubits ** 0.5))
        cols = kwargs.get("cols", (n_qubits + rows - 1) // rows)
        return CM.grid(rows, cols).edges()
    elif shape == "all_to_all":
        edges = CM.all_to_all(n_qubits).edges()
        return edges + [(b, a) for a, b in edges]
    else:
        raise ValueError(f"Unknown topology shape: {shape!r}")


SPECS: Dict[str, HardwareSpec] = {
    "ibm_eagle": HardwareSpec(
        name="ibm_eagle",
        n_qubits=127,
        native_gates=["id", "rz", "sx", "x", "cx", "ecr"],
        coupling_map=_coupling_edges("heavy_hex", 127),
    ),
    "ibm_heron": HardwareSpec(
        name="ibm_heron",
        n_qubits=133,
        native_gates=["id", "rz", "sx", "x", "ecr"],
        coupling_map=_coupling_edges("heavy_hex", 133),
    ),
    "rigetti_ankaa": HardwareSpec(
        name="rigetti_ankaa",
        n_qubits=84,
        native_gates=["rx", "rz", "cz"],
        coupling_map=_coupling_edges("grid", 84, rows=7, cols=12),
    ),
    "ionq_aria": HardwareSpec(
        name="ionq_aria",
        n_qubits=25,
        native_gates=["gpi", "gpi2", "ms"],
        coupling_map=_coupling_edges("all_to_all", 25),
    ),
    "ionq_forte": HardwareSpec(
        name="ionq_forte",
        n_qubits=36,
        native_gates=["gpi", "gpi2", "ms"],
        coupling_map=_coupling_edges("all_to_all", 36),
    ),
    "linear_5": HardwareSpec(
        name="linear_5",
        n_qubits=5,
        native_gates=["h", "x", "y", "z", "cx"],
        coupling_map=_coupling_edges("linear", 5),
    ),
    "jax": HardwareSpec(
        name="jax",
        n_qubits=40,
        native_gates=["all"],
        coupling_map=[],
    ),
    "cluster": HardwareSpec(
        name="cluster",
        n_qubits=100,
        native_gates=["all"],
        coupling_map=[],
    ),
}


def get_spec(name: str) -> Optional[HardwareSpec]:
    """Retrieve a hardware spec by name."""
    return SPECS.get(name.lower())


def list_devices() -> List[str]:
    """List all supported devices."""
    return list(SPECS.keys())
