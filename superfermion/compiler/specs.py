"""
Hardware Specifications — Metadata for world-class QPUs.

Provides connectivity graphs, native gate sets, and error rates for popular 
quantum hardware from IBM, Rigetti, IonQ, and others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class HardwareSpec:
    """Metadata for a specific quantum device."""
    name: str
    n_qubits: int
    native_gates: List[str]
    coupling_map: List[Tuple[int, int]]  # List of (control, target) pairs
    basis_gates: List[str] = field(default_factory=lambda: ["id", "rz", "sx", "x", "cx"])
    
    @property
    def is_fully_connected(self) -> bool:
        """Returns True if every qubit can talk to every other qubit."""
        expected = self.n_qubits * (self.n_qubits - 1)
        return len(self.coupling_map) == expected


# Predefined world-class hardware specs
SPECS = {
    # IBM Eagle (127 Qubits) - Heavy Hex topology
    "ibm_eagle": HardwareSpec(
        name="ibm_eagle",
        n_qubits=127,
        native_gates=["id", "rz", "sx", "x", "cx", "ecr"],
        coupling_map=[(0, 1), (1, 2), (2, 3)] # Reduced mapping for brevity in MVP
    ),
    
    # Rigetti Aspen-M-3 (80 Qubits) - Octagon-Square topology
    "rigetti_aspen_m3": HardwareSpec(
        name="rigetti_aspen_m3",
        n_qubits=80,
        native_gates=["rx", "rz", "cz", "cp", "xy"],
        coupling_map=[] # To be populated
    ),
    
    # IonQ Aria-1 (25 Qubits) - Fully connected entrapment
    "ionq_aria": HardwareSpec(
        name="ionq_aria",
        n_qubits=25,
        native_gates=["gpi", "gpi2", "ms"],
        coupling_map=[(i, j) for i in range(25) for j in range(25) if i != j]
    ),
    
    # Generic Linear Chain (for testing)
    "linear_5": HardwareSpec(
        name="linear_5",
        n_qubits=5,
        native_gates=["h", "x", "y", "z", "cx"],
        coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)]
    ),
    
    # D-Wave Advantage (5000+ Qubits) - Pegasus topology
    "dwave_advantage": HardwareSpec(
        name="dwave_advantage",
        n_qubits=5640,
        native_gates=["rz", "rzz"],
        coupling_map=[] # Pegasus is complex, left as abstract coupling here
    ),

    # Virtual Backends
    "jax": HardwareSpec(
        name="jax",
        n_qubits=40, # High-RAM limit
        native_gates=["all"],
        coupling_map=[] # Fully connected virtual
    ),
    "cluster": HardwareSpec(
        name="cluster",
        n_qubits=100, # Distributed limit
        native_gates=["all"],
        coupling_map=[]
    )
}


def get_spec(name: str) -> Optional[HardwareSpec]:
    """Retrieve a hardware spec by name."""
    return SPECS.get(name.lower())


def list_devices() -> List[str]:
    """List all supported devices."""
    return list(SPECS.keys())
