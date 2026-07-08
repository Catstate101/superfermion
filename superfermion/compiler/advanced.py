"""
Advanced Transpilation — Dynamical Decoupling, Scheduling, and Pauli Twirling.

These are **standalone utilities**, not part of the ``sf.compile()`` pipeline.
Routing is handled entirely by the Rust ``sf-router`` crate via ``_sf_core``.

Dynamical Decoupling (DD):
    Inserts identity-equivalent pulse sequences (X2, XY4, CPMG) between gates
    to suppress decoherence and extend circuit lifetime on noisy hardware.

Scheduling:
    Aligns gates to a discrete time grid, respecting gate durations and
    parallelism constraints.

Pauli Twirling:
    Converts coherent errors into stochastic Pauli errors by inserting
    random Pauli sandwiches around two-qubit gates.

Usage:
    >>> from superfermion.compiler.advanced import (
    ...     apply_dynamical_decoupling, schedule_circuit,
    ... )
    >>> decoupled = apply_dynamical_decoupling(compiled, sequence="XY4")
    >>> scheduled = schedule_circuit(decoupled)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from superfermion.circuit import Circuit, GateRecord
from superfermion.compiler.passes import Pass


# ═════════════════════════════════════════════════════════════════════════
# Dynamical Decoupling
# ═════════════════════════════════════════════════════════════════════════

# Standard DD sequences (identity-equivalent pulse trains)
DD_SEQUENCES = {
    "X2": ["X", "X"],  # Simplest: two X gates = I
    "XY4": ["X", "Y", "X", "Y"],  # Cancels σ_x, σ_y, σ_z
    "XY8": ["X", "Y", "X", "Y", "Y", "X", "Y", "X"],  # Higher-order
    "CPMG": ["X"],  # Single X, repeated with spacing
    "KDD": ["X", "Y", "Y", "X"],  # Knill DD
    "XY16": [
        "X", "Y", "X", "Y", "Y", "X", "Y", "X",
        "X", "Y", "X", "Y", "Y", "X", "Y", "X",
    ],
}


def apply_dynamical_decoupling(
    circuit: Circuit,
    sequence: str = "XY4",
    spacing: int = 1,
    qubits: Optional[List[int]] = None,
) -> Circuit:
    """Insert dynamical decoupling sequences between gates on idle qubits.

    Args:
        circuit: Input circuit.
        sequence: DD sequence name (``"X2"``, ``"XY4"``, ``"XY8"``, ``"CPMG"``,
                  ``"KDD"``, ``"XY16"``).
        spacing: Insert DD every N non-DD gates per qubit.
        qubits: Qubits to protect. If None, applies to all qubits.

    Returns:
        Circuit with DD sequences inserted.
    """
    if sequence.upper() not in DD_SEQUENCES:
        raise ValueError(
            f"Unknown DD sequence '{sequence}'. Available: {list(DD_SEQUENCES)}"
        )

    seq = DD_SEQUENCES[sequence.upper()]
    target_qubits = qubits or list(range(circuit.n_qubits))

    gates = circuit._gates
    new_gates: List[GateRecord] = []

    # Per-qubit gate counters for spacing
    gate_counts = {q: 0 for q in target_qubits}

    for gate in gates:
        new_gates.append(gate)

        # Track which qubits were touched
        touched = set(gate.qubits)

        # Insert DD on idle qubits
        for q in target_qubits:
            if q in touched:
                gate_counts[q] = 0  # Reset counter
            else:
                gate_counts[q] += 1
                if gate_counts[q] >= spacing:
                    # Insert DD sequence on this qubit
                    for pulse in seq:
                        if pulse == "X":
                            new_gates.append(
                                GateRecord(name="X", qubits=(q,), params=[])
                            )
                        elif pulse == "Y":
                            new_gates.append(
                                GateRecord(name="Y", qubits=(q,), params=[])
                            )
                    gate_counts[q] = 0

    base_name = getattr(circuit, '_name', None) or 'circuit'
    out = Circuit(
        circuit.n_qubits,
        circuit.n_cbits,
        name=f"{base_name}_dd_{sequence}",
    )
    out._gates = new_gates
    return out


# ═════════════════════════════════════════════════════════════════════════
# Scheduling
# ═════════════════════════════════════════════════════════════════════════

# Default gate durations (in time units, relative to single-qubit gate = 1)
DEFAULT_DURATIONS: Dict[str, float] = {
    "H": 1.0, "X": 1.0, "Y": 1.0, "Z": 1.0,
    "S": 1.0, "SDG": 1.0, "T": 1.0, "TDG": 1.0,
    "RX": 1.0, "RY": 1.0, "RZ": 1.0, "P": 1.0,
    "U": 2.0, "SX": 1.0,
    "CX": 3.0, "CNOT": 3.0, "CZ": 3.0, "CY": 3.0,
    "SWAP": 6.0, "TOFFOLI": 10.0,
    "CP": 4.0, "CH": 4.0,
    "MEASURE": 5.0,
}

# Default gate dependency overlap rules
# "CX(q0, q1)": touches qubits {q0, q1}, cannot overlap if qubit sets intersect


@dataclass
class ScheduledGate:
    """A gate placed on the timeline."""

    gate: GateRecord
    start_time: float
    duration: float
    qubits: Tuple[int, ...]


@dataclass
class Schedule:
    """Scheduled circuit with gates assigned to time slots."""

    gates: List[ScheduledGate] = field(default_factory=list)
    total_duration: float = 0.0
    n_qubits: int = 0
    critical_path: List[str] = field(default_factory=list)  # gate names along critical path

    @property
    def depth(self) -> int:
        """Number of time layers (discrete depth)."""
        return len(self.gates)

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    def timeline(self) -> List[Dict[str, Any]]:
        """Return timeline as list of {start, duration, gate, qubits} dicts."""
        return [
            {
                "start": g.start_time,
                "duration": g.duration,
                "gate": g.gate.name,
                "qubits": list(g.qubits),
            }
            for g in self.gates
        ]


def schedule_circuit(
    circuit: Circuit,
    gate_durations: Optional[Dict[str, float]] = None,
    alignment: float = 1.0,
) -> Schedule:
    """ASAP (As-Soon-As-Possible) scheduling of circuit gates.

    Args:
        circuit: Input circuit.
        gate_durations: Dict mapping gate name → duration. Uses defaults if None.
        alignment: Time grid unit (gates are aligned to multiples of this).

    Returns:
        Schedule with gates assigned to time slots.
    """
    durations = DEFAULT_DURATIONS | (gate_durations or {})

    # Per-qubit availability times
    qubit_free_time: Dict[int, float] = {q: 0.0 for q in range(circuit.n_qubits)}

    scheduled: List[ScheduledGate] = []

    for gate in circuit._gates:
        dur = durations.get(gate.name.upper(), 1.0)

        # Earliest start = max(qubit_free_time for all involved qubits)
        involved = list(gate.qubits)
        start = max(qubit_free_time.get(q, 0.0) for q in involved)

        # Align to grid
        start = np.ceil(start / alignment) * alignment

        scheduled.append(
            ScheduledGate(
                gate=gate,
                start_time=start,
                duration=dur,
                qubits=gate.qubits,
            )
        )

        # Update qubit availability
        finish = start + dur
        for q in involved:
            qubit_free_time[q] = finish

    total_duration = max(qubit_free_time.values()) if qubit_free_time else 0.0

    # Compute critical path (gates on the longest dependency chain)
    critical = _compute_critical_path(scheduled)

    return Schedule(
        gates=scheduled,
        total_duration=total_duration,
        n_qubits=circuit.n_qubits,
        critical_path=critical,
    )


def _compute_critical_path(scheduled: List[ScheduledGate]) -> List[str]:
    """Identify gates along the critical (longest) path."""
    if not scheduled:
        return []

    # Simple approximation: scan for gates whose finish time equals max
    max_t = max(g.start_time + g.duration for g in scheduled)
    critical = []

    for g in scheduled:
        if abs(g.start_time + g.duration - max_t) < 1e-6:
            critical.append(g.gate.name)

    return critical


# ═════════════════════════════════════════════════════════════════════════
# Compiler Pass wrappers
# ═════════════════════════════════════════════════════════════════════════

class DynamicalDecouplingPass:
    """Compiler pass: insert dynamical decoupling sequences."""

    def __init__(self, sequence: str = "XY4", **kwargs):
        self.sequence = sequence
        self.kwargs = kwargs

    def name(self) -> str:
        return "DynamicalDecouplingPass"

    def run(self, circuit: Circuit) -> Circuit:
        return apply_dynamical_decoupling(circuit, self.sequence, **self.kwargs)


class SchedulingPass:
    """Compiler pass: schedule gates to a time grid."""

    def __init__(self, gate_durations=None, alignment: float = 1.0):
        self.durations = gate_durations
        self.alignment = alignment

    def name(self) -> str:
        return "SchedulingPass"

    def run(self, circuit: Circuit) -> Circuit:
        """Note: scheduling returns a Schedule, not a Circuit.
        For pipeline compatibility, returns circuit unchanged.
        Use schedule_circuit() directly for the Schedule object.
        """
        return circuit


# ═════════════════════════════════════════════════════════════════════════
# Pauli Twirling
# ═════════════════════════════════════════════════════════════════════════

# Pauli twirling frame pairs for CNOT:
# For a CNOT gate G, we find (P_before, P_after) such that P_after · G = G · P_before.
# 14 unique pairs verified numerically.
_CNOT_TWIRL_PAIRS = [
    ("I", "I", "I", "I"),
    ("I", "X", "I", "X"),
    ("I", "Z", "Z", "Z"),
    ("I", "Y", "Z", "Y"),
    ("X", "I", "X", "X"),
    ("X", "X", "X", "I"),
    ("X", "Y", "Y", "Z"),
    ("Z", "I", "Z", "I"),
    ("Z", "X", "Z", "X"),
    ("Z", "Z", "I", "Z"),
    ("Z", "Y", "I", "Y"),
    ("Y", "I", "Y", "X"),
    ("Y", "X", "Y", "I"),
    ("Y", "Z", "X", "Y"),
]

# Pauli twirling frame pairs for CZ:
# 14 unique pairs verified numerically.
_CZ_TWIRL_PAIRS = [
    ("I", "I", "I", "I"),
    ("I", "X", "Z", "X"),
    ("I", "Z", "I", "Z"),
    ("I", "Y", "Z", "Y"),
    ("X", "I", "X", "Z"),
    ("X", "X", "Y", "Y"),
    ("X", "Z", "X", "I"),
    ("Z", "I", "Z", "I"),
    ("Z", "X", "I", "X"),
    ("Z", "Z", "Z", "Z"),
    ("Z", "Y", "I", "Y"),
    ("Y", "I", "Y", "Z"),
    ("Y", "Z", "Y", "I"),
    ("Y", "Y", "X", "X"),
]

# Gate -> Pauli application helper
_PAULI_GATE = {"I": None, "X": "x", "Y": "y", "Z": "z"}


def _apply_pauli(circuit: Circuit, pauli: str, qubit: int):
    """Apply a single Pauli gate to a qubit on the circuit (mutates in place)."""
    gate_name = _PAULI_GATE.get(pauli)
    if gate_name is not None:
        getattr(circuit, gate_name)(qubit)


class PauliTwirlingPass:
    """Compiler pass: apply Pauli twirling to two-qubit gates.

    Pauli twirling converts coherent errors into stochastic Pauli errors
    by inserting random Pauli sandwiches around each two-qubit gate.
    The inserted pairs satisfy P_after · G = G · P_before, preserving
    the logical circuit behavior.

    Args:
        seed: Random seed for reproducibility.
        gate_types: Two-qubit gates to twirl (default: ["CNOT", "CX", "CZ"]).
    """

    def __init__(self, seed: Optional[int] = None, gate_types: Optional[List[str]] = None):
        self.seed = seed
        self.gate_types = [g.upper() for g in (gate_types or ["CNOT", "CX", "CZ"])]
        self._rng = random.Random(seed)

    def name(self) -> str:
        return "PauliTwirlingPass"

    def run(self, circuit: Circuit) -> Circuit:
        new_gates: List[GateRecord] = []
        for gate in circuit._gates:
            name = gate.name.upper()
            if name in self.gate_types and len(gate.qubits) == 2:
                q0, q1 = gate.qubits[0], gate.qubits[1]

                # Select twirling frame
                if name in ("CZ",):
                    pairs = _CZ_TWIRL_PAIRS
                else:
                    pairs = _CNOT_TWIRL_PAIRS

                p1_b, p2_b, p1_a, p2_a = self._rng.choice(pairs)

                # Apply Pauli before: P1_b ⊗ P2_b
                _apply_pauli_to_list(new_gates, p1_b, q0)
                _apply_pauli_to_list(new_gates, p2_b, q1)

                # Insert the original gate
                new_gates.append(gate)

                # Apply Pauli after: P1_a ⊗ P2_a
                _apply_pauli_to_list(new_gates, p1_a, q0)
                _apply_pauli_to_list(new_gates, p2_a, q1)
            else:
                new_gates.append(gate)

        out = Circuit(circuit.n_qubits, circuit.n_cbits)
        out._gates = new_gates
        return out


def _apply_pauli_to_list(gate_list: List[GateRecord], pauli: str, qubit: int):
    """Append a Pauli gate record to a list."""
    gate_name = _PAULI_GATE.get(pauli)
    if gate_name is not None:
        gate_list.append(GateRecord(name=gate_name.upper(), qubits=(qubit,), params=[]))
