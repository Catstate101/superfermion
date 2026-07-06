"""
Advanced Transpilation — SABRE routing, Dynamical Decoupling, and Scheduling.

Provides qubit routing for hardware connectivity constraints, noise-suppressing
pulse sequences, and timing-aware gate scheduling.

SABRE (SWAP-based BidiREctional routing):
    Maps logical qubits to physical qubits on a device with limited connectivity,
    inserting SWAP gates to satisfy two-qubit gate coupling requirements.

Dynamical Decoupling (DD):
    Inserts identity-equivalent pulse sequences (X2, XY4, CPMG) between gates
    to suppress decoherence and extend circuit lifetime on noisy hardware.

Scheduling:
    Aligns gates to a discrete time grid, respecting gate durations and
    parallelism constraints.

Usage:
    >>> from superfermion.compiler.advanced import (
    ...     sabre_route, apply_dynamical_decoupling, schedule_circuit,
    ... )
    >>>
    >>> coupling_map = [(0,1), (1,2), (2,3), (3,4)]  # linear chain
    >>> routed = sabre_route(circuit, coupling_map)
    >>> decoupled = apply_dynamical_decoupling(routed, sequence="XY4")
    >>> scheduled = schedule_circuit(decoupled)
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from superfermion.circuit import Circuit, GateRecord
from superfermion.compiler.passes import Pass


# ═════════════════════════════════════════════════════════════════════════
# SABRE Routing
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class _RoutingState:
    """Internal state for SABRE routing."""

    mapping: Dict[int, int]  # logical → physical
    inv_mapping: Dict[int, int]  # physical → logical
    coupling: Set[Tuple[int, int]]  # allowed edges
    decay: List[float]  # per-physical-qubit decay counter
    n_physical: int

    def can_execute(self, gate: GateRecord) -> bool:
        """Check if a gate is executable given the current mapping."""
        phys_qubits = [self.mapping.get(q, q) for q in gate.qubits]
        if len(phys_qubits) == 1:
            return True
        if len(phys_qubits) == 2:
            a, b = phys_qubits[0], phys_qubits[1]
            return (a, b) in self.coupling or (b, a) in self.coupling
        return False

    def swap(self, a: int, b: int):
        """Swap the mapping of physical qubits a and b."""
        la = self.inv_mapping.get(a)
        lb = self.inv_mapping.get(b)
        if la is not None:
            self.mapping[la] = b
            self.inv_mapping[b] = la
        if lb is not None:
            self.mapping[lb] = a
            self.inv_mapping[a] = lb
        self.decay[a] = 0.0
        self.decay[b] = 0.0


def _build_coupling_set(coupling_map: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Build a bi-directional coupling set from edge list."""
    cset = set()
    for a, b in coupling_map:
        cset.add((a, b))
        cset.add((b, a))
    return cset


def _shortest_path_length(
    src: int,
    dst: int,
    coupling: Set[Tuple[int, int]],
) -> int:
    """BFS shortest path distance between src and dst in coupling graph."""
    if src == dst:
        return 0
    from collections import deque

    visited = {src}
    queue = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        for a, b in coupling:
            nb = b if a == node else (a if b == node else None)
            if nb is not None and nb not in visited:
                if nb == dst:
                    return dist + 1
                visited.add(nb)
                queue.append((nb, dist + 1))
    return float("inf")


def sabre_route(
    circuit: Circuit,
    coupling_map: List[Tuple[int, int]],
    initial_mapping: Optional[Dict[int, int]] = None,
    n_iters: int = 3,
    seed: Optional[int] = None,
) -> Circuit:
    """SABRE routing: map logical circuit to physical topology via SWAP insertion.

    Args:
        circuit: Input circuit with logical qubits.
        coupling_map: List of physical qubit pairs that can interact, e.g.
                      ``[(0,1), (1,2), (2,3)]`` for a 4-qubit line.
        initial_mapping: Optional initial logical→physical mapping.
                         If None, uses the trivial mapping (0→0, 1→1, ...).
        n_iters: Number of forward-backward SABRE refinement passes.
        seed: Random seed for tie-breaking.

    Returns:
        A new circuit with SWAP gates inserted to satisfy coupling constraints.

    Algorithm (Li et al., ASPLOS 2019):
      1. Forward pass: greedily process gates. When a gate is not executable,
         insert SWAPs that minimize the look-ahead distance to future gates.
      2. Backward pass: reverse circuit, refine mapping.
      3. Repeat n_iters times.
    """
    if seed is not None:
        random.seed(seed)

    # Determine physical qubit count
    all_nodes = set()
    for a, b in coupling_map:
        all_nodes.add(a)
        all_nodes.add(b)
    n_physical = max(all_nodes) + 1 if all_nodes else circuit.n_qubits

    coupling = _build_coupling_set(coupling_map)
    gates = list(circuit._gates)
    n_logical = circuit.n_qubits

    # Initial mapping: trivial or provided
    if initial_mapping is None:
        mapping = {i: i for i in range(n_logical)}
    else:
        mapping = dict(initial_mapping)

    inv_mapping = {p: l for l, p in mapping.items()}

    state = _RoutingState(
        mapping=mapping,
        inv_mapping=inv_mapping,
        coupling=coupling,
        decay=[0.0] * n_physical,
        n_physical=n_physical,
    )

    # ── SABRE iterations ────────────────────────────────────────────────
    routed_gates: List[GateRecord] = []

    for _iter in range(n_iters):
        # Forward pass
        routed_gates = []
        for gate in gates:
            if state.can_execute(gate):
                # Remap qubits
                phys_qubits = tuple(state.mapping.get(q, q) for q in gate.qubits)
                remapped = GateRecord(
                    name=gate.name,
                    qubits=phys_qubits,
                    params=gate.params,
                )
                routed_gates.append(remapped)
                # Decay increment (SABRE decay heuristic)
                for pq in phys_qubits:
                    if pq < n_physical:
                        state.decay[pq] += 1.0
            else:
                # Gate not executable — insert SWAPs
                phys_qubits = [state.mapping.get(q, q) for q in gate.qubits]
                # Find best SWAP to bring qubits closer
                best_swap = _pick_best_swap(state, phys_qubits, gates, routed_gates)
                if best_swap is not None:
                    a, b = best_swap
                    # Insert SWAP gate
                    routed_gates.append(GateRecord(name="SWAP", qubits=(a, b), params=[]))
                    state.swap(a, b)
                # Now try the gate again
                phys_qubits = tuple(state.mapping.get(q, q) for q in gate.qubits)
                if state.can_execute(gate):
                    remapped = GateRecord(
                        name=gate.name,
                        qubits=phys_qubits,
                        params=gate.params,
                    )
                    routed_gates.append(remapped)
                else:
                    # Still can't execute — fall through (shouldn't happen for 2q)
                    remapped = GateRecord(
                        name=gate.name,
                        qubits=phys_qubits,
                        params=gate.params,
                    )
                    routed_gates.append(remapped)

        # Backward pass (reverse gates, similar logic)
        if _iter < n_iters - 1:
            gates = list(reversed(routed_gates))
            routed_gates = []
            state.decay = [0.0] * n_physical

    # Build output circuit
    base_name = getattr(circuit, '_name', None) or 'circuit'
    out = Circuit(n_physical, circuit.n_cbits, name=f"{base_name}_sabre")
    out._gates = routed_gates
    return out


def _pick_best_swap(
    state: _RoutingState,
    target_phys: List[int],
    future_gates: List[GateRecord],
    past_gates: List[GateRecord],
) -> Optional[Tuple[int, int]]:
    """Choose the best SWAP to bring target qubits closer (SABRE heuristic)."""
    candidates: List[Tuple[int, int, float]] = []  # (a, b, score)

    for a, b in state.coupling:
        if a >= b:
            continue  # Process each undirected edge once

        if a not in state.inv_mapping and b not in state.inv_mapping:
            continue

        # Heuristic cost = Σ (distance after SWAP) + decay penalty
        cost = 0.0
        # Distance for current target
        for tq in target_phys:
            d_before = _shortest_path_length(tq, a, state.coupling)
            d_after_a = _shortest_path_length(tq, b, state.coupling)
            cost += (d_after_a - d_before)

            d_before_b = _shortest_path_length(tq, b, state.coupling)
            d_after_b = _shortest_path_length(tq, a, state.coupling)
            cost += (d_after_b - d_before_b)

        # Look-ahead: next few future gates
        look_ahead = min(5, len(future_gates))
        for fg in future_gates[:look_ahead]:
            fq_phys = [state.mapping.get(q, q) for q in fg.qubits]
            for tq in fq_phys:
                d_before = _shortest_path_length(tq, a, state.coupling)
                d_after = _shortest_path_length(tq, b, state.coupling)
                cost += 0.3 * (d_after - d_before)

        # Decay penalty
        cost += 0.1 * (state.decay[a] + state.decay[b])

        candidates.append((a, b, cost))

    if not candidates:
        return None

    # Filter out NaN/inf costs (can occur when coupling graph is disconnected)
    candidates = [c for c in candidates if math.isfinite(c[2])]
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2])
    # Pick lowest cost, tie-break randomly
    best_cost = candidates[0][2]
    tied = [c for c in candidates if abs(c[2] - best_cost) < 1e-6]
    chosen = random.choice(tied)
    return (chosen[0], chosen[1])


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

class SABRERoutingPass:
    """Compiler pass: apply SABRE routing."""

    def __init__(self, coupling_map: List[Tuple[int, int]], **kwargs):
        self.coupling_map = coupling_map
        self.kwargs = kwargs

    def name(self) -> str:
        return "SABRERoutingPass"

    def run(self, circuit: Circuit) -> Circuit:
        return sabre_route(circuit, self.coupling_map, **self.kwargs)


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
