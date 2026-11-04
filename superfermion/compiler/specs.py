"""
Hardware Specifications — Metadata for quantum hardware targets.

Provides connectivity graphs and native gate sets for compilation targets.
Topologies are generated from Rust ``CouplingMap`` — no hardcoded edge lists.
Named presets (``"ibm_eagle"``, ``"ionq_aria"``) are convenience aliases for
generic topology shapes (heavy-hex, all-to-all, grid, linear).

.. warning::

   "generic topology shapes" is literal: the presets describe the *shape*, not
   a vendor calibration.  ``CouplingMap.heavy_hex`` in particular is a square
   grid with vertical links on alternate columns, **not** a heavy-hex lattice:
   measured on ``n_qubits=127`` it has 174 undirected edges and the degree
   histogram ``{1: 10, 2: 56, 3: 18, 4: 43}``, while a real 127-qubit IBM
   heavy-hex (e.g. ``FakeBrisbane``/``FakeSherbrooke``) has 144 edges and
   ``{1: 2, 2: 89, 3: 36}`` -- a heavy-hex never has a degree-4 node, because
   every qubit there binds at most two hexagons.  ``ibm_eagle``/``ibm_heron``
   therefore give you a *constrained, connected* layout of roughly the right
   size for depth and routing studies; they do not reproduce the coupling map
   of a named device, and SWAP counts taken from them are not comparable to
   that device's.  Use :attr:`HardwareSpec.degree_histogram` or
   :attr:`HardwareSpec.max_degree` to check any topology claim before building
   on it.

   The gate sets are generic in the same way.  ``ibm_eagle`` and ``ibm_heron``
   both declare ``ecr`` as their two-qubit native, which is the vocabulary the
   compiler targets -- every emitted ``CX`` is rewritten into an exact
   ``{rz, sx, x, ecr}`` sequence by
   :mod:`superfermion.compiler.native_lowering`.  Measured against Qiskit's
   fake backends: all six of ``ibm_eagle``'s native names appear in the
   vocabulary of every 127-qubit snapshot (``FakeBrisbane`` and siblings)
   checked, while a 133-qubit Heron device (``FakeTorino``) declares
   ``cz`` and has no ``ecr`` at all.  So ``ibm_heron`` is a
   Heron-*sized* device with an Eagle-style gate set: ``native_gates`` answers
   "what does this preset compile to", not "what does a Heron board implement".
   A circuit that ``circuit_gates_outside(circuit, native=True)`` accepts is
   therefore clean for the preset and says nothing about the physical device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

#: Gates the compiler emits regardless of the target basis.  ``GateFusionPass``
#: is the last pass of the Rust pipeline and merges every run of one-qubit
#: gates into a single ``U(theta, phi, lambda)``, which is a standard gate of
#: OpenQASM 3 and is translated to ``rz``/``ry`` by any vendor front end.  It is
#: listed in every ``basis_gates`` so that "what the compiler emits" and "what
#: the spec allows" never disagree (HW-1, HW-3).
FUSION_OUTPUT_GATES: Tuple[str, ...] = ("u",)

#: Pauli gates the compiler also leaves in place on every target.  Emission
#: comes from two places, and both were verified against the compiled unitary
#: (overlap 1.000000000 on ``ibm_eagle``, ``ibm_heron``, ``rigetti_ankaa``,
#: ``ionq_aria`` and ``linear_5``): ``UnitaryDecompositionPass`` synthesises a
#: two-qubit block into a ``CX``/``ECR``/``CZ`` plus ``{X, Y, Z, U}``, and
#: :func:`superfermion.compiler.native_lowering.cx_via_ecr` uses an ``X``
#: conjugation.  ``X`` is already native on the IBM presets, so only the names
#: a target does not implement show up in :attr:`HardwareSpec.emulated`.
#: Declaring them here is what makes the ``basis_gates`` promise -- "what the
#: compiler may emit" -- true instead of merely asserted (HW-1).
PAULI_OUTPUT_GATES: Tuple[str, ...] = ("x", "y", "z")

#: Every gate name the emitter can produce on its own, whatever the target.
EMITTER_OUTPUT_GATES: Tuple[str, ...] = FUSION_OUTPUT_GATES + PAULI_OUTPUT_GATES


def _lowered(names) -> Set[str]:
    """Lower-case gate names as a set, for membership tests."""
    return {str(name).strip().lower() for name in (names or ())}


@dataclass
class HardwareSpec:
    """Metadata for a specific quantum device or compilation target.

    ``native_gates`` is what the device implements. ``basis_gates`` is what the
    compiler may *emit*, and defaults to ``native_gates`` so that the two can
    never contradict each other (HW-1).  When the compiler cannot translate a
    rotation all the way down -- ``RX`` on a ``{x, sx, rz}`` device, or any of
    ``CX``/``RX``/``RZ`` on an IonQ target whose native set is ``gpi``/``gpi2``/
    ``ms`` -- the extra names are listed in ``basis_gates`` and reported by
    :attr:`emulated`, instead of being missing from a hard-coded default that
    had nothing to do with the device.

    ``compile_basis`` is the set handed to the Rust basis translator.  It is
    ``native_gates`` (plus, for ``ibm_heron``, ``cx``) and only differs when a
    translator arm has to be steered onto a correct path; the emitted circuit
    is rewritten back to the native set by
    :mod:`superfermion.compiler.native_lowering` afterwards.
    """
    name: str
    n_qubits: int
    native_gates: List[str]
    coupling_map: List[Tuple[int, int]] = field(default_factory=list)
    basis_gates: Optional[List[str]] = None
    compile_basis: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.compile_basis is None:
            self.compile_basis = list(self.native_gates)
        else:
            self.compile_basis = list(self.compile_basis)
        # Anything the device implements can obviously also be compiled.
        for gate in self.native_gates:
            if gate.strip().lower() not in _lowered(self.compile_basis):
                self.compile_basis.append(gate)

        if self.basis_gates is None:
            self.basis_gates = list(self.compile_basis)
        else:
            self.basis_gates = list(self.basis_gates)
        # ``GateFusionPass`` is the last pass of the Rust pipeline and always
        # finishes with ``U``, and both the decomposition pass and the ECR
        # lowering emit Paulis.  Declaring that vocabulary is what makes
        # ``basis_gates`` mean "what the compiler may emit" (HW-1).
        for gate in tuple(EMITTER_OUTPUT_GATES):
            if gate.strip().lower() not in _lowered(self.basis_gates):
                self.basis_gates.append(gate)

    @property
    def is_fully_connected(self) -> bool:
        """Returns True if every qubit can talk to every other qubit."""
        expected = self.n_qubits * (self.n_qubits - 1)
        return len(self.coupling_map) == expected

    @property
    def emulated(self) -> List[str]:
        """Gate names the compiler emits that the device does not implement.

        Empty for every target the compiler can translate all the way down.
        """
        native = {gate.lower() for gate in self.native_gates}
        return sorted(
            {gate.lower() for gate in (self.basis_gates or [])} - native
        )

    def is_native(self, gate: str) -> bool:
        """Whether ``gate`` is in the device's native set."""
        native = {name.lower() for name in self.native_gates}
        return "all" in native or gate.lower() in native

    def is_emittable(self, gate: str) -> bool:
        """Whether ``gate`` may appear in compiled output for this target."""
        basis = {name.lower() for name in (self.basis_gates or [])}
        return "all" in basis or gate.lower() in basis

    @property
    def degree_histogram(self) -> Dict[int, int]:
        """How many qubits have each graph degree (isolated qubits counted as 0).

        A quick shape fingerprint: a heavy-hex has degrees ``{2, 3}`` only, a
        linear chain ``{1, 2}``, an all-to-all ``{n-1}``.  Anything with a
        degree-4 node is not a heavy-hex, whatever it is called.
        """
        counts: Dict[int, int] = {}
        for _node, neighbours in self._adjacency().items():
            degree = len(neighbours)
            counts[degree] = counts.get(degree, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def max_degree(self) -> int:
        """Largest number of couplers on any qubit (0 for an empty map)."""
        degrees = [len(neighbours) for neighbours in self._adjacency().values()]
        return max(degrees) if degrees else 0

    def _adjacency(self) -> Dict[int, set]:
        """Undirected neighbour sets, ignoring duplicate and reversed edges."""
        adj: Dict[int, set] = {}
        for a, b in self.coupling_map:
            ia, ib = int(a), int(b)
            if ia == ib:
                continue
            adj.setdefault(ia, set()).add(ib)
            adj.setdefault(ib, set()).add(ia)
        for qubit in range(self.n_qubits):
            adj.setdefault(qubit, set())
        return adj

    def circuit_gates_outside(self, circuit, *, native: bool = False) -> List[str]:
        """Gate names in ``circuit`` that the target does not allow.

        ``native=False`` (default) checks against ``basis_gates`` — what the
        compiler promises to emit. ``native=True`` checks against
        ``native_gates`` and therefore also reports the gates listed in
        :attr:`emulated`.
        """
        allowed = self.is_native if native else self.is_emittable
        circuit._ensure_gates()
        offenders = {
            record.name.upper() for record in circuit._gates
            if not allowed(record.name)
        }
        return sorted(offenders)


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
        # ``cx_decomp`` takes a numerically wrong arm when ECR is available but
        # CX is not (overlap 0.707106781), so the translator is handed CX and
        # :mod:`superfermion.compiler.native_lowering` rewrites the emitted CX
        # back into an exact {rz, sx, x, ecr} sequence afterwards.
        compile_basis=["id", "rz", "sx", "x", "cx", "ecr"],
        # ``cx`` is translator-internal only; the post-pass rewrites it back to
        # ECR, so the compiler never *emits* a CX on a Heron.  ``basis_gates``
        # therefore stays the native set (plus the fusion output ``u``).
        basis_gates=["id", "rz", "sx", "x", "ecr"],
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
        # No IonQ translation exists in the compiler yet: it emits the
        # superconducting primitives below. Declared so that
        # ``circuit_gates_outside()`` reports the truth instead of comparing
        # against an unrelated hard-coded default (HW-1, HW-3).
        basis_gates=["gpi", "gpi2", "ms", "cx", "rx", "rz"],
        coupling_map=_coupling_edges("all_to_all", 25),
    ),
    "ionq_forte": HardwareSpec(
        name="ionq_forte",
        n_qubits=36,
        native_gates=["gpi", "gpi2", "ms"],
        basis_gates=["gpi", "gpi2", "ms", "cx", "rx", "rz"],
        coupling_map=_coupling_edges("all_to_all", 36),
    ),
    "linear_5": HardwareSpec(
        name="linear_5",
        n_qubits=5,
        # {H, X, Y, Z, CX} is a *Clifford* set: it cannot express a rotation by
        # an arbitrary angle at all, so continuous rotations come out as RX/RZ.
        native_gates=["h", "x", "y", "z", "cx"],
        basis_gates=["h", "x", "y", "z", "cx", "rx", "rz"],
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
