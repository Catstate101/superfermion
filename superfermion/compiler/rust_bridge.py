"""
Rust Compilation Bridge — PyO3-powered native compilation pipeline.
====================================================================

Provides `compile_rust()` which routes the entire compilation pipeline
through the Rust `sf-compiler` crate for 10-100x speedup over Python.

Usage:
    >>> from superfermion.compiler.rust_bridge import compile_rust
    >>> compiled = compile_rust(circuit, level=2)
"""
from __future__ import annotations


from __future__ import annotations
import re
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from superfermion.circuit import Circuit

# Pre-import simplify_clifford at module level to avoid triggering
# numpy/stabilizer imports inside benchmark measurement contexts.
from superfermion.backends.stabilizer import simplify_clifford as _simplify_clifford

# ── Gate-name → SF Circuit method mapping (fast QASM parser) ──────────
_GATE_MAP: dict[str, str] = {
    "h": "h", "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",
    "x": "x", "y": "y", "z": "z", "id": "id",
    "cx": "cx", "cz": "cz", "cy": "cy", "swap": "swap",
    "rx": "rx", "ry": "ry", "rz": "rz", "sx": "sx",
    "p": "p", "u1": "p", "u2": None, "u3": None,
    "barrier": None, "measure": None, "reset": None,
}

# Pre-compiled regex: gate [optional(params)] q[idx] [, q[idx]] ;
_QASM_LINE_RE = re.compile(
    r"^\s*(\w+)\s*(?:\(([^)]*)\))?\s+q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?\s*;"
)

# Record names that are not unitary gates, so a 2-qubit record with one of
# these names is not a connectivity violation (measure/reset touch one wire
# today, but the list is checked before the arity test for safety).
_NON_UNITARY_RECORDS = frozenset({"measure", "measure_all", "barrier", "reset"})


def _fast_qasm_to_circuit(qasm: str, n_qubits: int) -> "Circuit | None":
    """Parse the compiler's predictable QASM3 output into an SF Circuit.

    Handles standard and parameterized gates: h, s, sdg, x, y, z, cx, cz,
    swap, rx, ry, rz, sx, p.  Returns None if any gate is unsupported,
    so the caller can fall back to the general-purpose parser.
    """
    from superfermion.circuit import Circuit

    c = Circuit(n_qubits)
    # Use batch append for performance
    records = []

    for line in qasm.splitlines():
        m = _QASM_LINE_RE.match(line)
        if m is None:
            continue  # skip headers, blank lines, bit decls, etc.

        gname = m.group(1).lower()
        method = _GATE_MAP.get(gname, _MISSING)
        if method is _MISSING:
            return None  # unknown gate — fall back to general parser
        if method is None:
            continue  # barrier, measure, reset — skip silently

        params_str = m.group(2)
        q0 = int(m.group(3))
        q1_str = m.group(4)
        q1 = int(q1_str) if q1_str is not None else None

        if params_str is not None:
            # Parameterized gate: rx(theta), rz(theta), sx, p(theta), etc.
            # The compiler emits angles as e.g. "0.5*pi", "-0.5*pi", "pi"
            try:
                angle = _eval_qasm_angle(params_str.strip())
            except (ValueError, SyntaxError):
                return None

            if method == "rx":
                records.append(_make_gate("RX", [q0], [angle]))
            elif method == "ry":
                records.append(_make_gate("RY", [q0], [angle]))
            elif method == "rz":
                records.append(_make_gate("RZ", [q0], [angle]))
            elif method == "sx":
                records.append(_make_gate("SX", [q0]))
            elif method == "p":
                records.append(_make_gate("P", [q0], [angle]))
            else:
                return None
        else:
            # Non-parameterized gate
            if q1 is not None:
                records.append(_make_gate(method.upper(), [q0, q1]))
            else:
                records.append(_make_gate(method.upper(), [q0]))

    if records:
        c._gates.extend(records)
    return c


_MISSING = object()


def _make_gate(name: str, qubits: list, params: list | None = None) -> "GateRecord":
    """Create a GateRecord."""
    from superfermion.circuit import GateRecord
    return GateRecord(name=name, qubits=qubits, params=params or [])


def _eval_qasm_angle(s: str) -> float:
    """Evaluate a QASM angle expression like '0.5*pi', '-pi/2', 'pi'."""
    import math
    s = s.replace(" ", "")
    # Replace 'pi' with math.pi
    s = s.replace("pi", str(math.pi))
    # Safe eval — only contains digits, operators, and math.pi
    return float(eval(s, {"__builtins__": {}}, {}))


def compile_rust(
    circuit: "Circuit",
    level: int = 1,
    target: Optional[object] = None,
    pre_simplified: bool = False,
    skip_routing: bool = False,
    twirl_seed: Optional[int] = None,
) -> "Circuit":
    """Compile a circuit using the Rust-native compilation pipeline.

    This is the **sole** compilation path. The Rust ``sf-compiler`` crate
    runs gate cancellation, SWAP decomposition, superconducting basis
    translation (when SX is in the native gate set), rotation merging,
    SABRE routing (when connectivity is provided), and Pauli twirling
    (at level >= 2).

    Args:
        circuit: SF Circuit to compile.
        level: Optimization level (0-2). Level 2 includes Pauli twirling.
        target: Optional ``HardwareSpec`` with ``coupling_map``, ``native_gates``,
                and ``n_qubits`` attributes.
        pre_simplified: Skip Clifford pre-simplification if True.
        skip_routing: Assume ``circuit`` already satisfies the target's
                connectivity. The Rust router is not run, so the wire
                permutation it would have applied is not applied either
                (see :mod:`superfermion.compiler.layout`).
        twirl_seed: Seed for the level >= 2 Pauli twirl. ``None`` draws a
                fresh thread-RNG seed, which makes the compiled gate list
                vary between calls of the same input; the unitary does not
                vary (measured at 1e-16).

    Returns:
        Compiled SF Circuit, on the same number of qubits as ``circuit``
        (or more, only if a gate was emitted on a higher wire).

    Raises:
        ImportError: If ``_sf_core`` is not available (Rust extension required).
        RuntimeError: If the Rust compiler encounters an unsupported gate.
    """
    from superfermion._sf_core import Compiler

    if target is not None:
        # ``compile_basis`` is what the Rust translator is allowed to emit.  It
        # equals ``native_gates`` except where a translator arm is steered onto
        # a correct path (see ``superfermion.compiler.native_lowering``).
        native_gates = list(
            getattr(target, "compile_basis", None)
            or getattr(target, "native_gates", [])
        )
        coupling_map = list(getattr(target, "coupling_map", []))
        n_qubits = getattr(target, "n_qubits", circuit.n_qubits)
        name = getattr(target, "name", "target")
    else:
        native_gates = []
        coupling_map = []
        n_qubits = circuit.n_qubits
        name = "none"

    # Wiring map for the connectivity check at the end of this function.
    # ``route_with_layout`` relabels the circuit's wires to region-local
    # positions and records the physical wire each position sits on in
    # ``initial_layout``; without that report the wire labels are physical
    # already.  Read it before any pass can replace the circuit object.
    position_to_physical = None
    try:
        from superfermion.compiler.layout import get_layout as _get_layout

        _layout = _get_layout(circuit)
    except Exception:  # pragma: no cover - layout module is optional
        _layout = None
    if _layout and _layout.get("initial_layout"):
        position_to_physical = {
            int(i): int(p) for i, p in enumerate(_layout["initial_layout"])
        }

    # What the target declares, kept even when routing is suppressed here: the
    # compiled output is validated against it below either way.
    declared_connectivity = list(coupling_map)

    if skip_routing:
        coupling_map = []

    # Clifford pre-simplification: reduces gate count via Aaronson-Gottesman
    # tableau decomposition. Only applied when there is no coupling map,
    # because the synthesis produces long-range CNOTs that destroy locality
    # and make routing harder (or impossible) on constrained topologies.
    # A pre-routed circuit is already local, so it keeps the same exemption.
    if not pre_simplified and not coupling_map and not skip_routing:
        simplified = _simplify_clifford(circuit)
        if simplified is not None:
            circuit = simplified

    # Repairs for the two numerically wrong ``BasisTranslationPass`` arms
    # (``Ry`` on an SX device, ``cx_decomp`` on an ECR device).  No-ops unless
    # the target actually selects one of them.
    from superfermion.compiler.native_lowering import (
        finalize as _finalize,
        prepare as _prepare,
    )

    circuit, lowering = _prepare(circuit, target)

    dag = circuit.to_ir()

    compiler = Compiler(
        name=name,
        native_gates=native_gates,
        n_qubits=n_qubits,
        connectivity=coupling_map,
        optimization_level=level,
        twirl_seed=twirl_seed,
    )

    compiled_dag = compiler.compile(dag)

    from superfermion.circuit import Circuit as CircuitCls
    records = compiled_dag.to_gate_records()

    # ``Dag::to_gate_records()`` has no QASM name for ``OpType::Unitary`` and
    # reports it as ``"unknown"`` with the matrix dropped.  A circuit rebuilt
    # from such a record looks fine until the first ``to_ir()``/``run()``, which
    # raises ``ValueError: Unknown gate: 'unknown'``.  Refuse the hand-off here
    # instead of returning an unusable object (the armless opcodes are lowered
    # in ``native_lowering`` precisely so that this never fires).
    nameless = sorted(
        {tuple(sorted(int(q) for q in qubits)) for name, qubits, _p in records
         if name.strip().lower() in ("unknown", "")}
    )
    if nameless:
        raise RuntimeError(
            f"the Rust compiler returned {len(nameless)} gate(s) it cannot name "
            f"(OpType::Unitary survives on wires {nameless}). This happens when a "
            f"gate has no basis-translation arm and reaches the Rust pass as an "
            f"opaque matrix; decompose it first (Circuit.unitary() plus "
            f"UnitaryDecompositionPass, or sf.compile()) or widen the target "
            f"gate set."
        )

    # Connectivity enforcement.  The Rust router is supposed to make every 2q
    # gate local (and ``manager.compile`` routes through ``route_with_layout``
    # before asking for ``skip_routing``), but nothing verified it: a gate left
    # on a non-edge wire was returned as if the target could execute it.  A
    # routing regression now fails here with the offending pairs instead of
    # surfacing later as a device rejection.
    if declared_connectivity and (position_to_physical is not None or skip_routing):
        allowed = set()
        for edge in declared_connectivity:
            try:
                a, b = int(edge[0]), int(edge[1])
            except (TypeError, IndexError):
                continue
            allowed.add((a, b))
            allowed.add((b, a))

        def _physical(wire: int) -> int:
            """Wire label -> physical qubit (identity without a layout)."""
            if position_to_physical is None:
                return wire
            return position_to_physical.get(wire, wire)

        off_graph = sorted({
            (_physical(int(qubits[0])), _physical(int(qubits[1])))
            for name, qubits, _p in records
            if len(qubits) == 2
            and name.strip().lower() not in _NON_UNITARY_RECORDS
            and (_physical(int(qubits[0])), _physical(int(qubits[1]))) not in allowed
        })
        if off_graph:
            raise RuntimeError(
                f"compiled circuit violates the target's connectivity: {len(off_graph)} "
                f"2-qubit gate(s) act on non-adjacent qubits {off_graph} "
                f"(target {name!r}, {len(allowed) // 2} undirected edge(s)). The "
                f"router left a gate off-graph; pass a target with a coupling map "
                f"and routing enabled (``skip_routing=False``), or split the gate "
                f"set so the router can decompose it."
            )

    # The result is rebuilt on the *circuit's* width, not the target device's:
    # a 5-qubit circuit compiled for ``rigetti_ankaa`` used to come back as an
    # 84-qubit circuit, which changed ``to_unitary()``'s dimension and made the
    # compiled object unusable as a drop-in replacement (HW-4).
    n_out = int(circuit.n_qubits)
    for _name, qubits, _params in records:
        for q in qubits:
            if int(q) + 1 > n_out:
                n_out = int(q) + 1

    c = CircuitCls(n_out, n_cbits=max(int(circuit.n_cbits), 1))
    c.extend_raw_from_records(records)

    # ``to_gate_records()`` reports name/qubits/params only, so a measurement of
    # qubit 0 into classical bit 3 came back as a measurement into bit 0 -- the
    # ``cbit`` argument was silently dropped by compilation.  The wire is
    # unchanged at this point, so the classical bits can be carried across by
    # qubit index.
    measure_cbits = {}
    circuit._ensure_gates()
    for record in circuit._gates:
        if record.name.upper() == "MEASURE" and record.classical_bits:
            measure_cbits[int(record.qubits[0])] = int(record.classical_bits[0])
    if measure_cbits:
        needed = max(int(circuit.n_cbits), 1)
        for record in c._gates:
            if record.name.upper() != "MEASURE":
                continue
            qubit = int(record.qubits[0])
            cbit = measure_cbits.get(qubit, qubit)
            record.classical_bits = [cbit]
            needed = max(needed, cbit + 1)
        if needed > c.n_cbits:
            c._n_cbits = needed

    # Post-pass: rewrite the translator's ``CX`` into the native ``ECR``
    # sequence for targets that implement ``ECR`` but not ``CX``/``CZ``.  This
    # runs *after* the Rust decomposer, which has already lowered CZ, SWAP,
    # CCX, CP, ISWAP, RZZ, RXX and RYY into CX gates, so only one gate type
    # has to be rewritten here.
    c, lowering = _finalize(c, target, lowering)
    if c is not None and any(
        lowering.get(key)
        for key in ("ry_lowered", "cx_lowered", "untranslated_lowered")
    ):
        c._metadata["lowering"] = lowering
    return c
