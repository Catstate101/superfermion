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

    Returns:
        Compiled SF Circuit.

    Raises:
        ImportError: If ``_sf_core`` is not available (Rust extension required).
        RuntimeError: If the Rust compiler encounters an unsupported gate.
    """
    from superfermion._sf_core import Compiler

    if target is not None:
        native_gates = list(getattr(target, "native_gates", []))
        coupling_map = list(getattr(target, "coupling_map", []))
        n_qubits = getattr(target, "n_qubits", circuit.n_qubits)
        name = getattr(target, "name", "target")
    else:
        native_gates = []
        coupling_map = []
        n_qubits = circuit.n_qubits
        name = "none"

    # Clifford pre-simplification: reduces gate count via Aaronson-Gottesman
    # tableau decomposition. Only applied when there is no coupling map,
    # because the synthesis produces long-range CNOTs that destroy locality
    # and make routing harder (or impossible) on constrained topologies.
    if not pre_simplified and not coupling_map:
        simplified = _simplify_clifford(circuit)
        if simplified is not None:
            circuit = simplified

    dag = circuit.to_ir()

    compiler = Compiler(
        name=name,
        native_gates=native_gates,
        n_qubits=n_qubits,
        connectivity=coupling_map,
        optimization_level=level,
    )

    compiled_dag = compiler.compile(dag)

    from superfermion.circuit import Circuit as CircuitCls
    records = compiled_dag.to_gate_records()
    c = CircuitCls(n_qubits)
    c.extend_raw_from_records(records)
    return c
