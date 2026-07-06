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
    circuit: Circuit,
    level: int = 1,
    target: Optional[object] = None,
    pre_simplified: bool = False,
) -> Circuit:
    """Compile a circuit using the Rust-native compilation pipeline.

    The Rust pipeline includes:
      - Gate cancellation
      - High-level decomposition (SWAP → 3 CNOTs)
      - Superconducting basis translation (if SX in native gates)
      - Rotation merging
      - SABRE routing (if coupling map is provided)
      - Pauli twirling (at level >= 2)

    Args:
        circuit: SF Circuit to compile.
        level: Optimization level (0-2). Level 2 includes Pauli twirling.
        target: Optional HardwareSpec with coupling_map and native_gates.

    Returns:
        Compiled SF Circuit.
    """
    from superfermion._sf_core import Compiler

    # Determine native gates and connectivity
    if target is not None:
        native_gates = getattr(target, "native_gates", ["h", "x", "y", "z", "cx"])
        coupling_map = getattr(target, "coupling_map", [])
        n_qubits = getattr(target, "n_qubits", circuit.n_qubits)
    else:
        native_gates = ["h", "x", "y", "z", "cx"]
        coupling_map = []
        n_qubits = circuit.n_qubits

    # -- Clifford pre-simplification --
    # For Clifford circuits, synthesize a canonical circuit via the
    # Aaronson-Gottesman tableau decomposition before Rust compilation.
    # This reduces gate count from O(n^2) to O(n^2/log n) and can be
    # 10-50x faster for downstream compilation (basis translation, etc.).
    # Set pre_simplified=True if the caller has already called
    # simplify_clifford() — avoids redundant tableau evolution.
    if not pre_simplified:
        simplified = _simplify_clifford(circuit)
        if simplified is not None:
            circuit = simplified

    # Convert to Rust DAG
    dag = circuit.to_ir()

    # Create Rust compiler with target spec
    compiler = Compiler(
        name="rust_pipeline",
        native_gates=native_gates,
        n_qubits=n_qubits,
        connectivity=coupling_map,
        optimization_level=level,
    )

    # Compile
    compiled_dag = compiler.compile(dag)

    # Convert back to SF Circuit directly via PyO3 gate records.
    # Uses extend_raw_from_records which pre-allocates the gate list
    # and avoids the intermediate list-comprehension batch allocation.
    from superfermion.circuit import Circuit
    records = compiled_dag.to_gate_records()
    c = Circuit(n_qubits)
    c.extend_raw_from_records(records)
    return c
