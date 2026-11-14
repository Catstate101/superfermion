"""
QASM3 Round-Trip — Bidirectional OpenQASM 3.0 conversion with verification.
"""

from __future__ import annotations

import re
from typing import Any

from superfermion.parameters import param as make_param


def to_qasm3(circuit: Any) -> str:
    """Convert a Circuit to OpenQASM 3.0 string.

    Delegates to the Circuit's built-in method.
    """
    return circuit.to_qasm3()


def from_qasm3(qasm_str: str, strict: bool = True) -> Any:
    """Parse an OpenQASM 3.0 string into a Circuit.

    Supports the standard gate set and parameterized gates, including the
    gates ``to_qasm3`` emits that used to be missing from the dispatch table
    (``r1``, ``cp``, ``cu``/``cu3``, ``crx``, ``cry``, ``crz``, ``u``/``u3``):
    those were *silently dropped*, so a round-trip returned a circuit missing
    operations but with the right qubit count.

    Args:
        qasm_str: OpenQASM 3.0 string.
        strict:   Raise ``ValueError`` on a statement this parser cannot
                  represent (default). ``strict=False`` restores the old
                  skip-unknown behaviour — use it only when the dropped
                  statements are known to be irrelevant.

    Returns:
        A superfermion.Circuit object.

    Raises:
        ValueError: on an unrepresentable statement when ``strict`` is set.
    """
    from superfermion.circuit import Circuit

    lines = qasm_str.strip().split("\n")
    n_qubits = 2  # Default
    n_cbits = 0

    # First pass: find qubit/bit declarations (QASM3 ``qubit[n] q;`` and
    # ``bit[n] c;``, plus the QASM2 ``qreg``/``creg`` spellings).
    for line in lines:
        line = line.strip()
        decl = re.match(
            r"(?:qubit\[(\d+)\]|qreg\s+\w+\[(\d+)\]|qubit)", line
        )
        if decl and not line.startswith("//"):
            size = decl.group(1) or decl.group(2)
            n_qubits = int(size) if size else 1
        bit_match = re.match(
            r"(?:bit\[(\d+)\]|creg\s+\w+\[(\d+)\]|bit)", line
        )
        if bit_match and not line.startswith("//"):
            size = bit_match.group(1) or bit_match.group(2)
            n_cbits = int(size) if size else 1

    circuit = Circuit(n_qubits, n_cbits or n_qubits)

    # QASM gate name -> Circuit method. Names that the IR spells differently
    # are aliased here; everything is checked against the live Circuit object
    # below, so a name this build does not implement fails loudly.
    gate_map = {
        "h": "h", "x": "x", "y": "y", "z": "z",
        "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",
        "sx": "sx", "id": "id",
        "cx": "cnot", "cnot": "cnot", "cz": "cz", "cy": "cy",
        "swap": "swap", "iswap": "iswap", "ecr": "ecr",
        "ccx": "ccx", "toffoli": "ccx", "cswap": "cswap",
        "rx": "rx", "ry": "ry", "rz": "rz", "p": "p", "phase": "p",
        "u1": "p", "u2": "u3", "r1": "r1", "cp": "cp", "cr1": "cp", "cphase": "cp",
        "u": "u", "u3": "u", "cu": "cu", "cu3": "cu",
        "crx": "crx", "cry": "cry", "crz": "crz",
        "rxx": "rxx", "ryy": "ryy", "rzz": "rzz",
        "barrier": "barrier", "reset": "reset",
    }

    def _dispatch(name: str, params: list, qubits: list, line: str):
        method_name = gate_map.get(name)
        method = getattr(circuit, method_name, None) if method_name else None
        if method is None:
            if strict:
                raise ValueError(
                    f"from_qasm3 cannot represent this statement: {line!r}. "
                    f"Supported gates: {', '.join(sorted(gate_map))}. "
                    f"Pass strict=False to skip it instead "
                    f"(the resulting circuit will be missing operations)."
                )
            return
        if name == "u2" and len(params) == 2:
            # u2(φ, λ) is u3(π/2, φ, λ).
            import math

            method(math.pi / 2, *params, *qubits)
            return
        if method_name == "cu" and len(params) > 3:
            # QASM writes cu(theta, phi, lam, gamma) q[c], q[t], but the Python
            # signature puts gamma *after* the qubits.
            theta, phi, lam, gamma = params[:4]
            circuit.cu(theta, phi, lam, qubits[0], qubits[1], gamma)
            return
        method(*params, *qubits)

    # Second pass: parse gates
    in_gate_def = False
    # Qiskit's QASM3 exporter wraps guarded bodies in braces and writes a
    # bare condition (``if (c[0]) { ... }``), unlike ``to_qasm3``'s
    # single-line form.  The active guard is carried on a stack while the
    # braced body is parsed; ``} else {`` flips it.  Circuit.c_if tests a
    # single bit, so only flat (non-nested) guards are representable.
    cond_stack: list[tuple[int, int]] = []

    def _apply_guarded(statement: str, guard: tuple[int, int]) -> None:
        statement = statement.rstrip(";").strip()
        circuit.c_if(*guard)
        meas = re.match(r"c\[(\d+)\]\s*=\s*measure\s+q\[(\d+)\]$", statement)
        if meas:
            circuit.measure(int(meas.group(2)), int(meas.group(1)))
            return
        try:
            _apply_statement(statement, circuit, gate_map, _dispatch, strict)
        finally:
            # A condition is consumed by the next gate; if nothing was
            # applied it must not leak onto an unrelated later gate.
            circuit._pending_condition = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM"):
            continue
        # Gate definitions: ``to_qasm3`` declares gates stdgates.inc lacks.
        # Their bodies use local argument names, so skip to the closing brace;
        # the calls themselves are dispatched by name below.
        if in_gate_def:
            if "}" in line:
                in_gate_def = False
            continue
        if re.match(r"gate\s+\w+", line):
            in_gate_def = "}" not in line
            continue
        if line.startswith("qubit") or line.startswith("bit") or line.startswith("include"):
            continue
        if line.startswith("qreg") or line.startswith("creg"):
            continue

        # Brace scopes from `if (...) { ... }` / `} else { ... }`.
        if line == "}":
            if cond_stack:
                cond_stack.pop()
            continue
        if re.match(r"\}\s*else\s*\{?$", line):
            if not cond_stack:
                raise ValueError(
                    f"from_qasm3 found an else block without a matching if: {line!r}"
                )
            cbit, value = cond_stack.pop()
            cond_stack.append((cbit, 1 - value))
            continue
        if line == "{":
            continue

        # Classical feed-forward: if (c[0] == true) x q[0];  QASM3 compares
        # a bit against a boolean literal (or nothing at all, meaning true),
        # but `== 1` / `== 0` is accepted too.
        cond_match = re.match(
            r"if\s*\(\s*c\[(\d+)\]\s*(?:==\s*(true|false|\d+))?\s*\)\s*(.*)$",
            line,
        )
        if cond_match:
            cbit = int(cond_match.group(1))
            raw_value = cond_match.group(2)
            if raw_value is None or raw_value == "true":
                value = 1
            elif raw_value == "false":
                value = 0
            else:
                value = int(raw_value)
            rest = cond_match.group(3).strip()
            if rest:
                # Inline braced body: `if (c[0]) { x q[0]; }`.
                if rest.startswith("{"):
                    rest = rest[1:].rstrip()
                    if rest.endswith("}"):
                        rest = rest[:-1]
                    rest = rest.strip()
            if not rest:
                if cond_stack:
                    raise ValueError(
                        "Nested QASM3 if blocks cannot be represented: "
                        "Circuit.c_if tests one classical bit at a time. "
                        f"Offending line: {line!r}"
                    )
                cond_stack.append((cbit, value))
                continue
            if cond_stack:
                raise ValueError(
                    "Nested QASM3 if blocks cannot be represented: "
                    "Circuit.c_if tests one classical bit at a time. "
                    f"Offending line: {line!r}"
                )
            _apply_guarded(rest, (cbit, value))
            continue

        # Measurement: c[0] = measure q[0];
        meas_match = re.match(r"c\[(\d+)\]\s*=\s*measure\s+q\[(\d+)\];", line)
        if meas_match:
            if cond_stack:
                circuit.c_if(*cond_stack[-1])
            circuit.measure(int(meas_match.group(2)), int(meas_match.group(1)))
            continue

        # Whole-register measurement: measure q;
        seed_match = re.match(r"measure\s+q;", line)
        if seed_match:
            for q in range(n_qubits):
                if cond_stack:
                    circuit.c_if(*cond_stack[-1])
                circuit.measure(q, q)
            continue

        if cond_stack:
            _apply_guarded(line, cond_stack[-1])
            continue

        _apply_statement(line.rstrip(";").strip(), circuit, gate_map, _dispatch, strict)

    return circuit


def _apply_statement(
    statement: str, circuit: Any, gate_map: dict, dispatch, strict: bool
) -> None:
    """Apply one ``gate(params) qubits`` / ``gate qubits`` statement."""
    param_gate_match = re.match(r"(\w+)\(([^)]+)\)\s+(.*)$", statement)
    if param_gate_match:
        gate_name = param_gate_match.group(1).lower()
        params = []
        for p in param_gate_match.group(2).split(","):
            p = p.strip()
            try:
                params.append(float(p))
            except ValueError:
                params.append(make_param(p))
        dispatch(gate_name, params, _parse_qubits(param_gate_match.group(3)), statement)
        return

    simple_match = re.match(r"(\w+)(?:\s+(.*))?$", statement)
    if simple_match:
        gate_name = simple_match.group(1).lower()
        qubits = _parse_qubits(simple_match.group(2) or "")
        if gate_name == "barrier":
            circuit.barrier(*qubits)
            return
        dispatch(gate_name, [], qubits, statement)
        return

    if strict:
        raise ValueError(f"from_qasm3 cannot parse this statement: {statement!r}")


def _parse_qubits(qubits_str: str) -> list:
    """Parse qubit references like 'q[0], q[1]' into indices."""
    qubits = []
    for match in re.finditer(r"q\[(\d+)\]", qubits_str):
        qubits.append(int(match.group(1)))
    return qubits


def verify_qasm3_roundtrip(circuit: Any) -> bool:
    """Verify that a circuit survives QASM3 serialization/deserialization.

    Args:
        circuit: Original Circuit.

    Returns:
        True if the round-trip preserves the circuit structure.
    """
    qasm = to_qasm3(circuit)
    reconstructed = from_qasm3(qasm)

    # Compare gate counts and structure
    if circuit.n_qubits != reconstructed.n_qubits:
        return False
    if int(circuit.gate_count) != int(reconstructed.gate_count):
        return False
    if int(circuit.depth) != int(reconstructed.depth):
        return False

    return True
