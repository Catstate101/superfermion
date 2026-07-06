"""
QASM3 Round-Trip — Bidirectional OpenQASM 3.0 conversion with verification.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from superfermion.parameters import param as make_param


def to_qasm3(circuit: Any) -> str:
    """Convert a Circuit to OpenQASM 3.0 string.

    Delegates to the Circuit's built-in method.
    """
    return circuit.to_qasm3()


def from_qasm3(qasm_str: str) -> Any:
    """Parse an OpenQASM 3.0 string into a Circuit.

    Supports the standard gate set and parameterized gates.

    Args:
        qasm_str: OpenQASM 3.0 string.

    Returns:
        A superfermion.Circuit object.
    """
    from superfermion.circuit import Circuit

    lines = qasm_str.strip().split("\n")
    n_qubits = 2  # Default
    n_cbits = 0

    # First pass: find qubit/bit declarations
    for line in lines:
        line = line.strip()
        qubit_match = re.match(r"qubit\[(\d+)\]\s+\w+;", line)
        if qubit_match:
            n_qubits = int(qubit_match.group(1))
        bit_match = re.match(r"bit\[(\d+)\]\s+\w+;", line)
        if bit_match:
            n_cbits = int(bit_match.group(1))

    circuit = Circuit(n_qubits, n_cbits or n_qubits)

    # QASM gate name -> Circuit method mapping
    gate_map = {
        "h": "h", "x": "x", "y": "y", "z": "z",
        "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",
        "sx": "sx", "id": "id",
        "cx": "cx", "cnot": "cnot", "cz": "cz", "cy": "cy",
        "swap": "swap", "iswap": "iswap", "ecr": "ecr",
        "ccx": "ccx", "cswap": "cswap",
        "rx": "rx", "ry": "ry", "rz": "rz", "p": "p",
        "rxx": "rxx", "ryy": "ryy", "rzz": "rzz",
        "barrier": "barrier", "reset": "reset",
    }

    # Second pass: parse gates
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM"):
            continue
        if line.startswith("qubit") or line.startswith("bit") or line.startswith("include"):
            continue

        # Measurement: c[0] = measure q[0];
        meas_match = re.match(r"c\[(\d+)\]\s*=\s*measure\s+q\[(\d+)\];", line)
        if meas_match:
            cbit = int(meas_match.group(1))
            qubit = int(meas_match.group(2))
            circuit.measure(qubit, cbit)
            continue

        # Parameterized gate: rx(1.57) q[0];
        param_gate_match = re.match(
            r"(\w+)\(([^)]+)\)\s+(.*);", line
        )
        if param_gate_match:
            gate_name = param_gate_match.group(1).lower()
            params_str = param_gate_match.group(2)
            qubits_str = param_gate_match.group(3)

            params = []
            for p in params_str.split(","):
                p = p.strip()
                try:
                    params.append(float(p))
                except ValueError:
                    params.append(make_param(p))

            qubits = _parse_qubits(qubits_str)

            if gate_name in gate_map:
                method = getattr(circuit, gate_map[gate_name])
                method(*params, *qubits)
            continue

        # Simple gate: h q[0];
        simple_match = re.match(r"(\w+)\s+(.*);", line)
        if simple_match:
            gate_name = simple_match.group(1).lower()
            qubits_str = simple_match.group(2)
            qubits = _parse_qubits(qubits_str)

            if gate_name in gate_map:
                method = getattr(circuit, gate_map[gate_name])
                method(*qubits)

    return circuit


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
