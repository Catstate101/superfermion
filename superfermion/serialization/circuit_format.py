"""
.sfc Format — SuperFermion Circuit serialization.

Binary + JSON hybrid format with versioning and integrity checks.

File structure:
    [4 bytes: magic "SFC\x01"]
    [4 bytes: version u32-le]
    [4 bytes: payload_length u32-le]
    [payload_length bytes: JSON-encoded circuit]
    [32 bytes: SHA-256 hash of payload]
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional


SFC_MAGIC = b"SFC\x01"
SFC_VERSION = 1


def circuit_to_bytes(circuit: Any) -> bytes:
    """Serialize a Circuit to .sfc bytes.

    Args:
        circuit: A superfermion.Circuit object.

    Returns:
        Bytes in .sfc format.
    """
    # Build JSON payload
    payload = _circuit_to_dict(circuit)
    payload_bytes = json.dumps(payload, indent=None, separators=(",", ":")).encode("utf-8")

    # Compute integrity hash
    payload_hash = hashlib.sha256(payload_bytes).digest()

    # Pack: magic + version + length + payload + hash
    header = SFC_MAGIC
    header += struct.pack("<I", SFC_VERSION)
    header += struct.pack("<I", len(payload_bytes))

    return header + payload_bytes + payload_hash


def circuit_from_bytes(data: bytes) -> Any:
    """Deserialize a Circuit from .sfc bytes.

    Args:
        data: Raw .sfc file bytes.

    Returns:
        A superfermion.Circuit object.

    Raises:
        ValueError: If data is corrupt or has wrong format.
    """
    if len(data) < 44:  # Minimum: 4 + 4 + 4 + 0 + 32
        raise ValueError("Data too short to be a valid .sfc file")

    # Parse header
    magic = data[:4]
    if magic != SFC_MAGIC:
        raise ValueError(f"Invalid .sfc magic bytes: {magic!r}")

    version = struct.unpack("<I", data[4:8])[0]
    if version > SFC_VERSION:
        raise ValueError(
            f"Unsupported .sfc version {version}. "
            f"This library supports up to version {SFC_VERSION}.\n"
            f"  Fix: Update superfermion to the latest version."
        )

    payload_length = struct.unpack("<I", data[8:12])[0]
    payload_bytes = data[12:12 + payload_length]
    stored_hash = data[12 + payload_length:12 + payload_length + 32]

    # Verify integrity
    computed_hash = hashlib.sha256(payload_bytes).digest()
    if computed_hash != stored_hash:
        raise ValueError(
            "Circuit file integrity check failed (SHA-256 mismatch).\n"
            "  The file may be corrupted or tampered with."
        )

    payload = json.loads(payload_bytes.decode("utf-8"))
    return _dict_to_circuit(payload)


def save_circuit(circuit: Any, path: str) -> Path:
    """Save a circuit to a .sfc file.

    Args:
        circuit: A superfermion.Circuit object.
        path: Output file path (will add .sfc extension if missing).

    Returns:
        Path to the saved file.
    """
    p = Path(path)
    if p.suffix != ".sfc":
        p = p.with_suffix(".sfc")

    data = circuit_to_bytes(circuit)
    p.write_bytes(data)
    return p


def load_circuit(path: str) -> Any:
    """Load a circuit from a .sfc file.

    Args:
        path: Path to .sfc file.

    Returns:
        A superfermion.Circuit object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Circuit file not found: {path}")

    data = p.read_bytes()
    return circuit_from_bytes(data)


def _circuit_to_dict(circuit: Any) -> Dict[str, Any]:
    """Convert a Circuit to a serializable dictionary."""
    from superfermion.parameters import SymbolicParameter

    gates = []
    for gate in circuit._gates:
        g = {
            "name": gate.name,
            "qubits": gate.qubits,
        }
        if gate.params:
            g["params"] = []
            for p in gate.params:
                if isinstance(p, SymbolicParameter):
                    g["params"].append({"type": "symbolic", "name": p.name})
                else:
                    g["params"].append({"type": "const", "value": float(p)})
        if gate.classical_bits:
            g["classical_bits"] = gate.classical_bits
        gates.append(g)

    return {
        "format": "sfc",
        "version": SFC_VERSION,
        "circuit": {
            "n_qubits": circuit.n_qubits,
            "n_cbits": circuit.n_cbits,
            "name": circuit._name,
            "gates": gates,
            "metadata": {
                "depth": int(circuit.depth),
                "gate_count": int(circuit.gate_count),
                "n_parameters": circuit.n_parameters,
            },
        },
    }


def _dict_to_circuit(data: Dict[str, Any]) -> Any:
    """Reconstruct a Circuit from a dictionary."""
    from superfermion.circuit import Circuit
    from superfermion.parameters import param as make_param

    cd = data["circuit"]
    circuit = Circuit(cd["n_qubits"], cd.get("n_cbits"), cd.get("name"))

    for g in cd["gates"]:
        params = []
        for p in g.get("params", []):
            if isinstance(p, dict):
                if p["type"] == "symbolic":
                    params.append(make_param(p["name"]))
                else:
                    params.append(p["value"])
            else:
                params.append(p)

        circuit._add_gate(
            name=g["name"],
            qubits=g["qubits"],
            params=params if params else None,
        )

    return circuit
