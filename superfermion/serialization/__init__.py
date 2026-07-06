"""
Superfermion Serialization — Circuit and model serialization with reproducibility.

Supports .sfc (SuperFermion Circuit) and .sfm (SuperFermion Model) formats,
plus QASM3 round-trip and reproducibility manifests.

Usage:
    >>> from superfermion.serialization import save_circuit, load_circuit
    >>> save_circuit(circuit, "my_circuit.sfc")
    >>> loaded = load_circuit("my_circuit.sfc")
"""

from __future__ import annotations

from superfermion.serialization.circuit_format import (
    save_circuit, load_circuit, circuit_to_bytes, circuit_from_bytes,
)
from superfermion.serialization.model_format import (
    save_model, load_model, ModelCheckpoint,
)
from superfermion.serialization.manifest import (
    ReproducibilityManifest, create_manifest,
)
from superfermion.serialization.qasm_roundtrip import (
    to_qasm3, from_qasm3, verify_qasm3_roundtrip,
)

__all__ = [
    "save_circuit", "load_circuit", "circuit_to_bytes", "circuit_from_bytes",
    "save_model", "load_model", "ModelCheckpoint",
    "ReproducibilityManifest", "create_manifest",
    "to_qasm3", "from_qasm3", "verify_qasm3_roundtrip",
]
