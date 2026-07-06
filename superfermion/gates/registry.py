"""GateRegistry — single source of truth for cross-framework gate name mappings.

Replaces 9 duplicated GATE_MAP dicts across bridge/__init__.py, serialization,
and backends. Bidirectional: sf_name <-> framework_name for each supported framework.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class GateRegistry:
    """Bidirectional gate name mapping for all supported frameworks.

    Usage:
        >>> GateRegistry.to_qiskit("H")   → "h"
        >>> GateRegistry.from_qiskit("h") → "H"
        >>> GateRegistry.to_ionq("CX")    → "cnot"
    """

    # SF canonical (uppercase) → Qiskit (lowercase)
    _SF_TO_QISKIT: Dict[str, str] = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'sdg', 'T': 't', 'TDG': 'tdg',
        'SX': 'sx', 'ID': 'id', 'I': 'id',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'P': 'p', 'U': 'u', 'U3': 'u3',
        'CU': 'cu', 'CU3': 'cu3', 'CP': 'cp',
        'CX': 'cx', 'CNOT': 'cx',
        'CZ': 'cz', 'CY': 'cy',
        'SWAP': 'swap', 'ISWAP': 'iswap',
        'ECR': 'ecr',
        'CCX': 'ccx', 'CSWAP': 'cswap',
        'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
        'MEASURE': 'measure',
        'BARRIER': 'barrier',
        'RESET': 'reset',
    }

    # SF canonical → PennyLane
    _SF_TO_PENNYLANE: Dict[str, str] = {
        'H': 'Hadamard', 'X': 'PauliX', 'Y': 'PauliY', 'Z': 'PauliZ',
        'S': 'S', 'T': 'T',
        'RX': 'RX', 'RY': 'RY', 'RZ': 'RZ',
        'CX': 'CNOT', 'CNOT': 'CNOT', 'CZ': 'CZ',
        'SWAP': 'SWAP',
        'CCX': 'Toffoli',
    }

    # SF canonical → IonQ API
    _SF_TO_IONQ: Dict[str, str] = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'si', 'T': 't', 'TDG': 'ti',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CX': 'cnot', 'CNOT': 'cnot',
        'CZ': 'cz',
        'SWAP': 'swap',
    }

    # SF canonical → Cirq type names
    _SF_TO_CIRQ: Dict[str, str] = {
        'H': 'H', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
        'S': 'S', 'T': 'T',
        'RX': 'Rx', 'RY': 'Ry', 'RZ': 'Rz',
        'CX': 'CNOT', 'CNOT': 'CNOT',
        'CZ': 'CZ',
        'SWAP': 'SWAP',
        'CCX': 'CCX',
    }

    # SF canonical → QASM
    _SF_TO_QASM: Dict[str, str] = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'sdg', 'T': 't', 'TDG': 'tdg',
        'SX': 'sx',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'U': 'u', 'U3': 'u', 'P': 'p',
        'CX': 'cx', 'CNOT': 'cx',
        'CZ': 'cz',
        'SWAP': 'swap',
        'CCX': 'ccx',
    }

    # ── Lazy-computed reverse mappings ──
    _QISKIT_TO_SF: Optional[Dict[str, str]] = None
    _PENNYLANE_TO_SF: Optional[Dict[str, str]] = None
    _IONQ_TO_SF: Optional[Dict[str, str]] = None
    _CIRQ_TO_SF: Optional[Dict[str, str]] = None
    _QASM_TO_SF: Optional[Dict[str, str]] = None

    @classmethod
    def _get_reverse(cls, forward: dict) -> dict:
        """Build reverse mapping, handling duplicates by keeping first."""
        rev = {}
        for sf_name, fw_name in forward.items():
            if fw_name not in rev:
                rev[fw_name] = sf_name
        return rev

    # ── Forward lookups (SF → framework) ──
    @classmethod
    def to_qiskit(cls, sf_name: str) -> Optional[str]:
        return cls._SF_TO_QISKIT.get(sf_name.upper())

    @classmethod
    def to_pennylane(cls, sf_name: str) -> Optional[str]:
        return cls._SF_TO_PENNYLANE.get(sf_name.upper())

    @classmethod
    def to_ionq(cls, sf_name: str) -> Optional[str]:
        return cls._SF_TO_IONQ.get(sf_name.upper())

    @classmethod
    def to_cirq(cls, sf_name: str) -> Optional[str]:
        return cls._SF_TO_CIRQ.get(sf_name.upper())

    @classmethod
    def to_qasm(cls, sf_name: str) -> Optional[str]:
        return cls._SF_TO_QASM.get(sf_name.upper())

    # ── Reverse lookups (framework → SF) ──
    @classmethod
    def from_qiskit(cls, qiskit_name: str) -> Optional[str]:
        if cls._QISKIT_TO_SF is None:
            cls._QISKIT_TO_SF = cls._get_reverse(cls._SF_TO_QISKIT)
        return cls._QISKIT_TO_SF.get(qiskit_name.lower())

    @classmethod
    def from_pennylane(cls, pl_name: str) -> Optional[str]:
        if cls._PENNYLANE_TO_SF is None:
            cls._PENNYLANE_TO_SF = cls._get_reverse(cls._SF_TO_PENNYLANE)
        return cls._PENNYLANE_TO_SF.get(pl_name)

    @classmethod
    def from_ionq(cls, ionq_name: str) -> Optional[str]:
        if cls._IONQ_TO_SF is None:
            cls._IONQ_TO_SF = cls._get_reverse(cls._SF_TO_IONQ)
        return cls._IONQ_TO_SF.get(ionq_name.lower())

    @classmethod
    def from_cirq(cls, cirq_name: str) -> Optional[str]:
        if cls._CIRQ_TO_SF is None:
            cls._CIRQ_TO_SF = cls._get_reverse(cls._SF_TO_CIRQ)
        return cls._CIRQ_TO_SF.get(cirq_name)

    @classmethod
    def from_qasm(cls, qasm_name: str) -> Optional[str]:
        if cls._QASM_TO_SF is None:
            cls._QASM_TO_SF = cls._get_reverse(cls._SF_TO_QASM)
        return cls._QASM_TO_SF.get(qasm_name.lower())

    @classmethod
    def supported_sf_gates(cls) -> list:
        """Return the set of canonical SF gate names."""
        return sorted(cls._SF_TO_QISKIT.keys())
