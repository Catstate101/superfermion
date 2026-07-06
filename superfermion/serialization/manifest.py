"""
Reproducibility Manifest — Complete record for experiment reproduction.

Captures everything needed to reproduce a quantum computation result:
circuit hash, backend, calibration, parameters, shots, seed, and result hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import superfermion


@dataclass
class ReproducibilityManifest:
    """Complete manifest for reproducing a quantum computation.

    Attributes:
        circuit_hash: SHA-256 hash of the circuit definition.
        backend: Backend identifier used.
        backend_version: Backend software version.
        calibration_date: Hardware calibration date (if applicable).
        shots: Number of measurement shots.
        seed: Random seed used.
        result_hash: SHA-256 hash of the result.
        parameters: Bound parameter values.
        sf_version: Superfermion version.
        python_version: Python version.
        platform: OS/platform info.
        timestamp: When the computation was performed.
        notes: Optional researcher notes.
    """
    circuit_hash: str = ""
    backend: str = ""
    backend_version: str = ""
    calibration_date: str = ""
    shots: int = 0
    seed: Optional[int] = None
    result_hash: str = ""
    parameters: Dict[str, float] = field(default_factory=dict)
    sf_version: str = field(default_factory=lambda: superfermion.__version__)
    python_version: str = field(default_factory=platform.python_version)
    platform_info: str = field(default_factory=platform.platform)
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> ReproducibilityManifest:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        """Save manifest to file."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> ReproducibilityManifest:
        """Load manifest from file."""
        from pathlib import Path
        return cls.from_json(Path(path).read_text())

    def verify(self, result_hash: str) -> bool:
        """Verify that a result matches this manifest."""
        return self.result_hash == result_hash

    def __repr__(self) -> str:
        return (
            f"ReproducibilityManifest(backend='{self.backend}', "
            f"shots={self.shots}, circuit_hash='{self.circuit_hash[:12]}...')"
        )


def create_manifest(
    circuit: Any,
    backend: str = "",
    shots: int = 0,
    seed: Optional[int] = None,
    parameters: Optional[Dict[str, float]] = None,
    result: Any = None,
    notes: str = "",
) -> ReproducibilityManifest:
    """Create a reproducibility manifest from a circuit execution.

    Args:
        circuit: The Circuit object.
        backend: Backend name used.
        shots: Number of shots.
        seed: Random seed.
        parameters: Bound parameters.
        result: Execution result (for result hash).
        notes: Optional notes.

    Returns:
        ReproducibilityManifest for the execution.
    """
    # Hash the circuit
    circuit_json = circuit.to_json() if hasattr(circuit, "to_json") else str(circuit)
    circuit_hash = hashlib.sha256(circuit_json.encode()).hexdigest()

    # Hash the result
    result_hash = ""
    if result is not None:
        result_str = str(result.counts if hasattr(result, "counts") else result)
        result_hash = hashlib.sha256(result_str.encode()).hexdigest()

    return ReproducibilityManifest(
        circuit_hash=circuit_hash,
        backend=backend,
        shots=shots,
        seed=seed,
        result_hash=result_hash,
        parameters=parameters or {},
        notes=notes,
    )
