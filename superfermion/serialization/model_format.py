"""
.sfm Format — SuperFermion Model checkpoint serialization.

Saves trained model parameters, circuit architecture, optimizer state,
and training metadata for reproducible research.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelCheckpoint:
    """A complete model checkpoint for saving/loading trained models.

    Attributes:
        name: Model name.
        parameters: Trained parameter values.
        architecture: Circuit/model architecture description.
        optimizer_state: Optimizer state for resuming training.
        metrics: Training metrics (loss, fidelity, etc.).
        metadata: Additional metadata (epoch, timestamp, etc.).
        version: Checkpoint format version.
    """
    name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)
    optimizer_state: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (numpy arrays converted to lists)."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        result = {
            "format": "sfm",
            "version": self.version,
            "name": self.name,
            "parameters": json.loads(json.dumps(self.parameters, default=_convert)),
            "architecture": self.architecture,
            "optimizer_state": json.loads(json.dumps(self.optimizer_state, default=_convert)),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelCheckpoint:
        """Deserialize from dictionary."""
        def _restore(obj: Any) -> Any:
            if isinstance(obj, dict) and obj.get("__ndarray__"):
                return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
            return obj

        params = {}
        for k, v in data.get("parameters", {}).items():
            params[k] = _restore(v)

        return cls(
            name=data.get("name", ""),
            parameters=params,
            architecture=data.get("architecture", {}),
            optimizer_state=data.get("optimizer_state", {}),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
        )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of model parameters for reproducibility."""
        param_str = json.dumps(self.parameters, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()


def save_model(checkpoint: ModelCheckpoint, path: str) -> Path:
    """Save a model checkpoint to a .sfm file.

    Args:
        checkpoint: ModelCheckpoint to save.
        path: Output file path.

    Returns:
        Path to saved file.
    """
    p = Path(path)
    if p.suffix != ".sfm":
        p = p.with_suffix(".sfm")

    p.parent.mkdir(parents=True, exist_ok=True)

    data = checkpoint.to_dict()
    data["integrity_hash"] = checkpoint.compute_hash()

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return p


def load_model(path: str) -> ModelCheckpoint:
    """Load a model checkpoint from a .sfm file.

    Args:
        path: Path to .sfm file.

    Returns:
        ModelCheckpoint object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoint = ModelCheckpoint.from_dict(data)

    # Verify integrity
    stored_hash = data.get("integrity_hash", "")
    if stored_hash and checkpoint.compute_hash() != stored_hash:
        import warnings
        warnings.warn(
            "Model checkpoint integrity hash mismatch. "
            "Parameters may have been modified.",
            UserWarning,
        )

    return checkpoint
