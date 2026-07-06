"""
LocalTracker — file-based experiment tracker satisfying ``TrackerProtocol``.

Stores run metadata as JSON files under ``~/.superfermion/runs/<name>/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from superfermion.experiment.protocols import TrackerProtocol


class LocalTracker:
    """Simple, file-backed tracker for local development.

    Each ``on_run_complete`` call appends a JSON record to the run
    directory. Useful for reproducibility without any external service.

    Args:
        name: Experiment name (becomes the directory name).
        base_dir: Root directory for runs.  Defaults to
            ``~/.superfermion/runs``.
    """

    def __init__(
        self,
        name: str = "default",
        base_dir: Optional[str] = None,
    ) -> None:
        self.name = name
        self._base = Path(base_dir or Path.home() / ".superfermion" / "runs")
        self._dir = self._base / name
        self._runs: List[Dict[str, Any]] = []
        self._current_start: Optional[Dict[str, Any]] = None
        self._run_counter = 0

    def on_run_start(
        self,
        circuit: Any,
        device: str,
        shots: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._current_start = {
            "device": device,
            "shots": shots,
            "n_qubits": getattr(circuit, "n_qubits", None),
            "depth": int(getattr(circuit, "depth", 0)),
            "gate_count": int(getattr(circuit, "gate_count", 0)),
            "started_at": time.time(),
            **(metadata or {}),
        }

    def on_run_complete(
        self,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            **(self._current_start or {}),
            "completed_at": time.time(),
            "shots": getattr(result, "shots", 0),
            "n_outcomes": len(getattr(result, "counts", {}) or {}),
            **(metadata or {}),
        }
        self._runs.append(record)
        self._run_counter += 1
        self._persist(record)
        self._current_start = None

    def on_run_error(
        self,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            **(self._current_start or {}),
            "error": str(error),
            "error_type": type(error).__name__,
            "failed_at": time.time(),
            **(metadata or {}),
        }
        self._runs.append(record)
        self._run_counter += 1
        self._persist(record)
        self._current_start = None

    def _persist(self, record: Dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"run_{self._run_counter:04d}.json"
        path.write_text(json.dumps(record, indent=2, default=str))

    @property
    def runs(self) -> List[Dict[str, Any]]:
        """All recorded runs in this tracker session."""
        return list(self._runs)
