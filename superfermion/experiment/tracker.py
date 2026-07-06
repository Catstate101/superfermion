"""
Experiment Tracker — Log metrics, parameters, and artifacts across runs.

Provides MLflow/W&B-style tracking without external dependencies.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


class RunStatus(Enum):
    """Experiment run status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MetricEntry:
    """A single metric data point."""
    key: str
    value: float
    step: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentRun:
    """A single experiment run with logged metrics and parameters.

    Attributes:
        run_id: Unique run identifier.
        experiment_name: Parent experiment name.
        status: Current run status.
        params: Logged parameters.
        metrics: Logged metrics.
        tags: Run tags.
        artifacts: Artifact file paths.
        start_time: Run start timestamp.
        end_time: Run end timestamp.
        notes: Optional researcher notes.
    """
    run_id: str = ""
    experiment_name: str = ""
    status: RunStatus = RunStatus.CREATED
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[MetricEntry]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    notes: str = ""

    @property
    def duration_seconds(self) -> float:
        if self.end_time <= 0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter value."""
        self.params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.params.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number (for training curves).
        """
        if key not in self.metrics:
            self.metrics[key] = []

        actual_step = step if step is not None else len(self.metrics[key])
        self.metrics[key].append(MetricEntry(key=key, value=value, step=actual_step))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, path: str) -> None:
        """Log an artifact file path."""
        self.artifacts.append(path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a run tag."""
        self.tags[key] = value

    def get_metric(self, key: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1].value
        return None

    def get_metric_history(self, key: str) -> List[float]:
        """Get all values of a metric."""
        return [m.value for m in self.metrics.get(key, [])]

    def finish(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Mark the run as finished."""
        self.status = status
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize run data."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "params": self.params,
            "metrics": {
                k: [{"value": m.value, "step": m.step, "timestamp": m.timestamp}
                    for m in entries]
                for k, entries in self.metrics.items()
            },
            "tags": self.tags,
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }

    def summary(self) -> Dict[str, Any]:
        """Get a summary with final metric values."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "duration_s": round(self.duration_seconds, 2),
            "params": self.params,
            "final_metrics": {k: self.get_metric(k) for k in self.metrics},
        }

    def __repr__(self) -> str:
        n_metrics = sum(len(v) for v in self.metrics.values())
        return (
            f"ExperimentRun('{self.run_id}', status={self.status.value}, "
            f"params={len(self.params)}, metric_entries={n_metrics})"
        )


class Tracker:
    """Experiment tracker for managing multiple runs.

    Args:
        experiment_name: Name of the experiment.
        storage_dir: Directory for persisting run data.

    Examples:
        >>> tracker = Tracker("vqe_h2")
        >>> with tracker.run("run_001") as run:
        ...     run.log_param("ansatz", "UCCSD")
        ...     run.log_param("optimizer", "Adam")
        ...     for i in range(100):
        ...         run.log_metric("energy", -1.0 + 0.01 * i, step=i)
        >>> tracker.compare("energy")
    """

    def __init__(
        self,
        experiment_name: str = "default",
        storage_dir: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._runs: Dict[str, ExperimentRun] = {}
        self._active_run: Optional[ExperimentRun] = None
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._run_counter = 0

        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def run(
        self,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[ExperimentRun, None, None]:
        """Create and manage an experiment run.

        Args:
            run_id: Unique run ID. Auto-generated if not provided.
            tags: Optional run tags.

        Yields:
            ExperimentRun for logging.
        """
        if run_id is None:
            self._run_counter += 1
            run_id = f"run_{self._run_counter:04d}"

        exp_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            status=RunStatus.RUNNING,
            tags=tags or {},
        )

        self._runs[run_id] = exp_run
        self._active_run = exp_run

        try:
            yield exp_run
            exp_run.finish(RunStatus.COMPLETED)
        except Exception as e:
            exp_run.finish(RunStatus.FAILED)
            exp_run.set_tag("error", str(e))
            raise
        finally:
            self._active_run = None
            if self._storage_dir:
                self._save_run(exp_run)

    def start_run(
        self,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """Start a run without context manager."""
        if run_id is None:
            self._run_counter += 1
            run_id = f"run_{self._run_counter:04d}"

        exp_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            status=RunStatus.RUNNING,
            tags=tags or {},
        )
        self._runs[run_id] = exp_run
        self._active_run = exp_run
        return exp_run

    def end_run(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """End the active run."""
        if self._active_run:
            self._active_run.finish(status)
            if self._storage_dir:
                self._save_run(self._active_run)
            self._active_run = None

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a run by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        status: Optional[RunStatus] = None,
    ) -> List[ExperimentRun]:
        """List all runs, optionally filtered by status."""
        runs = list(self._runs.values())
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.start_time, reverse=True)

    def compare(self, metric_key: str) -> Dict[str, Optional[float]]:
        """Compare final metric values across all runs.

        Args:
            metric_key: Metric name to compare.

        Returns:
            Dict mapping run_id to final metric value.
        """
        return {
            run_id: run.get_metric(metric_key)
            for run_id, run in self._runs.items()
        }

    def best_run(self, metric_key: str, minimize: bool = True) -> Optional[ExperimentRun]:
        """Find the run with the best metric value.

        Args:
            metric_key: Metric to optimize.
            minimize: If True, lower is better.

        Returns:
            Best ExperimentRun, or None.
        """
        valid_runs = [
            (run, run.get_metric(metric_key))
            for run in self._runs.values()
            if run.get_metric(metric_key) is not None
        ]
        if not valid_runs:
            return None

        if minimize:
            return min(valid_runs, key=lambda x: x[1])[0]
        return max(valid_runs, key=lambda x: x[1])[0]

    def _save_run(self, run: ExperimentRun) -> None:
        """Persist run data to disk."""
        if not self._storage_dir:
            return
        run_file = self._storage_dir / f"{run.run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def export_all(self) -> List[Dict[str, Any]]:
        """Export all runs as dictionaries."""
        return [run.to_dict() for run in self._runs.values()]

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def __repr__(self) -> str:
        return (
            f"Tracker('{self.experiment_name}', "
            f"runs={self.n_runs})"
        )
