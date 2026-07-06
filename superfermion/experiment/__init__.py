"""
Superfermion Experiment — Experiment tracking and model registry.

Track metrics, parameters, circuits, and results across hundreds of runs.

Usage:
    >>> from superfermion.experiment import Tracker, ExperimentRun
    >>> tracker = Tracker("my_experiment")
    >>> with tracker.run("run_001") as run:
    ...     run.log_param("lr", 0.01)
    ...     run.log_metric("loss", 0.5)
"""

from __future__ import annotations

from superfermion.experiment.tracker import (
    Tracker, ExperimentRun, RunStatus,
)
from superfermion.experiment.registry import (
    ModelRegistry, RegisteredModel,
)

__all__ = [
    "Tracker", "ExperimentRun", "RunStatus",
    "ModelRegistry", "RegisteredModel",
]
