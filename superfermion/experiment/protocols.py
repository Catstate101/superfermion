"""
TrackerProtocol — the interface for experiment tracking backends.

Any object satisfying this protocol can be used with ``sf.experiment()``
or passed directly to ``sf.run(tracker=...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from superfermion.circuit import Circuit
    from superfermion.results import RunResult


@runtime_checkable
class TrackerProtocol(Protocol):
    """Lifecycle hooks called by the runner around each execution."""

    def on_run_start(
        self,
        circuit: "Circuit",
        device: str,
        shots: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called before execution begins."""
        ...

    def on_run_complete(
        self,
        result: "RunResult",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called after successful execution."""
        ...

    def on_run_error(
        self,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when execution raises an exception."""
        ...
