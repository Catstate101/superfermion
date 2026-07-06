"""
sf.experiment() — context manager for scoped experiment tracking.

Uses ``contextvars`` for thread-safe, implicit tracker binding so that
``sf.run()`` automatically picks up the active tracker without explicit
``tracker=`` arguments.

Usage::

    with sf.experiment("bell-test", tracker=LocalTracker()):
        result = sf.run(circuit, device="cpu")
        # tracker.on_run_start / on_run_complete called automatically
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from superfermion.experiment.protocols import TrackerProtocol

_active_tracker: ContextVar[Optional["TrackerProtocol"]] = ContextVar(
    "_active_tracker", default=None,
)


def _get_active_tracker() -> Optional["TrackerProtocol"]:
    """Return the tracker for the current context, or ``None``."""
    return _active_tracker.get()


@contextmanager
def experiment(name: str, tracker: Optional["TrackerProtocol"] = None):
    """Activate a tracker for all ``sf.run()`` calls within this block.

    If *tracker* is ``None``, a ``LocalTracker`` with the given name is
    created automatically.

    Args:
        name: Human-readable experiment name (used for local storage path).
        tracker: A ``TrackerProtocol``-satisfying object. Defaults to
            ``LocalTracker(name)``.

    Yields:
        The active tracker instance.
    """
    if tracker is None:
        from superfermion.experiment.local_tracker import LocalTracker
        tracker = LocalTracker(name)

    token = _active_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _active_tracker.reset(token)
