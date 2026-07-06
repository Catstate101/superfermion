"""
Superfermion Experiment — protocol-based experiment tracking.

Usage::

    import superfermion as sf
    from superfermion.experiment import LocalTracker

    with sf.experiment("my-experiment"):
        result = sf.run(circuit, device="cpu")
"""

from superfermion.experiment.protocols import TrackerProtocol
from superfermion.experiment.context import experiment, _get_active_tracker
from superfermion.experiment.local_tracker import LocalTracker

__all__ = [
    "TrackerProtocol",
    "experiment",
    "LocalTracker",
    "_get_active_tracker",
]
