"""Routing infrastructure for Singularity — decides which simulation regime to use."""

from superfermion.backends.routing.topology_cache import TopologyCache
from superfermion.backends.routing.singularity_router import SingularityRouter
from superfermion.backends.routing.regime_handlers import (
    RegimeStrategy,
    NumpyTurboStrategy,
    RustRayonStrategy,
    MPSDirectStrategy,
    StabilizerStrategy,
)

__all__ = [
    "TopologyCache",
    "SingularityRouter",
    "RegimeStrategy",
    "NumpyTurboStrategy",
    "RustRayonStrategy",
    "MPSDirectStrategy",
    "StabilizerStrategy",
]
