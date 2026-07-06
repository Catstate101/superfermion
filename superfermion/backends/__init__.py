"""
Backends module for Superfermion.
Provides access to simulators, accelerators, and QPU interfaces.
"""

from superfermion.backends.base import Backend
from superfermion.backends.registry import BackendRegistry, get_backend, list_backends
from superfermion.backends.simulator import StatevectorBackend
from superfermion.backends.names import BackendName, resolve_backend_name

# NOTE: Special-purpose backends (CUDA, MPS, JAX) are loaded lazily
# through the declarative factory to avoid heavy import dependencies.

__all__ = [
    "Backend",
    "BackendName",
    "BackendRegistry",
    "get_backend",
    "list_backends",
    "StatevectorBackend",
    "resolve_backend_name",
]
