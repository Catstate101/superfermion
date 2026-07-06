"""
Backends module for Superfermion.

Provides access to simulators, accelerators, and QPU interfaces
through a declarative factory.
"""

from superfermion.backends.base import Backend
from superfermion.backends.factory import get_backend, list_backends

__all__ = [
    "Backend",
    "get_backend",
    "list_backends",
]
