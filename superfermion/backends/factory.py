"""
Backend factory — compatibility shim.

All simulation now routes through the Rust core (RustBackend).
This module exists only for backwards compatibility with code that
calls ``get_backend("rust")`` or ``get_backend("statevector")``.

New code should use ``sf.run(circuit, device="cpu")`` directly.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List

from superfermion.backends.base import Backend

_BACKEND_REGISTRY: Dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend factory callable for the given name."""
    _BACKEND_REGISTRY[name.lower()] = factory


def get_backend(name: str | None = None) -> Backend:
    """Retrieve a backend instance by name.

    All names now resolve to RustBackend (the single simulation engine).
    """
    if name is None:
        name = "rust"
    key = name.lower()

    if key not in _BACKEND_REGISTRY:
        # Fall back to rust for any unrecognized name
        warnings.warn(
            f"Backend '{key}' not recognized, using 'rust'. "
            f"Use sf.run(circuit, device='cpu') for the new API.",
            DeprecationWarning,
            stacklevel=2,
        )
        key = "rust"
    return _BACKEND_REGISTRY[key]()


def list_backends() -> List[str]:
    """Return all registered backend names."""
    return list(_BACKEND_REGISTRY.keys())


def _make_lazy(module: str, classname: str) -> Callable[[], Backend]:
    """Return a zero-arg callable that imports and instantiates on first access."""
    import importlib

    def factory() -> Backend:
        mod = importlib.import_module(module)
        cls = getattr(mod, classname)
        return cls()

    return factory


# All names route to RustBackend — the only simulation backend
register_backend("rust", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend("statevector", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend("simulator", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend("auto", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend("singularity", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend("numpy", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))

# These remain for backwards compat but will be deprecated
register_backend("mps", _make_lazy("superfermion.backends.mps", "MPSSimulatorBackend"))
register_backend("stabilizer", _make_lazy("superfermion.backends.stabilizer", "StabilizerBackend"))
register_backend("density_matrix", _make_lazy("superfermion.backends.density_matrix", "DensityMatrixBackend"))
