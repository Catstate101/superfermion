"""
Declarative backend factory with lazy instantiation.

Backends are registered via a dict mapping plain string names to factory
callables. This is the single source of truth for backend lookup.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from superfermion.backends.base import Backend

_BACKEND_REGISTRY: Dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend factory callable for the given name."""
    _BACKEND_REGISTRY[name.lower()] = factory


def get_backend(name: str | None = None) -> Backend:
    """Retrieve a backend instance by name.

    If *name* is ``None``, auto-selects CUDA if available, otherwise
    STATEVECTOR.
    """
    if name is None:
        name = _auto_select()
    key = name.lower()

    if key not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Backend '{key}' is not registered. "
            f"Registered: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[key]()


def list_backends() -> List[str]:
    """Return all registered backend names."""
    return list(_BACKEND_REGISTRY.keys())


def _auto_select() -> str:
    """Auto-select best available backend."""
    try:
        import cupy  # noqa: F401
        return "cuda"
    except ImportError:
        return "statevector"


def _make_lazy(module: str, classname: str) -> Callable[[], Backend]:
    """Return a zero-arg callable that imports and instantiates on first access."""
    import importlib

    def factory() -> Backend:
        mod = importlib.import_module(module)
        cls = getattr(mod, classname)
        return cls()

    return factory


# -- Register core backends (lazy imports via _make_lazy) --

register_backend("statevector", _make_lazy("superfermion.backends.simulator", "StatevectorBackend"))
register_backend("jax", _make_lazy("superfermion.backends.jax_sim", "JAXBackend"))
register_backend("rust", _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend("cuda", _make_lazy("superfermion.backends.cuda", "CUSimulatorBackend"))
register_backend("cupy", _make_lazy("superfermion.backends.cupy_sim", "CupyBackend"))
register_backend("mps", _make_lazy("superfermion.backends.mps", "MPSSimulatorBackend"))
register_backend("cluster", _make_lazy("superfermion.backends.cluster", "DistributedJAXBackend"))
register_backend("jax_mps", _make_lazy("superfermion.backends.jax_mps", "JAXMPSBackend"))
register_backend("cuda_mps", _make_lazy("superfermion.backends.cuda_mps", "CupyMPSBackend"))
register_backend("singularity", _make_lazy("superfermion.backends.singularity", "SingularityBackend"))
register_backend("dwave", _make_lazy("superfermion.backends.dwave", "DWaveBackend"))
register_backend("supremacy", _make_lazy("superfermion.backends.supremacy_core", "SupremacyBackend"))
register_backend("density_matrix", _make_lazy("superfermion.backends.density_matrix", "DensityMatrixBackend"))
register_backend("stabilizer", _make_lazy("superfermion.backends.stabilizer", "StabilizerBackend"))

# Convenience aliases
register_backend("simulator", _make_lazy("superfermion.backends.simulator", "StatevectorBackend"))
register_backend("auto", lambda: get_backend(None))
