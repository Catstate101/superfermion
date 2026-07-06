"""
Declarative backend factory with lazy instantiation.

Backends are registered via a dict mapping BackendName → factory callable.
This replaces the 22-branch if-elif chain in the old registry.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from superfermion.backends.base import Backend
from superfermion.backends.names import BackendName, resolve_backend_name

# -- Registry: name → zero-arg callable that returns a Backend instance --
# Keys are strings (not BackendName) so plugin backends can register freely.
_BACKEND_REGISTRY: Dict[str, Callable[[], Backend]] = {}


def register_backend(name: BackendName | str, factory: Callable[[], Backend]) -> None:
    """Register a backend factory callable for the given name.

    If *name* is a known ``BackendName``, it is canonicalized. Plugin backends
    with custom string names are also accepted.
    """
    key = _normalize(name)
    _BACKEND_REGISTRY[key] = factory


def get_backend(name: BackendName | str | None = None) -> Backend:
    """Retrieve a backend instance by name.

    If *name* is None, auto-selects CUDA if available, otherwise STATEVECTOR.
    """
    if name is None:
        name = _auto_select().value
    key = _normalize(name)

    if key not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Backend '{key}' is not registered. "
            f"Registered: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[key]()


def list_backends() -> List[str]:
    """Return all registered backend names."""
    return list(_BACKEND_REGISTRY.keys())


def _normalize(name: BackendName | str) -> str:
    """Convert a name to its canonical string form.

    Known ``BackendName`` enum values are canonicalized. Plugin-provided
    strings pass through as-is.
    """
    if isinstance(name, BackendName):
        return name.value
    # Check if it can be resolved to a known canonical name
    try:
        return resolve_backend_name(name).value
    except ValueError:
        return name  # plugin backends pass through


def _auto_select() -> BackendName:
    """Auto-select best available backend."""
    try:
        import cupy  # noqa: F401
        return BackendName.CUDA
    except ImportError:
        return BackendName.STATEVECTOR


def _make_lazy(module: str, classname: str) -> Callable[[], Backend]:
    """Return a zero-arg callable that imports and instantiates on first access."""
    import importlib

    def factory() -> Backend:
        mod = importlib.import_module(module)
        cls = getattr(mod, classname)
        return cls()

    return factory


# -- Register core backends (lazy imports via _make_lazy) --

register_backend(BackendName.STATEVECTOR, _make_lazy("superfermion.backends.simulator", "StatevectorBackend"))
register_backend(BackendName.JAX, _make_lazy("superfermion.backends.jax_sim", "JAXBackend"))
register_backend(BackendName.RUST, _make_lazy("superfermion.backends.rust_sim", "RustBackend"))
register_backend(BackendName.CUDA, _make_lazy("superfermion.backends.cuda", "CUSimulatorBackend"))
register_backend(BackendName.CUPY, _make_lazy("superfermion.backends.cupy_sim", "CupyBackend"))
register_backend(BackendName.MPS, _make_lazy("superfermion.backends.mps", "MPSSimulatorBackend"))
register_backend(BackendName.CLUSTER, _make_lazy("superfermion.backends.cluster", "DistributedJAXBackend"))
register_backend(BackendName.JAX_MPS, _make_lazy("superfermion.backends.jax_mps", "JAXMPSBackend"))
register_backend(BackendName.CUDA_MPS, _make_lazy("superfermion.backends.cuda_mps", "CupyMPSBackend"))
register_backend(BackendName.SINGULARITY, _make_lazy("superfermion.backends.singularity", "SingularityBackend"))
register_backend(BackendName.DWAVE, _make_lazy("superfermion.backends.dwave", "DWaveBackend"))
register_backend(BackendName.SUPREMACY, _make_lazy("superfermion.backends.supremacy_core", "SupremacyBackend"))
register_backend(BackendName.DENSITY_MATRIX, _make_lazy("superfermion.backends.density_matrix", "DensityMatrixBackend"))
register_backend(BackendName.STABILIZER, _make_lazy("superfermion.backends.stabilizer", "StabilizerBackend"))

# Convenience aliases
register_backend(BackendName.SIMULATOR, _make_lazy("superfermion.backends.simulator", "StatevectorBackend"))
register_backend(BackendName.AUTO, lambda: get_backend(None))
