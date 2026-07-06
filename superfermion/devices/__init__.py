"""
Device abstraction layer — protocol-driven device executors.

Provides the ``DeviceExecutor`` protocol for pluggable quantum execution
targets, ``DeviceCapabilities`` for introspection, and ``_resolve_builtin``
for resolving shorthand strings like ``"cpu"`` and ``"gpu"`` to concrete
backend-backed devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from superfermion.circuit import Circuit
    from superfermion.results import RunResult


@dataclass
class DeviceCapabilities:
    """Describes what a device can do, enabling the runner to adapt."""

    max_qubits: int = 32
    native_gates: List[str] = field(default_factory=lambda: ["all"])
    coupling_map: Optional[List[tuple]] = None
    skip_fusion: bool = False
    supports_statevector: bool = True
    is_simulator: bool = True


@runtime_checkable
class DeviceExecutor(Protocol):
    """Protocol that all device executors must satisfy.

    Any object with ``execute()`` and ``capabilities()`` can serve as
    a device target for ``sf.run()``.
    """

    def execute(self, circuit: "Circuit", shots: int = 1000, **kwargs: Any) -> "RunResult":
        """Execute a circuit and return results synchronously."""
        ...

    def capabilities(self) -> DeviceCapabilities:
        """Return device capabilities for the runner to introspect."""
        ...


def _resolve_builtin(name: str) -> "DeviceExecutor":
    """Resolve a shorthand device string to a concrete ``DeviceExecutor``.

    Supported shorthands:
        - ``"cpu"`` → SingularityBackend (auto-routing local simulator)
        - ``"gpu"`` → JAX or CuPy backend (falls back to singularity)
        - Any other registered backend name → wrapped in ``LocalDevice``

    Raises:
        ValueError: If the name cannot be resolved.
    """
    from superfermion.devices.local import LocalDevice

    CPU_ALIASES = {"cpu", "singularity", "auto"}
    GPU_ALIASES = {"gpu", "cuda", "cupy"}

    lower = name.lower()
    if lower in CPU_ALIASES:
        return LocalDevice("singularity")
    if lower in GPU_ALIASES:
        try:
            return LocalDevice("jax")
        except Exception:
            try:
                return LocalDevice("cupy")
            except Exception:
                return LocalDevice("singularity")

    return LocalDevice(lower)


__all__ = [
    "DeviceExecutor",
    "DeviceCapabilities",
    "_resolve_builtin",
]
