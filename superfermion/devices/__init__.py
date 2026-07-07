"""
Device abstraction layer — protocol-driven device executors.

Provides the ``DeviceExecutor`` protocol for pluggable quantum execution
targets and ``DeviceCapabilities`` for introspection.

Resolution rules (used by ``sf.run()``):
    - ``"cpu"`` → RustDevice (CPU, Rayon+AVX statevector simulation)
    - ``"gpu"`` → RustDevice (CUDA GPU statevector simulation)
    - Any ``DeviceExecutor`` object → used directly (IBMDevice, IonQDevice, etc.)
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

    Only two strings are supported:
        - ``"cpu"`` → RustDevice with hardware=cpu
        - ``"gpu"`` → RustDevice with hardware=gpu

    Raises:
        ValueError: If the name is not recognized.
    """
    from superfermion.devices.rust_device import RustDevice

    lower = name.lower().strip()

    if lower == "cpu":
        return RustDevice(hardware="cpu")
    elif lower == "gpu":
        return RustDevice(hardware="gpu")
    else:
        raise ValueError(
            f"Unknown device '{name}'. Use 'cpu', 'gpu', or pass a DeviceExecutor object.\n"
            f"  Examples:\n"
            f"    sf.run(circuit, device='cpu')           # local CPU simulation\n"
            f"    sf.run(circuit, device='gpu')           # local GPU simulation\n"
            f"    sf.run(circuit, device=ibm('ibm_fez')) # QPU via provider object"
        )


__all__ = [
    "DeviceExecutor",
    "DeviceCapabilities",
    "_resolve_builtin",
]
