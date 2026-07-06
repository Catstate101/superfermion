"""
LocalDevice — wraps existing simulation backends as a ``DeviceExecutor``.

This is the primary device for local simulation. It delegates execution to
whichever backend is requested via ``factory.get_backend()``.
"""

from __future__ import annotations

from typing import Any

from superfermion.backends.factory import get_backend
from superfermion.circuit import Circuit
from superfermion.devices import DeviceCapabilities, DeviceExecutor
from superfermion.results import RunResult


class LocalDevice:
    """Wraps a simulation backend as a ``DeviceExecutor``.

    Args:
        backend_name: Registered backend string (e.g. ``"statevector"``,
            ``"rust"``, ``"singularity"``).
    """

    def __init__(self, backend_name: str = "singularity") -> None:
        self._backend_name = backend_name
        self._backend = get_backend(backend_name)

    def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        return self._backend.run(circuit, shots=shots, **kwargs)

    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            max_qubits=getattr(self._backend, "n_qubits", 32),
            native_gates=getattr(self._backend, "supported_gates", ["all"]),
            skip_fusion=(self._backend_name in {"stabilizer", "mps"}),
            supports_statevector=True,
            is_simulator=True,
        )

    def __repr__(self) -> str:
        return f"LocalDevice({self._backend_name!r})"
