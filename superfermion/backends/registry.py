"""BackendRegistry — thin compatibility shim delegating to the declarative factory."""

from __future__ import annotations

from typing import Dict, List, Optional

from superfermion.backends.base import Backend
from superfermion.backends.names import BackendName, resolve_backend_name
from superfermion.backends.factory import (
    _BACKEND_REGISTRY,
    register_backend,
    get_backend as _factory_get_backend,
    list_backends as _factory_list_backends,
)


class BackendRegistry:
    """Singleton registry — delegates to declarative factory module.

    Retained for backwards compatibility. New code should use
    ``superfermion.backends.factory.get_backend()`` directly.
    """

    _default_backend_name: str = BackendName.STATEVECTOR.value
    _backends: Dict[str, Backend] = {}

    @classmethod
    def register(cls, backend_name: str | BackendName, backend: Backend) -> None:
        """Register a pre-instantiated backend."""
        key = resolve_backend_name(backend_name)
        cls._backends[key.value] = backend
        register_backend(key, lambda b=backend: b)

    @classmethod
    def get_backend(cls, name: Optional[str | BackendName] = None) -> Backend:
        """Retrieve a backend by name. Delegates to factory."""
        # Check legacy pre-instantiated cache first
        if isinstance(name, str) and name in cls._backends:
            return cls._backends[name]
        if isinstance(name, BackendName) and name.value in cls._backends:
            return cls._backends[name.value]

        # Handle legacy compatibility aliases
        if isinstance(name, str):
            if name in ("ibm", "ibm_eagle", "trapped_ion", "rigetti", "aws", "braket"):
                return cls.get_backend(None)
            if name == "simulator":
                return cls.get_backend(BackendName.STATEVECTOR)

        return _factory_get_backend(name)

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names."""
        names = list(_factory_list_backends())  # factory returns strings now
        names.extend(cls._backends.keys())
        return sorted(set(names))

    @classmethod
    def set_default(cls, name: str | BackendName) -> None:
        """Set the default backend."""
        key = resolve_backend_name(name) if isinstance(name, str) else name
        cls._default_backend_name = key.value


# Module-level convenience functions (preserve existing public API)
def get_backend(name: Optional[str] = None) -> Backend:
    return BackendRegistry.get_backend(name)


def list_backends() -> List[str]:
    return BackendRegistry.list_backends()
