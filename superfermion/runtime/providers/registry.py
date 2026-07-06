"""ProviderRegistry — maps provider names to Provider instances."""

from __future__ import annotations

from typing import Dict, Optional

from superfermion.runtime.providers.base import Provider
from superfermion.runtime.providers.local_provider import LocalSimulatorProvider


class ProviderRegistry:
    """Central registry for quantum hardware and simulator providers."""

    _providers: Dict[str, Provider] = {}

    @classmethod
    def register(cls, name: str, provider: Provider) -> None:
        cls._providers[name] = provider

    @classmethod
    def get(cls, name: str) -> Optional[Provider]:
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> list:
        return sorted(cls._providers.keys())

    @classmethod
    def get_or_create(cls, name: str) -> Optional[Provider]:
        """Get a provider by name, lazily creating it if possible."""
        if name in cls._providers:
            return cls._providers[name]

        # Lazy-load hardware providers
        if name in ("ibm", "ibm_eagle"):
            from superfermion.runtime.providers.ibm_provider import IBMProvider
            provider = IBMProvider()
        elif name in ("aws", "braket"):
            from superfermion.runtime.providers.aws_provider import AWSProvider
            provider = AWSProvider()
        elif name in ("ionq", "ionq_aria"):
            from superfermion.runtime.providers.ionq_provider import IonQProvider
            provider = IonQProvider()
        else:
            return None

        cls._providers[name] = provider
        return provider


# Register local simulator providers
ProviderRegistry.register("statevector", LocalSimulatorProvider("statevector"))
ProviderRegistry.register("jax", LocalSimulatorProvider("jax"))
ProviderRegistry.register("mps", LocalSimulatorProvider("mps"))
