"""Provider ABC — common interface for all quantum hardware providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Provider(ABC):
    """Abstract base class for quantum hardware providers.

    Each concrete provider implements the full lifecycle:
    ``submit()`` → ``status()`` → ``result()`` → ``cancel()``
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'IBM Quantum', 'AWS Braket')."""
        ...

    @property
    @abstractmethod
    def max_qubits(self) -> int:
        """Maximum qubit count supported by this provider."""
        ...

    @property
    @abstractmethod
    def cost_per_shot(self) -> float:
        """Cost in USD per shot."""
        ...

    @abstractmethod
    def submit(self, circuit: Any, shots: int = 1000, **kwargs) -> str:
        """Submit a circuit for execution. Returns a job ID."""
        ...

    @abstractmethod
    def status(self, job_id: str) -> str:
        """Query job status. Returns one of: 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'."""
        ...

    @abstractmethod
    def result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job results."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running/queued job. Returns True if successful."""
        ...

    def is_available(self) -> bool:
        """Check if the provider is reachable and configured."""
        return True
