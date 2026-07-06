"""LocalSimulatorProvider — wraps simulation backends behind the Provider interface."""

from __future__ import annotations

from typing import Any, Dict, Optional

import superfermion as sf
from superfermion.runtime.providers.base import Provider


class LocalSimulatorProvider(Provider):
    """Local simulation provider wrapping Superfermion backends.

    Usage:
        >>> provider = LocalSimulatorProvider(backend="jax")
        >>> job_id = provider.submit(circuit, shots=1000)
        >>> result = provider.result(job_id)
    """

    def __init__(self, backend: str = "statevector"):
        from superfermion.backends.registry import get_backend
        self._backend_name = backend
        self._backend = get_backend(backend)
        self._jobs: Dict[str, Any] = {}

    @property
    def provider_name(self) -> str:
        return f"LocalSimulator({self._backend_name})"

    @property
    def max_qubits(self) -> int:
        return getattr(self._backend, "n_qubits", 32)

    @property
    def cost_per_shot(self) -> float:
        return 0.0

    def submit(self, circuit: Any, shots: int = 1000, **kwargs) -> str:
        import uuid
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"circuit": circuit, "shots": shots, "kwargs": kwargs, "status": "QUEUED"}
        return job_id

    def status(self, job_id: str) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return "UNKNOWN"
        return job.get("status", "UNKNOWN")

    def result(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        result = self._backend.run(job["circuit"], shots=job["shots"], **job["kwargs"])
        job["status"] = "COMPLETED"
        return {"counts": result.counts, "statevector": getattr(result, "statevector", None)}

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is not None and job["status"] in ("QUEUED", "RUNNING"):
            job["status"] = "CANCELLED"
            return True
        return False
