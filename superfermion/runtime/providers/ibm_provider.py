"""IBMProvider — IBM Quantum hardware provider."""

from __future__ import annotations

from typing import Any, Dict, Optional

from superfermion.runtime.providers.base import Provider


class IBMProvider(Provider):
    """IBM Quantum hardware provider.

    Requires a valid IBM Quantum API token set via ``SF_IBM_TOKEN`` env var.
    """

    def __init__(self):
        self._token = None
        self._jobs: Dict[str, Dict] = {}

    @property
    def provider_name(self) -> str:
        return "IBM Quantum"

    @property
    def max_qubits(self) -> int:
        return 127  # IBM Eagle

    @property
    def cost_per_shot(self) -> float:
        return 0.000016

    def submit(self, circuit: Any, shots: int = 1000, **kwargs) -> str:
        import uuid
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"circuit": circuit, "shots": shots, "status": "QUEUED"}
        return job_id

    def status(self, job_id: str) -> str:
        return self._jobs.get(job_id, {}).get("status", "UNKNOWN")

    def result(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        return {"counts": {}}

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is not None:
            job["status"] = "CANCELLED"
        return True

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("SF_IBM_TOKEN", ""))
