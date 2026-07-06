"""
OpenQuantumDevice — DeviceExecutor adapter for the OpenQuantum platform.

Absorbs logic from the old ``runtime/providers/openquantum.py`` and exposes
it through the ``DeviceExecutor`` protocol.

Requires: ``pip install openquantum-sdk``
"""

from __future__ import annotations

import time
from typing import Any, Optional

from superfermion.devices import DeviceCapabilities


class OpenQuantumDeviceExecutor:
    """Executor bound to a specific OpenQuantum target."""

    def __init__(self, scheduler: Any, org_id: str, backend_id: str, target_name: str) -> None:
        self._scheduler = scheduler
        self._org_id = org_id
        self._backend_id = backend_id
        self._target_name = target_name

    def execute(self, circuit: "Circuit", shots: int = 1000, **kwargs: Any) -> "RunResult":
        from superfermion.results import RunResult
        from superfermion.bridge import to_qasm

        try:
            from openquantum_sdk.clients import JobSubmissionConfig
        except ImportError as exc:
            raise ImportError(
                "openquantum-sdk not found. Run 'pip install openquantum-sdk'"
            ) from exc

        qasm_str = to_qasm(circuit)

        config = JobSubmissionConfig(
            organization_id=self._org_id,
            backend_class_id=self._backend_id,
            job_subcategory_id="e929e161-bafc-4df8-a68c-0d282e6a7409",
            name=f"SF_{int(time.time())}",
            shots=shots,
            verbose=False,
        )

        job_read = self._scheduler.submit_job(config, file_content=qasm_str.encode("utf-8"))
        job_id = job_read.id

        timeout = kwargs.pop("timeout", 600)
        start = time.time()
        while True:
            job_read = self._scheduler.get_job(job_id)
            status = job_read.status
            if status == "COMPLETED":
                data = self._scheduler.download_job_output(job_read)
                counts = data.get("counts", data.get("meas", {}))
                return RunResult(
                    counts=counts,
                    shots=shots,
                    metadata={
                        "backend": self._target_name,
                        "provider": "openquantum",
                        "job_id": job_id,
                    },
                )
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    f"OpenQuantum job {job_id} {status}: {getattr(job_read, 'message', '')}"
                )
            if time.time() - start > timeout:
                raise TimeoutError(f"OpenQuantum job {job_id} timed out after {timeout}s")
            time.sleep(10)

    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            max_qubits=25,
            skip_fusion=False,
            supports_statevector=False,
            is_simulator=False,
        )


class OpenQuantumDevice:
    """Callable factory returning an ``OpenQuantumDeviceExecutor``.

    Usage::

        oq = OpenQuantumDevice(client_id="...", client_secret="...")
        result = sf.run(circuit, device=oq("ionq"))
    """

    _ALIASES = {"ionq": "forte-1", "rigetti": "ankaa-3", "iqm": "garnet"}

    def __init__(self, client_id: str, client_secret: str) -> None:
        try:
            from openquantum_sdk import SchedulerClient, ManagementClient
            from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
        except ImportError as exc:
            raise ImportError(
                "openquantum-sdk not found. Run 'pip install openquantum-sdk'"
            ) from exc

        creds = ClientCredentials(client_id=client_id, client_secret=client_secret)
        auth = ClientCredentialsAuth(creds=creds)
        self._scheduler = SchedulerClient(auth=auth)
        self._management = ManagementClient(auth=auth)
        self._backend_map = {
            b.name.lower(): b.id
            for b in self._management.list_backend_classes().backend_classes
        }
        orgs = self._management.list_user_organizations().organizations
        if not orgs:
            raise RuntimeError("No OpenQuantum organizations found for this account.")
        self._org_id = orgs[0].id

    def _resolve_backend_id(self, target: str) -> str:
        name = self._ALIASES.get(target.lower(), target.lower())
        for b_name, b_id in self._backend_map.items():
            if name in b_name.lower():
                return b_id
        raise ValueError(
            f"Backend '{target}' not found on OpenQuantum. "
            f"Available: {list(self._backend_map.keys())}"
        )

    def __call__(self, target: str = "ionq") -> OpenQuantumDeviceExecutor:
        backend_id = self._resolve_backend_id(target)
        return OpenQuantumDeviceExecutor(
            self._scheduler, self._org_id, backend_id, target,
        )
