"""
IonQDevice — DeviceExecutor adapter for IonQ quantum hardware.

Absorbs logic from the old ``runtime/providers/ionq.py`` and exposes it
through the ``DeviceExecutor`` protocol. Execution blocks until the job
completes (polling the IonQ REST API).

Requires: ``pip install requests``
"""
from __future__ import annotations


from __future__ import annotations

import time
from typing import Any, Optional

from superfermion.devices import DeviceCapabilities


class IonQDeviceExecutor:
    """Executor bound to a specific IonQ target."""

    _BASE_URL = "https://api.ionq.co/v0.3/jobs"

    def __init__(self, api_key: str, target: str) -> None:
        self._api_key = api_key
        self._target = target

    def execute(self, circuit: "Circuit", shots: int = 1024, **kwargs: Any) -> "RunResult":
        import requests
        from superfermion.circuit import Circuit
        from superfermion.results import RunResult
        from superfermion.bridge import to_ionq

        ionq_circuit = to_ionq(circuit)

        target_str = (
            "qpu." + self._target.split(".")[-1]
            if any(x in self._target for x in ("aria", "forte"))
            else "simulator"
        )

        payload = {
            "lang": "json",
            "body": {"qubits": circuit.n_qubits, "circuit": ionq_circuit},
            "target": target_str,
            "shots": shots,
            "name": f"Superfermion_{int(time.time())}",
        }
        headers = {
            "Authorization": f"apiKey {self._api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self._BASE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        job_id = resp.json()["id"]

        timeout = kwargs.pop("timeout", 600)
        counts = self._poll(job_id, headers, timeout)

        return RunResult(
            counts=counts,
            shots=shots,
            metadata={"backend": self._target, "provider": "ionq", "job_id": job_id},
        )

    def _poll(self, job_id: str, headers: dict, timeout: float) -> dict:
        import requests

        start = time.time()
        while True:
            resp = requests.get(f"{self._BASE_URL}/{job_id}", headers=headers)
            resp.raise_for_status()
            info = resp.json()
            status = info.get("status")

            if status == "completed":
                res = requests.get(f"{self._BASE_URL}/{job_id}/results", headers=headers)
                res.raise_for_status()
                return res.json()

            if status in ("failed", "canceled"):
                raise RuntimeError(
                    f"IonQ job {job_id} {status}: {info.get('failure_reason')}"
                )

            if time.time() - start > timeout:
                raise TimeoutError(f"IonQ job {job_id} timed out after {timeout}s")
            time.sleep(2)

    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            max_qubits=25,
            native_gates=["gpi", "gpi2", "ms"],
            skip_fusion=False,
            supports_statevector=False,
            is_simulator=False,
        )


class IonQDevice:
    """Callable factory returning an ``IonQDeviceExecutor`` for a target.

    Usage::

        ionq = IonQDevice(api_key="...")
        result = sf.run(circuit, device=ionq("ionq.aria-1"))
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def __call__(self, target: str = "ionq.aria-1") -> IonQDeviceExecutor:
        if self._api_key is None:
            raise ValueError(
                "IonQDevice requires an api_key. Pass api_key= or set "
                "IONQ_API_KEY in your environment."
            )
        return IonQDeviceExecutor(self._api_key, target)
