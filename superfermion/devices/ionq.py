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


def _ionq_key_to_sf(key: Any, n_qubits: int) -> str:
    """Convert one IonQ result key into an SF-order bitstring.

    IonQ's ``/results`` payload keys are integers whose binary form has
    qubit 0 as the most significant bit, so ``3`` on 3 qubits means
    ``q0=0, q1=1, q2=1``.  SF stores q0 last (see ``results.py``).
    Strings that are already ``n_qubits`` wide are treated as bitstrings in
    the same q0-first convention.
    """
    text = str(key)
    if len(text) == n_qubits and set(text) <= {"0", "1"}:
        bits = text
    else:
        bits = format(int(text), f"0{n_qubits}b")
    return bits[::-1]


def _extract_ionq_counts(payload: Any, shots: int, n_qubits: int) -> dict:
    """Turn an IonQ ``/results`` document into SF counts.

    The adapter used to hand the entire JSON document to ``RunResult`` as
    ``counts``, so ``result.counts`` held ``{'id','status','histogram',...}``
    and ``expectation()`` died with ``TypeError: unsupported operand type(s)
    for +: 'int' and 'str'``.  Accept the documented ``histogram``, fall back
    to ``probabilities`` scaled by the submitted shots, and normalise the
    little/big-endian difference at the same time.
    """
    if isinstance(payload, dict) and "histogram" in payload:
        source = payload.get("histogram") or {}
    elif isinstance(payload, dict) and "probabilities" in payload:
        probs = payload.get("probabilities") or {}
        total = sum(float(v) for v in probs.values()) or 1.0
        source = {k: float(v) / total * shots for k, v in probs.items()}
    elif isinstance(payload, dict) and payload and all(
        isinstance(v, (int, float)) for v in payload.values()
    ):
        # Already a flat counts map (older API or a mock).
        source = payload
    else:
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise RuntimeError(
            f"Could not parse IonQ result: expected a 'histogram' or "
            f"'probabilities' field, got {keys}"
        )

    counts: dict = {}
    for key, value in source.items():
        sf_key = _ionq_key_to_sf(key, n_qubits)
        counts[sf_key] = counts.get(sf_key, 0) + int(round(float(value)))
    return counts


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
        payload = self._poll(job_id, headers, timeout)
        counts = _extract_ionq_counts(payload, shots, circuit.n_qubits)

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
