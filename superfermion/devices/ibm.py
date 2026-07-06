"""
IBMDevice — DeviceExecutor adapter for IBM Quantum hardware.

Absorbs the real integration logic from the old ``runtime/providers/ibm.py``
and exposes it through the ``DeviceExecutor`` protocol. Execution is
synchronous (blocks until the job completes).

Requires: ``pip install qiskit-ibm-runtime``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from superfermion.devices import DeviceCapabilities, DeviceExecutor


class IBMDeviceExecutor:
    """Executor bound to a specific IBM backend."""

    def __init__(self, service: Any, backend_name: str) -> None:
        self._service = service
        self._backend_name = backend_name

    def execute(self, circuit: "Circuit", shots: int = 1024, **kwargs: Any) -> "RunResult":
        from superfermion.circuit import Circuit
        from superfermion.results import RunResult
        from superfermion.bridge import to_qiskit

        ibmq_backend = self._service.backend(self._backend_name)

        qc = to_qiskit(circuit)

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(
            optimization_level=kwargs.pop("optimization_level", 3),
            backend=ibmq_backend,
        )
        isa_circuit = pm.run(qc)

        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=ibmq_backend)
        raw_job = sampler.run([isa_circuit])
        remote_res = raw_job.result()

        try:
            pub_result = remote_res[0]
            if hasattr(pub_result.data, "meas"):
                counts = pub_result.data.meas.get_counts()
            elif hasattr(pub_result.data, "c"):
                counts = pub_result.data.c.get_counts()
            else:
                counts = pub_result.data[next(iter(pub_result.data._fields))].get_counts()
        except Exception as exc:
            raise RuntimeError(f"Could not parse IBM result: {exc}") from exc

        return RunResult(
            counts=counts,
            shots=shots,
            metadata={"backend": self._backend_name, "provider": "ibm"},
        )

    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            max_qubits=127,
            skip_fusion=False,
            supports_statevector=False,
            is_simulator=False,
        )


class IBMDevice:
    """Callable factory that returns an ``IBMDeviceExecutor`` for a specific backend.

    Usage::

        ibm = IBMDevice(token="...")
        result = sf.run(circuit, device=ibm("ibm_fez"))
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token
        self._service: Any = None

    def _ensure_service(self) -> Any:
        if self._service is None:
            if self._token is None:
                raise ValueError(
                    "IBMDevice requires a token. Pass token= or set "
                    "QISKIT_IBM_TOKEN in your environment."
                )
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform", token=self._token,
            )
        return self._service

    def __call__(self, backend_name: str = "ibm_fez") -> IBMDeviceExecutor:
        return IBMDeviceExecutor(self._ensure_service(), backend_name)
