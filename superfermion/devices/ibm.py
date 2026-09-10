"""
IBMDevice — DeviceExecutor adapter for IBM Quantum hardware.

Absorbs the real integration logic from the old ``runtime/providers/ibm.py``
and exposes it through the ``DeviceExecutor`` protocol. Execution is
synchronous (blocks until the job completes).

Requires: ``pip install qiskit-ibm-runtime``
"""
from __future__ import annotations


from __future__ import annotations

import os

from typing import Any, List, Optional

from superfermion.devices import DeviceCapabilities, DeviceExecutor, DeviceInfo


def _ensure_measurements(qc: Any) -> Any:
    """Append ``measure_all()`` when the circuit has no measurement ops.

    IBM SamplerV2 rejects circuits without explicit measurements (runtime
    error 1515). SF circuits carry measurement implicitly (Sampler-style
    semantics), so ``bridge.to_qiskit`` emits none; add a full Z-basis
    measurement before transpilation so real-hardware runs are accepted.
    """
    try:
        has_measure = any(
            getattr(inst.operation, "name", "") == "measure"
            for inst in qc.data
        )
    except Exception:
        return qc  # circuit cannot be introspected (e.g. mocked in tests)
    if not has_measure:
        qc.measure_all()
    return qc


def _normalize_counts_to_sf(counts: dict[str, int], width: int) -> dict[str, int]:
    """Convert IBM SamplerV2 counts to SF's q0-last bitstring convention.

    Verified on real hardware (ibm_fez/ibm_marrakesh, 2026-09):
    ``bridge.to_qiskit`` reverses qubit indices (sf i -> qiskit n-1-i) and
    ``measure_all`` maps creg[i] <- qiskit q[i], so the two reversals cancel
    and SamplerV2 keys come back q0-first (bitstring position i holds SF
    qubit i). SF counts are q0-last (qubit q is bit n-1-q, see results.py),
    so each key of the measured width is reversed. Keys of other widths
    (e.g. exotic multi-register mid-circuit layouts) are left untouched.
    """
    return {
        (k[::-1] if len(k) == width else k): v
        for k, v in (counts or {}).items()
    }


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
        qc = _ensure_measurements(qc)

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(
            optimization_level=kwargs.pop("optimization_level", 3),
            backend=ibmq_backend,
        )
        isa_circuit = pm.run(qc)

        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=ibmq_backend)
        raw_job = sampler.run([isa_circuit], shots=shots)
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

        # SamplerV2 keys are q0-first; SF counts are q0-last. Normalize so
        # all SF counts consumers (results.py, parameter_shift, readout
        # correction) read the documented convention.
        counts = _normalize_counts_to_sf(
            counts, circuit.n_cbits or circuit.n_qubits)

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
            token = self._token or os.getenv("QISKIT_IBM_TOKEN")
            if token is None:
                raise ValueError(
                    "IBMDevice requires a token. Pass token= or set "
                    "QISKIT_IBM_TOKEN in your environment."
                )
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform", token=token,
            )
        return self._service

    def __call__(self, backend_name: str = "ibm_fez") -> IBMDeviceExecutor:
        return IBMDeviceExecutor(self._ensure_service(), backend_name)

    def list_devices(self) -> List[DeviceInfo]:
        """List IBM Quantum backends available to this account.

        Returns SF-style ``DeviceInfo`` summaries (name, n_qubits, status,
        is_simulator) using the same token plumbing as execution
        (``token=`` or the ``QISKIT_IBM_TOKEN`` environment variable).
        """
        service = self._ensure_service()
        infos: List[DeviceInfo] = []
        for backend in service.backends():
            status = backend.status()
            # Older QiskitRuntimeService exposes status().name; newer
            # versions expose .operational (+ status_msg) only.
            status_name = getattr(status, "name", None)
            if status_name is None:
                if hasattr(status, "operational"):
                    status_name = (
                        "operational" if status.operational else "offline")
                else:
                    status_name = str(status)
            infos.append(DeviceInfo(
                name=str(backend.name),
                n_qubits=int(getattr(backend, "num_qubits", -1)),
                status=str(status_name),
                is_simulator=bool(getattr(backend, "simulator", False)),
            ))
        return infos
