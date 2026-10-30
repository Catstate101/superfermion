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

    # A partial measurement is the second silent-corruption case: SF result
    # keys are full-width, so a circuit that measures only q0 delivers
    # ``P(q) = [0.02, 0.0, 0.0]`` for the untouched qubits -- they are absent
    # from the bitstring, and ``expectation`` reads them as 0 (measured on
    # 0.1.22 against FakeSherbrooke).  Measuring every qubit on a copy leaves
    # the supplied measurements in place (same basis, same final state) and
    # makes the ``meas`` register available with full width.
    try:
        measured = {
            qc.find_bit(inst.qubits[0]).index
            for inst in qc.data
            if getattr(inst.operation, "name", "") == "measure"
        }
    except Exception:
        return qc
    if len(measured) < qc.num_qubits:
        completed = qc.copy()
        completed.measure_all()
        return completed
    return qc


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

        try:
            # SamplerV2 is deprecated in qiskit-ibm-runtime >= 0.4x in favour
            # of ``executor_sampler.Sampler``; keep the old import first and
            # fall back so the adapter survives its removal.  Both take
            # ``(mode=...)`` and return ``data.<reg>.get_counts()``.
            from qiskit_ibm_runtime import SamplerV2 as Sampler
        except Exception:  # pragma: no cover - future runtime
            from qiskit_ibm_runtime.executor_sampler import Sampler
        sampler = Sampler(mode=ibmq_backend)
        raw_job = sampler.run([isa_circuit], shots=shots)
        try:
            job_id = str(raw_job.job_id())
        except Exception:  # mocked or older runtime
            job_id = None
        # ``timeout`` was previously swallowed into **kwargs, so a job queued
        # behind a long backlog blocked forever with no job id in hand.  Pass
        # it through when the runtime supports it.
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            remote_res = raw_job.result()
        else:
            try:
                remote_res = raw_job.result(timeout=timeout)
            except TypeError:  # runtime without the timeout keyword
                remote_res = raw_job.result()
            except Exception as exc:
                # The wait aborted but the job keeps running server-side, and
                # the runtime's own error carries no handle.  Re-raise with the
                # id we already captured so the caller can poll or cancel the
                # job instead of hunting for it in the dashboard.
                if "Timeout" not in type(exc).__name__:
                    raise
                handle = f"IBM job {job_id}" if job_id else "IBM job"
                raise type(exc)(
                    f"{handle} still running after {timeout}s (not cancelled). "
                    "Poll or cancel it with that job id."
                ) from exc

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

        # No key transformation here. ``to_qiskit`` keeps qubit labels
        # unchanged (SF and Qiskit both use little-endian, q0 = LSB) and
        # ``_ensure_measurements`` maps creg[q] <- qubit q, so a SamplerV2 key
        # already holds SF qubit q at position n-1-q -- exactly SF's documented
        # q0-last order (see results.py). A reversal used to live here to
        # compensate for the bridge's n-1-q reversal, which 59f29bc removed;
        # left in place it mirrored every multi-qubit hardware result.
        return RunResult(
            counts=counts,
            shots=shots,
            metadata={
                "backend": self._backend_name,
                "provider": "ibm",
                "job_id": job_id,
            },
        )

    def capabilities(self) -> DeviceCapabilities:
        # Was hardcoded to 127, which under-reports 156-qubit Herons and
        # over-reports smaller devices.  Ask the backend, keep 127 as the
        # offline fallback so constructors stay usable without a network.
        max_qubits = 127
        try:
            reported = int(getattr(
                self._service.backend(self._backend_name), "num_qubits", 0))
            if reported > 0:
                max_qubits = reported
        except Exception:
            pass
        return DeviceCapabilities(
            max_qubits=max_qubits,
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
