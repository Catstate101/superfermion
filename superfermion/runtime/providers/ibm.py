"""
IBM Quantum Provider — Direct connection to IBM's quantum cloud.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import superfermion as sf
from superfermion.runtime import Job, JobStatus, LocalJob
from superfermion.results import RunResult


class IBMJob(Job):
    """Asynchronous job on IBM Quantum hardware."""
    
    def __init__(self, remote_job: Any, backend_name: str):
        super().__init__(job_id=remote_job.job_id())
        self._remote_job = remote_job
        self.backend_name = backend_name

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Wait for and return the IBM job result."""
        sf.utils.info(f"Waiting for IBM job {self.job_id} on {self.backend_name}...")
        
        # 1. Fetch result from remote job
        remote_res = self._remote_job.result()
        
        # 2. Extract counts (SamplerV2 format)
        try:
            pub_result = remote_res[0]
            if hasattr(pub_result.data, 'meas'):
                 counts = pub_result.data.meas.get_counts()
            elif hasattr(pub_result.data, 'c'):
                 counts = pub_result.data.c.get_counts()
            else:
                 counts = pub_result.data[next(iter(pub_result.data._fields))].get_counts()
        except Exception as e:
            sf.utils.error(f"Could not parse IBM counts: {e}")
            raise
            
        return RunResult(counts=counts, shots=1024, metadata={"backend": self.backend_name, "job_id": self.job_id})

    @property
    def status(self) -> JobStatus:
        """Map IBM status to Superfermion JobStatus."""
        remote_status = str(self._remote_job.status())
        if "DONE" in remote_status: return JobStatus.COMPLETED
        if "RUNNING" in remote_status: return JobStatus.RUNNING
        if "QUEUED" in remote_status: return JobStatus.QUEUED
        if "ERROR" in remote_status or "FAILED" in remote_status: return JobStatus.FAILED
        if "CANCELLED" in remote_status: return JobStatus.CANCELLED
        return JobStatus.CREATED

    def cancel(self):
        sf.utils.info(f"Cancelling IBM job {self.job_id}")
        self._remote_job.cancel()


class IBMProvider:
    """Entry point for IBM Quantum Cloud services."""
    
    def __init__(self, token: Optional[str] = None):
        if token is None:
            from superfermion.security.credentials import CredentialStore
            token = CredentialStore().get("ibm_token")
            
        self.token = token
        self._service = None
        if token:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    def run(self, circuit: sf.Circuit, backend: str = "ibm_fez", shots: int = 1024) -> IBMJob:
        """Submit a circuit to IBM Quantum via Superfermion-Native runtime."""
        if not self._service:
            raise ValueError("IBMProvider not initialized with token.")
            
        sf.utils.info(f"Submitting to IBM Cloud ({backend}) via Superfermion-Native...")
        
        # 1. Resolve IBM backend
        ibmq_backend = self._service.backend(backend)
        
        # 2. Transpile via bridge
        from superfermion.bridge import to_qiskit
        qc = to_qiskit(circuit)
        
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=3, backend=ibmq_backend)
        isa_circuit = pm.run(qc)
        
        # 3. Submit
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=ibmq_backend)
        raw_job = sampler.run([isa_circuit])
        
        return IBMJob(raw_job, backend)

    def get_noise_data(self, backend: str, n_qubits: Optional[int] = None) -> Dict[str, Any]:
        """Fetch real-time error data for QEC logic.
        
        Args:
            backend: Name of the IBM backend.
            n_qubits: Number of qubits to fetch (default: all).
        """
        if not self._service: return {}
        ibmq_backend = self._service.backend(backend)
        props = ibmq_backend.properties()
        
        total = ibmq_backend.num_qubits
        if n_qubits is not None:
            total = min(total, n_qubits)
        
        def safe_get(fn, i):
            try:
                v = fn(i)
                return v if v is not None else 0.0
            except Exception:
                return 0.0
        
        noise_map = {
            "readout_error": [safe_get(props.readout_error, i) for i in range(total)],
            "t1": [safe_get(props.t1, i) for i in range(total)],
            "t2": [safe_get(props.t2, i) for i in range(total)],
            "num_qubits": ibmq_backend.num_qubits,
        }
        return noise_map

    def list_backends(self) -> List[Dict[str, Any]]:
        """List available IBM Quantum backends with key specs."""
        if not self._service:
            return []
        backends = self._service.backends()
        result = []
        for b in backends:
            try:
                st = b.status()
                result.append({
                    "name": b.name,
                    "num_qubits": b.num_qubits,
                    "pending_jobs": getattr(st, 'pending_jobs', None),
                    "status_msg": str(st.status_msg) if st and hasattr(st, 'status_msg') else "unknown",
                })
            except Exception:
                result.append({"name": b.name, "num_qubits": getattr(b, 'num_qubits', '?')})
        return result

    def retrieve_job(self, job_id: str) -> IBMJob:
        """Fetch an existing IBM job by its ID and wrap it in an IBMJob object."""
        if not self._service:
            raise ValueError("IBMProvider not initialized with token.")
            
        sf.utils.info(f"Retrieving IBM job {job_id}...")
        raw_job = self._service.job(job_id)
        # We need to find the backend name. 
        # In Qiskit Runtime, the job object usually has it or we can try to get it.
        backend_name = raw_job.backend().name if hasattr(raw_job, 'backend') else "unknown"
        
        return IBMJob(raw_job, backend_name)
