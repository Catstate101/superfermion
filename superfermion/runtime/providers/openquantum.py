"""
OpenQuantum Provider — Access IonQ, IQM, and Rigetti via the OpenQuantum SDK.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import superfermion as sf
from superfermion.runtime import Job, JobStatus
from superfermion.results import RunResult

try:
    from openquantum_sdk import SchedulerClient, ManagementClient
    from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
    from openquantum_sdk.clients import JobSubmissionConfig
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


class OpenQuantumJob(Job):
    """Asynchronous job on the OpenQuantum platform using the official SDK."""
    
    def __init__(self, job_id: str, scheduler: SchedulerClient, target_name: str):
        super().__init__(job_id=job_id)
        self.scheduler = scheduler
        self.target_name = target_name

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Wait for and return the OpenQuantum job result."""
        sf.utils.info(f"Waiting for OpenQuantum job {self.job_id} on {self.target_name}...")
        
        start_time = time.time()
        while True:
            job_read = self.scheduler.get_job(self.job_id)
            status = job_read.status
            
            if status == "COMPLETED":
                # Download results using the SDK's built-in downloader
                data = self.scheduler.download_job_output(job_read)
                # Results are typically in a 'counts' or 'meas' field
                counts = data.get("counts", data.get("meas", {}))
                return RunResult(counts=counts, shots=1024) # Shots handled during submission
            
            if status in ["FAILED", "CANCELLED"]:
                raise RuntimeError(f"OpenQuantum Job {self.job_id} {status}: {job_read.message}")
                
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"OpenQuantum job {self.job_id} timed out.")
                
            time.sleep(10)

    @property
    def status(self) -> JobStatus:
        """Map SDK status string to JobStatus."""
        try:
            job_read = self.scheduler.get_job(self.job_id)
            mapping = {
                "CREATED": JobStatus.CREATED,
                "QUEUED": JobStatus.QUEUED,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELLED": JobStatus.CANCELLED
            }
            return mapping.get(job_read.status, JobStatus.QUEUED)
        except:
            return JobStatus.FAILED

    def cancel(self):
        self.scheduler.cancel_job(self.job_id)


class OpenQuantumProvider:
    """Entry point for OpenQuantum unified cloud services via SDK."""
    
    def __init__(self, client_id: str, client_secret: str):
        if not SDK_AVAILABLE:
            raise ImportError("openquantum-sdk not found. Run 'pip install openquantum-sdk'")
            
        self.creds = ClientCredentials(client_id=client_id, client_secret=client_secret)
        self.auth = ClientCredentialsAuth(creds=self.creds)
        self.scheduler = SchedulerClient(auth=self.auth)
        self.management = ManagementClient(auth=self.auth)
        
        # Hardware Mapping (Dynamic discovery)
        self._backend_map = {b.name.lower(): b.id for b in self.management.list_backend_classes().backend_classes}
        
    def _get_backend_id(self, target: str) -> str:
        """Map human name (e.g. 'ionq') to OpenQuantum UUID."""
        # Common aliases
        aliases = {
            "ionq": "forte-1",
            "rigetti": "ankaa-3",
            "iqm": "garnet"
        }
        name = aliases.get(target.lower(), target.lower())
        
        for b_name, b_id in self._backend_map.items():
            if name in b_name.lower():
                return b_id
        raise ValueError(f"Backend '{target}' not found on OpenQuantum. Available: {list(self._backend_map.keys())}")

    def run(self, circuit: sf.Circuit, target: str = "ionq", shots: int = 1000) -> OpenQuantumJob:
        """Submit a circuit using the SDK's submit_job (One-Call)."""
        backend_id = self._get_backend_id(target)
        
        # 1. Convert to QASM
        from superfermion.bridge import to_qasm
        qasm_str = to_qasm(circuit)
        
        # 2. Get Organization
        orgs = self.management.list_user_organizations().organizations
        if not orgs:
            raise RuntimeError("No OpenQuantum organizations found for this account.")
        org_id = orgs[0].id
        
        # 3. Use Scheduler SDK to submit
        config = JobSubmissionConfig(
            organization_id=org_id,
            backend_class_id=backend_id,
            job_subcategory_id="e929e161-bafc-4df8-a68c-0d282e6a7409", # 'Other (Specify)' category
            name=f"SF_Discovery_{int(time.time())}",
            shots=shots,
            verbose=False
        )
        
        sf.utils.info(f"Dispatching to OpenQuantum QPU: {target} (Backend: {backend_id})")
        job_read = self.scheduler.submit_job(config, file_content=qasm_str.encode("utf-8"))
        
        return OpenQuantumJob(job_read.id, self.scheduler, target)

    def retrieve_job(self, job_id: str) -> OpenQuantumJob:
        return OpenQuantumJob(job_id, self.scheduler, "unknown")

    def list_jobs(self, limit: int = 10) -> List[OpenQuantumJob]:
        """List recent jobs from OpenQuantum."""
        orgs = self.management.list_user_organizations().organizations
        if not orgs:
            return []
        org_id = orgs[0].id
        
        paginated = self.scheduler.list_jobs(organization_id=org_id, limit=limit)
        return [OpenQuantumJob(j.id, self.scheduler, j.backend_class_id) for j in paginated.jobs]
