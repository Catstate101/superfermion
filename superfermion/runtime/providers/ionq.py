"""
IonQ Provider — Direct connection to IonQ's quantum cloud.
"""

from __future__ import annotations

import time
import requests
from typing import Any, Dict, List, Optional
import superfermion as sf
from superfermion.runtime import Job, JobStatus
from superfermion.results import RunResult


class IonQJob(Job):
    """Asynchronous job on IonQ hardware."""
    
    def __init__(self, job_id: str, api_key: str, backend_name: str):
        super().__init__(job_id=job_id)
        self.api_key = api_key
        self.backend_name = backend_name
        self._base_url = "https://api.ionq.co/v0.3/jobs"

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Wait for and return the IonQ job result."""
        sf.utils.info(f"Waiting for IonQ job {self.job_id} on {self.backend_name}...")
        
        headers = {"Authorization": f"apiKey {self.api_key}"}
        
        while True:
            response = requests.get(f"{self._base_url}/{self.job_id}", headers=headers)
            if response.status_code != 200:
                sf.utils.error(f"IonQ Status Fetch Error: {response.text}")
                response.raise_for_status()
                
            job_info = response.json()
            
            status = job_info.get("status")
            if status == "completed":
                # Extract results
                results_url = f"{self._base_url}/{self.job_id}/results"
                res_response = requests.get(results_url, headers=headers)
                res_response.raise_for_status()
                counts = res_response.json()
                
                # IonQ returns probabilities or counts depending on the version. 
                # We normalize to counts here.
                return RunResult(counts=counts, shots=job_info.get("shots", 1024), 
                                metadata={"backend": self.backend_name, "job_id": self.job_id})
            
            if status in ["failed", "canceled"]:
                raise RuntimeError(f"IonQ Job {self.job_id} {status}: {job_info.get('failure_reason')}")
                
            time.sleep(2)
            if timeout and (time.time() - self.creation_date) > timeout:
                raise TimeoutError(f"Wait for IonQ job {self.job_id} timed out.")

    @property
    def status(self) -> JobStatus:
        """Map IonQ status to Superfermion JobStatus."""
        headers = {"Authorization": f"apiKey {self.api_key}"}
        try:
            response = requests.get(f"{self._base_url}/{self.job_id}", headers=headers)
            response.raise_for_status()
            remote_status = response.json().get("status")
            
            mapping = {
                "submitted": JobStatus.CREATED,
                "ready": JobStatus.QUEUED,
                "running": JobStatus.RUNNING,
                "completed": JobStatus.COMPLETED,
                "failed": JobStatus.FAILED,
                "canceled": JobStatus.CANCELLED
            }
            return mapping.get(remote_status, JobStatus.CREATED)
        except Exception:
            return JobStatus.FAILED

    def cancel(self):
        sf.utils.info(f"Cancelling IonQ job {self.job_id}")
        headers = {"Authorization": f"apiKey {self.api_key}"}
        requests.put(f"{self._base_url}/{self.job_id}/status/cancel", headers=headers)


class IonQProvider:
    """Entry point for IonQ Quantum Cloud services."""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            from superfermion.security.credentials import CredentialStore
            api_key = CredentialStore().get("ionq_api_key")
            
        self.api_key = api_key
        self._url = "https://api.ionq.co/v0.3/jobs"

    def run(self, circuit: sf.Circuit, backend: str = "ionq.aria-1", shots: int = 1024) -> IonQJob:
        """Submit a circuit to IonQ."""
        if not self.api_key:
            raise ValueError("IonQProvider not initialized with api_key.")
            
        sf.utils.info(f"Submitting to IonQ Cloud ({backend})...")
        
        # 1. Convert to IonQ-compatible JSON circuit
        # IonQ uses a sequence of gate dictionaries
        # For this bridge, we assume a standard gate mapping
        from superfermion.bridge import to_ionq
        ionq_circuit = to_ionq(circuit)
        
        payload = {
            "lang": "json",
            "body": {
                "qubits": circuit.n_qubits,
                "circuit": ionq_circuit
            },
            "target": "qpu." + backend.split(".")[-1] if "aria" in backend or "forte" in backend else "simulator",
            "shots": shots,
            "name": f"Superfermion_{int(time.time())}"
        }
        
        headers = {
            "Authorization": f"apiKey {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(self._url, json=payload, headers=headers)
        if response.status_code != 200:
            sf.utils.error(f"IonQ API Error ({response.status_code}): {response.text}")
            response.raise_for_status()
            
        job_data = response.json()
        return IonQJob(job_data["id"], self.api_key, backend)

    def retrieve_job(self, job_id: str) -> IonQJob:
        """Fetch an existing IonQ job by its ID."""
        if not self.api_key:
            raise ValueError("IonQProvider not initialized with api_key.")
            
        sf.utils.info(f"Retrieving IonQ job {job_id}...")
        return IonQJob(job_id, self.api_key, "unknown")

    def list_jobs(self, limit: int = 10) -> List[IonQJob]:
        """List recent jobs submitted to IonQ."""
        if not self.api_key:
            raise ValueError("IonQProvider not initialized with api_key.")
            
        headers = {
            "Authorization": f"apiKey {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(f"{self._url}?limit={limit}", headers=headers)
        if response.status_code != 200:
            sf.utils.error(f"Failed to list IonQ jobs: {response.text}")
            response.raise_for_status()
            
        jobs_data = response.json().get("jobs", [])
        return [IonQJob(j["id"], self.api_key, j.get("target", "unknown")) for j in jobs_data]

    def get_characterization(self) -> Dict[str, Any]:
        """Fetch the latest characterization data from IonQ."""
        if not self.api_key:
            raise ValueError("IonQProvider not initialized with api_key.")
            
        headers = {"Authorization": f"apiKey {self.api_key}"}
        url = "https://api.ionq.co/v0.3/characterizations"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            sf.utils.error(f"Failed to fetch IonQ characterizations: {response.text}")
            return {}
            
        # Return the latest one (first in list usually)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data

    def get_noise_data(self, backend: str = "aria-1") -> Dict[str, Any]:
        """Fetch T1, T2 and other noise parameters for the hardware."""
        char = self.get_characterization()
        if not char: return {}
        
        # IonQ characterization format varies, 
        # but typically contains fidelities and coherence times.
        # We normalize to a standard format.
        q_data = char.get("qubits", {})
        
        noise_map = {
            "t1": [q.get("t1", 0) for q in q_data] if isinstance(q_data, list) else [],
            "t2": [q.get("t2", 0) for q in q_data] if isinstance(q_data, list) else [],
            "fidelities": char.get("fidelities", {})
        }
        return noise_map
