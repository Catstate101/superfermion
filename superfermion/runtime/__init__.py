"""
Superfermion Runtime — Unified interface for local and remote quantum execution.

This module provides the 'Job' and 'Runtime' classes to handle asynchronous 
execution, cloud dispatch, and queue management for any QPU in the world.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

import superfermion as sf
from superfermion.results import RunResult


class JobStatus(Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(ABC):
    """Represents a quantum execution task."""
    
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id or str(uuid.uuid4())
        self.creation_date = time.time()

    @property
    @abstractmethod
    def status(self) -> JobStatus:
        """Return the current status of the job."""
        pass
    
    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Wait for and return the result of the job."""
        pass
        
    @abstractmethod
    def cancel(self):
        """Cancel the job."""
        pass


class LocalJob(Job):
    """A job that runs immediately on a local backend."""
    def __init__(self, result_obj: RunResult):
        super().__init__()
        self._result = result_obj
        self._status = JobStatus.COMPLETED
        
    @property
    def status(self) -> JobStatus:
        return self._status
        
    def result(self, timeout: Optional[float] = None) -> RunResult:
        return self._result
        
    def cancel(self):
        pass


class Runtime:
    """The central hub for managing hardware connections and job dispatch."""
    
    def __init__(self):
        self._connections: Dict[str, Any] = {}
        self._active_jobs: Dict[str, Job] = {}
        
    def connect(self, provider: str, **credentials):
        """Connect to a cloud provider (e.g., 'ibm', 'aws', 'pasqal').
        
        Usage:
            sf.runtime.connect('ibm', token='YOUR_TOKEN')
        """
        # This will be populated as we build provider bridges
        self._connections[provider] = credentials
        sf.utils.info(f"Connected to {provider.upper()} provider.")
        
    def run(self, circuit: sf.Circuit, backend: Union[str, Any] = "jax", shots: int = 1000, **kwargs) -> Job:
        """Submit a circuit for execution and return a Job.
        
        This is the preferred entry point for world-class, multi-backend execution.
        """
        # 1. Handle Remote Backends (e.g. IBM)
        if isinstance(backend, str) and backend.startswith("ibm_"):
            ibm_creds = self._connections.get("ibm")
            if not ibm_creds or "token" not in ibm_creds:
                raise ValueError("Not connected to IBM. Run sf.runtime.connect('ibm', token='...') first.")
            
            from superfermion.runtime.providers.ibm import IBMProvider
            provider = IBMProvider(token=ibm_creds["token"])
            job = provider.run(circuit, backend=backend, shots=shots)
            self._active_jobs[job.job_id] = job
            return job

        # 1b. Handle IonQ Remote Backends
        if isinstance(backend, str) and backend.startswith("ionq."):
            ionq_creds = self._connections.get("ionq")
            if not ionq_creds or "api_key" not in ionq_creds:
                raise ValueError("Not connected to IonQ. Run sf.runtime.connect('ionq', api_key='...') first.")
            
            from superfermion.runtime.providers.ionq import IonQProvider
            provider = IonQProvider(api_key=ionq_creds["api_key"])
            job = provider.run(circuit, backend=backend, shots=shots)
            self._active_jobs[job.job_id] = job
            return job

        # 1c. Handle OpenQuantum Remote Backends
        if isinstance(backend, str) and backend.startswith("oq."):
            oq_creds = self._connections.get("openquantum")
            if not oq_creds or ("client_id" not in oq_creds and "api_key" not in oq_creds):
                raise ValueError("Not connected to OpenQuantum. Run sf.runtime.connect('openquantum', client_id='...', client_secret='...') first.")
            
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            target = backend.replace("oq.", "") # e.g. oq.ionq -> ionq
            provider = OpenQuantumProvider(
                client_id=oq_creds.get("client_id", oq_creds.get("api_key")),
                client_secret=oq_creds.get("client_secret", "DUMMY")
            )
            job = provider.run(circuit, target=target, shots=shots)
            self._active_jobs[job.job_id] = job
            return job
        if isinstance(backend, str):
            from superfermion.backends.registry import get_backend
            backend_obj = get_backend(backend)
        else:
            backend_obj = backend
            
        # 3. Dispatch to Local Simulator
        sf.utils.debug(f"Dispatching job to {backend_obj.name if hasattr(backend_obj, 'name') else backend_obj}...")
        
        if hasattr(backend_obj, "run"):
            result = backend_obj.run(circuit, shots=shots, **kwargs)
            job = LocalJob(result)
            self._active_jobs[job.job_id] = job
            return job
        else:
            raise NotImplementedError(f"Backend {backend} (resolved to {backend_obj}) does not support .run() and no remote provider found.")

    def retrieve_job(self, job_id: str, provider: str = "ibm") -> Job:
        """Retrieve an existing job from a remote provider."""
        if provider == "ibm":
            ibm_creds = self._connections.get("ibm")
            if not ibm_creds or "token" not in ibm_creds:
                raise ValueError("Not connected to IBM. Run sf.runtime.connect('ibm', token='...') first.")
            
            from superfermion.runtime.providers.ibm import IBMProvider
            provider_obj = IBMProvider(token=ibm_creds["token"])
            job = provider_obj.retrieve_job(job_id)
            self._active_jobs[job_id] = job
            return job
        elif provider == "ionq":
            ionq_creds = self._connections.get("ionq")
            if not ionq_creds or "api_key" not in ionq_creds:
                raise ValueError("Not connected to IonQ. Run sf.runtime.connect('ionq', api_key='...') first.")
            
            from superfermion.runtime.providers.ionq import IonQProvider
            provider_obj = IonQProvider(api_key=ionq_creds["api_key"])
            job = provider_obj.retrieve_job(job_id)
            self._active_jobs[job_id] = job
            return job
        elif provider == "openquantum":
            oq_creds = self._connections.get("openquantum")
            if not oq_creds or "api_key" not in oq_creds:
                raise ValueError("Not connected to OpenQuantum. Run sf.runtime.connect('openquantum', api_key='...') first.")
            
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            provider_obj = OpenQuantumProvider(api_key=oq_creds["api_key"])
            job = provider_obj.retrieve_job(job_id)
            self._active_jobs[job_id] = job
            return job
        else:
            raise ValueError(f"Retrieval not supported for provider: {provider}")

    def list_jobs(self, provider: str = "ionq", limit: int = 10) -> List[Job]:
        """List recent jobs from a remote provider."""
        if provider == "ionq":
            ionq_creds = self._connections.get("ionq")
            if not ionq_creds or "api_key" not in ionq_creds:
                raise ValueError("Not connected to IonQ. Run sf.runtime.connect('ionq', api_key='...') first.")
            
            from superfermion.runtime.providers.ionq import IonQProvider
            provider_obj = IonQProvider(api_key=ionq_creds["api_key"])
            return provider_obj.list_jobs(limit=limit)
        elif provider == "openquantum":
            oq_creds = self._connections.get("openquantum")
            if not oq_creds or "client_id" not in oq_creds:
                raise ValueError("Not connected to OpenQuantum. Run sf.runtime.connect('openquantum', ...) first.")
            
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            provider_obj = OpenQuantumProvider(
                client_id=oq_creds["client_id"], 
                client_secret=oq_creds["client_secret"]
            )
            return provider_obj.list_jobs(limit=limit)
        else:
            raise NotImplementedError(f"Job listing not yet implemented for provider: {provider}")

# Global runtime instance
runtime = Runtime()

def connect(provider: str, **credentials):
    """Module-level entry for connecting to a provider."""
    return runtime.connect(provider, **credentials)

def run(circuit: sf.Circuit, backend: Union[str, Any] = "jax", shots: int = 1000, **kwargs) -> Job:
    """Module-level entry for dispatching a job."""
    return runtime.run(circuit, backend=backend, shots=shots, **kwargs)

def retrieve_job(job_id: str, provider: str = "ibm") -> Job:
    """Module-level entry for retrieving an existing job."""
    return runtime.retrieve_job(job_id, provider=provider)

def list_jobs(provider: str = "ionq", limit: int = 10) -> List[Job]:
    """Module-level entry for listing recent jobs."""
    return runtime.list_jobs(provider=provider, limit=limit)


# ═══ Job Orchestrator (2026-05-28) ═══
from superfermion.runtime.orchestrator import JobOrchestrator, OrchestratorResult

# ═══ Cloud Scheduler (2026-05-28) ═══
from superfermion.runtime.scheduler import (
    CloudScheduler,
    SchedulerJob,
    JobPriority,
    SchedulingPolicy,
    BackendRegistration,
    BatchResult,
    get_scheduler,
    submit as scheduler_submit,
)

__all__ = [
    "JobOrchestrator",
    "OrchestratorResult",
    "CloudScheduler",
    "SchedulerJob",
    "JobPriority",
    "SchedulingPolicy",
    "BackendRegistration",
    "BatchResult",
    "get_scheduler",
    "scheduler_submit",
]
