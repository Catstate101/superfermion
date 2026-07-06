"""
Cloud Job Scheduler — Distributed quantum job execution engine.

Provides async priority-queue scheduling with support for:
- Multi-provider dispatch (IBM, IonQ, local simulators)
- Priority-based queueing (CRITICAL → LOW)
- Batch job submission
- Cost-aware routing policy
- Concurrent worker pool
- Job dependency DAGs
- Result persistence and retrieval

Usage:
    >>> from superfermion.runtime.scheduler import CloudScheduler, JobPriority
    >>>
    >>> scheduler = CloudScheduler(max_workers=4)
    >>> scheduler.register_backend("ibm_brisbane", provider="ibm")
    >>> scheduler.register_backend("jax", provider="local")
    >>>
    >>> job_id = scheduler.submit(circuit, backend="jax", priority=JobPriority.HIGH)
    >>> result = scheduler.wait_for(job_id, timeout=30)
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import heapq
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import superfermion as sf
from superfermion.results import RunResult


# ── Enums and Data Classes ──────────────────────────────────────────────


class JobPriority(Enum):
    """Priority levels for queued jobs (higher = dequeued first)."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BATCH = 1

    def __lt__(self, other: "JobPriority") -> bool:
        return self.value < other.value

    def __le__(self, other: "JobPriority") -> bool:
        return self.value <= other.value


class JobStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchedulingPolicy(Enum):
    """Global scheduling strategy for the CloudScheduler."""
    PRIORITY = "priority"          # Strictly by JobPriority
    COST_AWARE = "cost_aware"      # Cheapest backend first
    ROUND_ROBIN = "round_robin"    # Fair distribution across backends
    LATENCY_FIRST = "latency"      # Fastest (local) backends first


@dataclass(order=True)
class _PrioritizedJob:
    """Internal heap entry. Sorted by (negative priority, submission_time)."""
    sort_key: Tuple[int, float]
    job: "SchedulerJob" = field(compare=False)


@dataclass
class SchedulerJob:
    """A single quantum job tracked by the CloudScheduler.

    Attributes:
        job_id: Unique job identifier (UUID).
        circuit: The quantum circuit to execute.
        backend: Target backend name.
        shots: Number of measurement shots (0 = statevector).
        priority: Queue priority.
        status: Current job status.
        dependencies: Job IDs that must complete before this one.
        policy_hint: Override scheduling policy for this job.
        submission_time: When the job was submitted (epoch).
        start_time: When execution started.
        completion_time: When execution finished.
        result: The RunResult after completion.
        error: Error message if failed.
        metadata: Arbitrary user metadata.
    """
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    circuit: Optional[sf.Circuit] = None
    backend: str = "jax"
    shots: int = 0
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    policy_hint: Optional[SchedulingPolicy] = None
    submission_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    result: Optional[RunResult] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        return 0.0

    @property
    def wait_time(self) -> float:
        if self.start_time:
            return self.start_time - self.submission_time
        return time.time() - self.submission_time


@dataclass
class BackendRegistration:
    """Metadata for a registered backend."""
    name: str
    provider: str = "local"           # "local", "ibm", "ionq"
    max_concurrent: int = 4
    estimated_cost_per_shot: float = 0.0
    average_latency_ms: float = 0.0
    max_qubits: int = 127
    status: str = "online"            # "online", "offline", "busy"
    active_jobs: int = 0

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent - self.active_jobs)


@dataclass
class BatchResult:
    """Aggregated result from a batch submission."""
    job_ids: List[str] = field(default_factory=list)
    results: Dict[str, Optional[RunResult]] = field(default_factory=dict)
    errors: Dict[str, Optional[str]] = field(default_factory=dict)
    total_time_ms: float = 0.0
    completed: int = 0
    failed: int = 0

    @property
    def all_success(self) -> bool:
        return self.failed == 0 and self.completed == len(self.job_ids)


# ── Core Scheduler ──────────────────────────────────────────────────────


class CloudScheduler:
    """Distributed quantum job scheduler with priority queueing.

    Manages a pool of worker threads that execute quantum circuits
    across registered backends. Supports priority ordering, batch
    submission, dependency chains, and cost-aware routing.

    Args:
        max_workers: Maximum concurrent execution threads.
        policy: Default scheduling policy.
        poll_interval: Seconds between queue polls.
    """

    def __init__(
        self,
        max_workers: int = 8,
        policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        poll_interval: float = 0.05,
    ) -> None:
        self.max_workers = max_workers
        self.policy = policy
        self.poll_interval = poll_interval

        # Backend registry
        self._backends: Dict[str, BackendRegistration] = {}
        # Job store
        self._jobs: Dict[str, SchedulerJob] = {}
        # Priority heap of queued jobs
        self._queue: List[_PrioritizedJob] = []
        self._queue_lock = threading.Lock()
        # Thread pool
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._futures: Dict[concurrent.futures.Future, str] = {}
        # Lifecycle
        self._running = threading.Event()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._completion_callbacks: List[Callable[[SchedulerJob], None]] = []

    # ── Backend management ────────────────────────────────────────

    def register_backend(
        self,
        name: str,
        provider: str = "local",
        max_concurrent: int = 4,
        estimated_cost_per_shot: float = 0.0,
        average_latency_ms: float = 0.0,
        max_qubits: int = 127,
    ) -> BackendRegistration:
        """Register a backend for job dispatch.

        Args:
            name: Backend name (matches ``sf.get_backend(name)``).
            provider: ``"local"``, ``"ibm"``, or ``"ionq"``.
            max_concurrent: Max parallel jobs on this backend.
            estimated_cost_per_shot: Estimated cost per shot.
            average_latency_ms: Average round-trip latency.
            max_qubits: Maximum qubits supported.

        Returns:
            The registered ``BackendRegistration``.
        """
        reg = BackendRegistration(
            name=name,
            provider=provider,
            max_concurrent=max_concurrent,
            estimated_cost_per_shot=estimated_cost_per_shot,
            average_latency_ms=average_latency_ms,
            max_qubits=max_qubits,
        )
        self._backends[name] = reg
        return reg

    def unregister_backend(self, name: str) -> None:
        """Remove a backend from the registry."""
        self._backends.pop(name, None)

    def list_backends(self) -> Dict[str, BackendRegistration]:
        """Return all registered backends."""
        return dict(self._backends)

    # ── Job submission ────────────────────────────────────────────

    def submit(
        self,
        circuit: sf.Circuit,
        backend: str = "jax",
        shots: int = 0,
        priority: JobPriority = JobPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        policy_hint: Optional[SchedulingPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a circuit for scheduled execution.

        Args:
            circuit: The quantum circuit.
            backend: Target backend name.
            shots: Number of shots (0 = statevector).
            priority: Queue priority.
            dependencies: Job IDs that must complete first.
            policy_hint: Per-job scheduling override.
            metadata: Arbitrary user metadata.

        Returns:
            The job ID (UUID string).
        """
        job = SchedulerJob(
            circuit=circuit,
            backend=backend,
            shots=shots,
            priority=priority,
            dependencies=dependencies or [],
            policy_hint=policy_hint,
            metadata=metadata or {},
        )

        if job.dependencies:
            if self._dependencies_satisfied(job):
                job.status = JobStatus.QUEUED
            else:
                job.status = JobStatus.PENDING
        else:
            job.status = JobStatus.QUEUED

        self._jobs[job.job_id] = job

        if job.status == JobStatus.QUEUED:
            self._enqueue(job)

        # Auto-start dispatcher if not running
        if not self._running.is_set():
            self.start()

        return job.job_id

    def submit_batch(
        self,
        circuits: List[sf.Circuit],
        backend: str = "jax",
        shots: int = 0,
        priority: JobPriority = JobPriority.BATCH,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchResult:
        """Submit multiple circuits as a batch.

        All circuits share the same backend and priority.
        """
        job_ids = []
        for circuit in circuits:
            jid = self.submit(
                circuit,
                backend=backend,
                shots=shots,
                priority=priority,
                metadata=metadata,
            )
            job_ids.append(jid)

        return BatchResult(job_ids=job_ids)

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the dispatcher and worker pool."""
        if self._running.is_set():
            return

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self._running.set()

        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="sf-scheduler-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()

    def stop(self, wait: bool = True) -> None:
        """Stop the scheduler gracefully.

        Args:
            wait: If True, wait for running jobs to finish.
        """
        self._running.clear()

        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=5.0)

        if self._executor:
            self._executor.shutdown(wait=wait)

    def shutdown(self) -> None:
        """Alias for stop(wait=False)."""
        self.stop(wait=False)

    # ── Job management ────────────────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or queued job.

        Args:
            job_id: The job to cancel.

        Returns:
            True if the job was cancelled.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if job.status in (JobStatus.QUEUED, JobStatus.PENDING):
            job.status = JobStatus.CANCELLED
            # Remove from heap
            with self._queue_lock:
                self._queue = [
                    pj for pj in self._queue if pj.job.job_id != job_id
                ]
                heapq.heapify(self._queue)
            # Unblock dependents
            self._check_dependents(job_id)
            return True

        return False

    def status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a job."""
        job = self._jobs.get(job_id)
        return job.status if job else None

    def result(self, job_id: str) -> Optional[RunResult]:
        """Get the result of a completed job (non-blocking)."""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.COMPLETED:
            return job.result
        return None

    def wait_for(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[RunResult]:
        """Block until a job completes or timeout elapses.

        Args:
            job_id: Job to wait for.
            timeout: Maximum wait in seconds (None = forever).

        Returns:
            RunResult or None if timeout.

        Raises:
            RuntimeError: If the job failed.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        deadline = time.time() + timeout if timeout else float("inf")

        while time.time() < deadline:
            job = self._jobs[job_id]
            if job.status == JobStatus.COMPLETED:
                return job.result
            if job.status == JobStatus.FAILED:
                raise RuntimeError(
                    f"Job {job_id} failed: {job.error}"
                )
            if job.status == JobStatus.CANCELLED:
                return None
            time.sleep(self.poll_interval)

        return None

    def on_completion(
        self,
        callback: Callable[[SchedulerJob], None],
    ) -> None:
        """Register a callback invoked on every job completion."""
        self._completion_callbacks.append(callback)

    # ── Metrics ───────────────────────────────────────────────────

    def metrics(self) -> Dict[str, Any]:
        """Return scheduler-wide metrics.

        Returns:
            Dict with ``total_jobs``, ``queued``, ``running``,
            ``completed``, ``failed``, ``avg_wait_ms``,
            ``avg_duration_ms``, ``backend_loads``.
        """
        status_counts = defaultdict(int)
        total_wait = 0.0
        total_duration = 0.0
        completed_count = 0
        backend_loads = defaultdict(int)

        for job in self._jobs.values():
            status_counts[job.status.value] += 1
            if job.status in (JobStatus.DISPATCHED, JobStatus.RUNNING):
                backend_loads[job.backend] += 1
            if job.status == JobStatus.COMPLETED:
                completed_count += 1
                total_wait += job.wait_time
                total_duration += job.duration

        return {
            "total_jobs": len(self._jobs),
            "queued": status_counts.get("queued", 0),
            "running": status_counts.get("running", 0)
                + status_counts.get("dispatched", 0),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
            "cancelled": status_counts.get("cancelled", 0),
            "avg_wait_ms": (total_wait / max(completed_count, 1)) * 1000,
            "avg_duration_ms": (total_duration / max(completed_count, 1)) * 1000,
            "backend_loads": dict(backend_loads),
            "executor_active": self._executor is not None,
        }

    # ── Internal: Dispatch loop ───────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Main dispatcher loop — polls queue and assigns to workers."""
        while self._running.is_set():
            dispatched = self._dispatch_one()
            if not dispatched:
                time.sleep(self.poll_interval)

    def _dispatch_one(self) -> bool:
        """Try to dispatch one job from the queue. Returns True on success."""
        # Collect currently runnable jobs
        runnable: List[SchedulerJob] = []

        with self._queue_lock:
            if not self._queue:
                return False

            # Check backend availability
            backends_available = {
                name: reg.available_slots
                for name, reg in self._backends.items()
            }
            # Default jax/local always available
            backends_available.setdefault("jax", self.max_workers)
            backends_available.setdefault("statevector", self.max_workers)

            # Select job based on policy
            selected: Optional[_PrioritizedJob] = None
            selected_idx: Optional[int] = None

            for idx, pj in enumerate(self._queue):
                job = pj.job
                backend = job.backend

                # Skip if backend is full
                if backends_available.get(backend, 1) <= 0:
                    continue

                # Skip if dependencies not met
                if not self._dependencies_satisfied(job):
                    continue

                if self.policy == SchedulingPolicy.COST_AWARE:
                    # Compare costs: pick the cheapest backend
                    if selected is None:
                        selected = pj
                        selected_idx = idx
                    else:
                        curr_cost = _get_job_cost(pj.job)
                        best_cost = _get_job_cost(selected.job)
                        if curr_cost < best_cost:
                            selected = pj
                            selected_idx = idx
                    continue

                elif self.policy == SchedulingPolicy.LATENCY_FIRST:
                    # Pick local/fastest backend
                    reg = self._backends.get(job.backend)
                    latency = reg.average_latency_ms if reg else 0.0
                    selected_reg = (
                        self._backends.get(selected.job.backend)
                        if selected else None
                    )
                    selected_latency = (
                        selected_reg.average_latency_ms
                        if selected_reg else 0.0
                    )
                    if latency < selected_latency:
                        selected = pj
                        selected_idx = idx
                    continue

                elif self.policy == SchedulingPolicy.ROUND_ROBIN:
                    if selected is None:
                        selected = pj
                        selected_idx = idx
                    continue

                else:  # PRIORITY
                    if selected is None or pj.sort_key < selected.sort_key:
                        selected = pj
                        selected_idx = idx

            if selected is None or selected_idx is None:
                return False

            # Pop the selected job from heap
            self._queue.pop(selected_idx)
            heapq.heapify(self._queue)

        # Update backend load
        job = selected.job
        backend_name = job.backend
        if backend_name in self._backends:
            self._backends[backend_name].active_jobs += 1

        job.status = JobStatus.DISPATCHED

        # Submit to thread pool
        if self._executor:
            future = self._executor.submit(
                self._execute_job,
                job,
            )
            self._futures[future] = job.job_id

        return True

    def _execute_job(self, job: SchedulerJob) -> None:
        """Execute a single job in a worker thread."""
        job.start_time = time.time()
        job.status = JobStatus.RUNNING

        try:
            if job.circuit is None:
                raise ValueError("Job has no circuit")

            sim = sf.get_backend(job.backend)
            result = sim.run(job.circuit, shots=job.shots)
            job.result = result
            job.status = JobStatus.COMPLETED
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.FAILED
        finally:
            job.completion_time = time.time()

            # Release backend slot
            if job.backend in self._backends:
                self._backends[job.backend].active_jobs = max(
                    0, self._backends[job.backend].active_jobs - 1
                )

            # Fire completion callbacks
            for cb in self._completion_callbacks:
                try:
                    cb(job)
                except Exception:
                    pass

            # Unblock dependent jobs
            self._check_dependents(job.job_id)

    # ── Internal: Dependency management ───────────────────────────

    def _dependencies_satisfied(self, job: SchedulerJob) -> bool:
        """Check if all dependencies of a job are completed."""
        if not job.dependencies:
            return True
        for dep_id in job.dependencies:
            dep = self._jobs.get(dep_id)
            if dep is None:
                return False
            if dep.status != JobStatus.COMPLETED:
                return False
        return True

    def _check_dependents(self, completed_job_id: str) -> None:
        """Enqueue any jobs whose dependencies are now satisfied."""
        for job in self._jobs.values():
            if job.status == JobStatus.PENDING:
                if completed_job_id in job.dependencies:
                    if self._dependencies_satisfied(job):
                        job.status = JobStatus.QUEUED
                        self._enqueue(job)

    def _enqueue(self, job: SchedulerJob) -> None:
        """Add a job to the priority heap."""
        # Sort key: (-priority.value, submission_time)
        # Higher priority = lower value = dequeued first
        sort_key = (-job.priority.value, job.submission_time)
        with self._queue_lock:
            heapq.heappush(self._queue, _PrioritizedJob(sort_key, job))

    # ── Context manager ───────────────────────────────────────────

    def __enter__(self) -> "CloudScheduler":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        m = self.metrics()
        return (
            f"CloudScheduler(policy={self.policy.value}, "
            f"backends={len(self._backends)}, "
            f"queued={m['queued']}, running={m['running']}, "
            f"completed={m['completed']})"
        )


# ── Module-level convenience ────────────────────────────────────────────

_default_scheduler: Optional[CloudScheduler] = None


def get_scheduler(**kwargs: Any) -> CloudScheduler:
    """Get or create the module-level default CloudScheduler."""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = CloudScheduler(**kwargs)
        _default_scheduler.register_backend("jax", provider="local")
        _default_scheduler.register_backend(
            "statevector", provider="local"
        )
    return _default_scheduler


def submit(
    circuit: sf.Circuit,
    backend: str = "jax",
    shots: int = 0,
    priority: JobPriority = JobPriority.NORMAL,
    **kwargs: Any,
) -> str:
    """Submit a circuit to the default scheduler."""
    return get_scheduler().submit(
        circuit, backend=backend, shots=shots, priority=priority, **kwargs
    )


__all__ = [
    "CloudScheduler",
    "SchedulerJob",
    "JobPriority",
    "JobStatus",
    "SchedulingPolicy",
    "BackendRegistration",
    "BatchResult",
    "get_scheduler",
    "submit",
]
