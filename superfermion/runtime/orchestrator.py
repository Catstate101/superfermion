"""
Job Orchestrator — Multi-provider quantum execution with race, fanout, and cheapest modes.

Extends the Runtime to enable:
  - **race**: Submit to all providers, return first result.
  - **fanout**: Submit to all providers, return all results for comparison.
  - **cheapest**: Estimate costs across providers and pick the cheapest.

Usage:
    >>> from superfermion.runtime.orchestrator import JobOrchestrator
    >>>
    >>> sf.runtime.connect("ibm", token="...")
    >>> sf.runtime.connect("ionq", api_key="...")
    >>>
    >>> orch = JobOrchestrator()
    >>>
    >>> # Race: first result wins
    >>> result = orch.race(circuit, backends=["statevector", "jax", "rust"])
    >>>
    >>> # Fanout: compare across backends
    >>> results = orch.fanout(circuit, backends=["statevector", "jax", "mps"])
    >>> comparison = orch.compare(results)
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import superfermion as sf
from superfermion.results import RunResult


@dataclass
class OrchestratorResult:
    """Aggregated result from a JobOrchestrator execution."""

    backend: str
    result: Optional[RunResult] = None
    statevector: Optional[np.ndarray] = None
    counts: Dict[str, int] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    success: bool = False

    @property
    def top_bitstring(self) -> str:
        if self.counts:
            return max(self.counts, key=self.counts.get)
        return ""

    @property
    def top_probability(self) -> float:
        if self.counts:
            total = sum(self.counts.values())
            return self.counts[self.top_bitstring] / max(total, 1)
        return 0.0


class JobOrchestrator:
    """Multi-provider quantum job orchestrator.

    Execute circuits across multiple backends simultaneously with
    race, fanout, or cheapest-cost strategies.
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._cost_registry: Dict[str, float] = {
            "statevector": 0.0,
            "jax": 0.0,
            "jax_mps": 0.0,
            "rust": 0.0,
            "mps": 0.0,
            "stabilizer": 0.0,
            "density_matrix": 0.0,
            "cuda": 0.0,
        }

    def set_cost(self, backend: str, cost: float):
        """Register the estimated cost for a backend (used by cheapest mode)."""
        self._cost_registry[backend] = cost

    # ── Race: first to return wins ────────────────────────────────────

    def race(
        self,
        circuit: sf.Circuit,
        backends: List[str],
        shots: int = 0,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> OrchestratorResult:
        """Submit circuit to all backends in parallel; return the first result.

        Args:
            circuit: Circuit to execute.
            backends: List of backend names (e.g. ``["statevector", "jax", "rust"]``).
            shots: Number of shots (0 = statevector mode).
            timeout: Maximum wait time in seconds.

        Returns:
            OrchestratorResult of the winner.
        """
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {}
            for backend in backends:
                fut = executor.submit(
                    _run_backend, circuit, backend, shots, kwargs
                )
                futures[fut] = backend

            # Wait for first completion
            done, not_done = concurrent.futures.wait(
                futures.keys(),
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            # Cancel remaining
            for fut in not_done:
                fut.cancel()

            if done:
                first_fut = list(done)[0]
                backend = futures[first_fut]
                result = first_fut.result()
                return OrchestratorResult(
                    backend=backend,
                    result=result.get("result"),
                    statevector=result.get("statevector"),
                    counts=result.get("counts", {}),
                    execution_time_ms=(time.perf_counter() - t0) * 1000,
                    success=result.get("success", True),
                    error=result.get("error"),
                )

        return OrchestratorResult(
            backend="",
            error="All backends timed out",
            execution_time_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Fanout: run everywhere, compare ────────────────────────────────

    def fanout(
        self,
        circuit: sf.Circuit,
        backends: List[str],
        shots: int = 0,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, OrchestratorResult]:
        """Execute circuit on all backends in parallel and return all results.

        Args:
            circuit: Circuit to execute.
            backends: List of backend names.
            shots: Number of shots.
            timeout: Maximum wait per backend.

        Returns:
            Dict mapping backend name → OrchestratorResult.
        """
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    _run_backend, circuit, backend, shots, kwargs
                ): backend
                for backend in backends
            }

            results: Dict[str, OrchestratorResult] = {}
            for future in concurrent.futures.as_completed(futures):
                backend = futures[future]
                try:
                    data = future.result()
                    results[backend] = OrchestratorResult(
                        backend=backend,
                        result=data.get("result"),
                        statevector=data.get("statevector"),
                        counts=data.get("counts", {}),
                        execution_time_ms=data.get("time_ms", 0),
                        success=data.get("success", True),
                        error=data.get("error"),
                    )
                except Exception as e:
                    results[backend] = OrchestratorResult(
                        backend=backend,
                        error=str(e),
                    )

        return results

    # ── Cheapest: pick by estimated cost ──────────────────────────────

    def cheapest(
        self,
        circuit: sf.Circuit,
        backends: List[str],
        shots: int = 0,
        **kwargs,
    ) -> OrchestratorResult:
        """Run on the cheapest backend (by registered cost estimates).

        Args:
            circuit: Circuit to execute.
            backends: Candidate backends.
            shots: Number of shots.

        Returns:
            OrchestratorResult from the cheapest backend.
        """
        # Sort by cost
        sorted_backends = sorted(
            backends,
            key=lambda b: self._cost_registry.get(b, float("inf")),
        )

        for backend in sorted_backends:
            try:
                data = _run_backend(circuit, backend, shots, kwargs)
                if data.get("success", True):
                    return OrchestratorResult(
                        backend=backend,
                        result=data.get("result"),
                        statevector=data.get("statevector"),
                        counts=data.get("counts", {}),
                        execution_time_ms=data.get("time_ms", 0),
                        success=True,
                        cost_estimate=self._cost_registry.get(backend),
                    )
            except Exception as e:
                continue  # Try next cheapest

        return OrchestratorResult(
            backend="",
            error="All backends failed in cheapest mode",
        )

    # ── Compare results ───────────────────────────────────────────────

    def compare(
        self,
        results: Dict[str, OrchestratorResult],
    ) -> Dict[str, Any]:
        """Compute fidelity/correlation matrix across fanout results.

        Args:
            results: Dict from fanout().

        Returns:
            Dict with ``"backend_order"``, ``"overlap_matrix"``,
            ``"top_bitstrings"``, ``"all_agree"``.
        """
        backends = sorted(results.keys())
        n = len(backends)

        # Fidelity (statevector overlap) matrix
        overlap = np.zeros((n, n))
        top_bits: Dict[str, str] = {}

        for i, b1 in enumerate(backends):
            r1 = results[b1]
            top_bits[b1] = r1.top_bitstring

            for j, b2 in enumerate(backends):
                r2 = results[b2]
                if i == j:
                    overlap[i, j] = 1.0
                elif r1.statevector is not None and r2.statevector is not None:
                    overlap[i, j] = float(
                        np.abs(np.dot(
                            np.conj(r1.statevector), r2.statevector
                        )) ** 2
                    )

        # Check agreement
        unique_tops = set(top_bits.values()) - {""}

        return {
            "backend_order": backends,
            "overlap_matrix": overlap.tolist(),
            "top_bitstrings": top_bits,
            "all_agree": len(unique_tops) <= 1,
            "count": n,
        }


# ── Internal ──────────────────────────────────────────────────────────

def _run_backend(
    circuit: sf.Circuit,
    backend: str,
    shots: int,
    kwargs: dict,
) -> Dict[str, Any]:
    """Execute on a single backend and return normalized result dict."""
    t0 = time.perf_counter()
    try:
        sim = sf.get_backend(backend)
        result = sim.run(circuit, shots=shots, **kwargs)
        t1 = time.perf_counter()

        sv = None
        if result.statevector is not None:
            sv = np.asarray(result.statevector).flatten()

        return {
            "result": result,
            "statevector": sv,
            "counts": result.counts or {},
            "success": True,
            "time_ms": (t1 - t0) * 1000,
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "result": None,
            "statevector": None,
            "counts": {},
            "success": False,
            "error": str(e),
            "time_ms": (t1 - t0) * 1000,
        }
