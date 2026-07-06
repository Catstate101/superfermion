"""
Superfermion Cloud — QPU budget controls and cloud orchestration.

Usage:
    >>> import superfermion as sf
    >>> sf.cloud.set_budget(max_cost_usd=10.00)
    >>> sf.cloud.run(circuit, backend="ibm", shots=1000)  # aborts if would exceed budget
"""

from __future__ import annotations

from typing import Optional


class Cloud:
    """Cloud orchestration layer with cost controls and offline fallback."""

    def __init__(self):
        from superfermion.runtime.arbiter import arbiter, budget
        self._arbiter = arbiter
        self._budget = budget

    def set_budget(self, max_cost_usd: Optional[float] = None,
                   per_job_limit_usd: Optional[float] = None) -> None:
        """Set QPU spending limits to prevent surprise cloud bills.

        Args:
            max_cost_usd: Total budget cap. None = unlimited.
            per_job_limit_usd: Per-job spending cap. None = unlimited.
        """
        self._budget.set_budget(max_cost_usd, per_job_limit_usd)

    def estimate_cost(self, backend: str, shots: int = 1000) -> float:
        """Estimate USD cost for a job on the given backend."""
        return self._budget.estimate_cost(backend, shots)

    def check_budget(self, backend: str, shots: int = 1000) -> bool:
        """Check if a job fits within budget. Raises RuntimeError if not."""
        return self._budget.check_budget(backend, shots)

    @property
    def total_spent(self) -> float:
        """Total USD spent on QPU jobs."""
        return self._budget.total_spent

    @property
    def remaining_budget(self) -> Optional[float]:
        """Remaining budget in USD. None if no budget set."""
        return self._budget.remaining_budget

    def run(self, circuit, backend: str = "auto", shots: int = 1000, **kwargs):
        """Run a circuit with budget enforcement and offline fallback.

        First checks budget, then attempts the preferred backend. If
        unavailable, falls back through GPU sim → CPU sim.
        """
        # Check budget
        if backend not in ("auto", "jax", "statevector", "mps", "singularity", "stabilizer"):
            self.check_budget(backend, shots)

        # Offline fallback: guaranteed to find a working backend
        resolved = self._arbiter.offline_fallback(circuit, backend)

        from superfermion.runtime import runtime
        job = runtime.run(circuit, backend=resolved, shots=shots, **kwargs)

        # Record spend
        if resolved not in ("auto", "jax", "statevector", "mps", "singularity", "stabilizer"):
            self._budget.record_spend(resolved, shots)

        return job


# Singleton
cloud = Cloud()
