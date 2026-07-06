"""
Resource Arbiter — Facade delegating to Router, SecurityValidator, and Fallback.

Was a God Object with routing + security + fallback in one class.
Now delegates to single-responsibility components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import superfermion as sf
from superfermion.runtime.router import BackendRouter
from superfermion.runtime.security_validator import SecurityValidator
from superfermion.runtime.fallback import FallbackOrchestrator


class ResourceArbiter:
    """Facade for runtime routing, security, and fallback decisions."""

    def __init__(self):
        self._router = BackendRouter()
        self._security = SecurityValidator()
        self._fallback = FallbackOrchestrator()

    def route(self, circuit: sf.Circuit = None, n_qubits: int = None) -> str:
        """Alias for select_best_backend."""
        if n_qubits is not None and circuit is None:
            circuit = sf.Circuit(n_qubits)
        if circuit is None:
            raise ValueError("Must provide either 'circuit' or 'n_qubits' to route().")
        return self.select_best_backend(circuit)

    def select_best_backend(self, circuit: sf.Circuit, requested_target: Optional[str] = None) -> str:
        """Delegates to BackendRouter."""
        return self._router.select(circuit, requested_target)

    def validate_security(self, circuit: sf.Circuit, user_quota: int = 100) -> bool:
        """Delegates to SecurityValidator."""
        return self._security.validate(circuit)

    def offline_fallback(self, circuit: sf.Circuit, preferred_backend: str = "auto") -> str:
        """Delegates to FallbackOrchestrator."""
        return self._fallback.resolve(circuit, preferred_backend)


class BudgetManager:
    """QPU cost controls -- prevents surprise cloud bills."""

    _COST_PER_SHOT = {
        "ibm": 0.000016,
        "ionq": 0.00003,
        "aws": 0.00005,
    }

    def __init__(self):
        self._total_spent: float = 0.0
        self._max_budget = None
        self._per_job_limit = None

    def set_budget(self, max_cost_usd=None, per_job_limit_usd=None):
        self._max_budget = max_cost_usd
        self._per_job_limit = per_job_limit_usd

    def estimate_cost(self, backend: str, shots: int = 1000) -> float:
        return self._COST_PER_SHOT.get(backend, 0.0) * shots

    def check_budget(self, backend: str, shots: int = 1000) -> bool:
        cost = self.estimate_cost(backend, shots)
        if self._per_job_limit is not None and cost > self._per_job_limit:
            raise RuntimeError(f"Job cost ${cost:.4f} exceeds per-job limit ${self._per_job_limit:.2f}")
        if self._max_budget is not None and self._total_spent + cost > self._max_budget:
            raise RuntimeError(f"Budget exceeded: ${self._total_spent:.4f} spent + ${cost:.4f} > ${self._max_budget:.2f}")
        return True

    def record_spend(self, backend: str, shots: int = 1000) -> float:
        cost = self.estimate_cost(backend, shots)
        self._total_spent += cost
        return cost

    @property
    def total_spent(self) -> float:
        return self._total_spent

    @property
    def remaining_budget(self):
        if self._max_budget is None:
            return None
        return max(0.0, self._max_budget - self._total_spent)


arbiter = ResourceArbiter()
budget = BudgetManager()
