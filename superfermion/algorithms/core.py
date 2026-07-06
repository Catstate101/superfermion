"""
Core base classes for Variational Quantum Algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AlgorithmResult:
    """Standardized result object for VQE, QAOA, etc."""
    optimal_value: float
    optimal_params: Dict[str, Any]
    history: List[float] = field(default_factory=list)
    fidelity_history: Optional[List[float]] = None
    final_fidelity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        s = f"AlgorithmResult(optimal_value={self.optimal_value:.6f}"
        if self.final_fidelity is not None:
            s += f", final_fidelity={self.final_fidelity:.6f}"
        s += ")"
        return s
