"""Base class and interface for Superfermion backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from superfermion.circuit import Circuit

from superfermion.results import RunResult


class Backend(ABC):
    """Abstract base class for all Superfermion execution backends.
    
    A backend is responsible for:
    1. Validating that a circuit can run on its hardware/simulator.
    2. Transpiling if necessary (usually handled by sf-compiler).
    3. Executing the circuit and returning a standardized RunResult.
    """

    def __init__(self, name: str, options: Optional[Dict[str, Any]] = None):
        self.name = name
        self.options = options or {}

    @abstractmethod
    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        """Run a quantum circuit on this backend.
        
        Args:
            circuit: The Superfermion Circuit to execute.
            shots: Total number of measurement repetitions.
            **kwargs: Backend-specific execution options.
            
        Returns:
            A RunResult containing counts, statevector, or probabilities.
        """
        pass

    def pre_bake(self, circuit: Circuit):
        """
        Ahead-of-Time (AOT) Hyper-Baking.
        Pre-compiles the circuit for this backend to eliminate cold-start latency.
        """
        # Default implementation is a no-op; specific backends will override.
        pass

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Maximum number of qubits supported by this backend."""
        pass

    @property
    @abstractmethod
    def supported_gates(self) -> List[str]:
        """List of gate names natively supported by this backend."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
