"""
Benchmark Protocols — contracts for SDK-agnostic benchmarking.

Pattern: Strategy + Abstract Factory + Protocol
Problem: Benchmark operations differ per SDK (SF, Qiskit, Cirq) but must share
         the same interface for fair comparison.
Solution: Each SDK implements SDKStrategy; circuit families implement CircuitFactory;
          hardware targets satisfy BenchmarkBackend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SDKStrategy(Protocol):
    """Strategy for a specific quantum SDK.

    Each SDK (Superfermion, Qiskit, Cirq, ...) implements this protocol.
    The BenchmarkRunner calls these methods without knowing which SDK it uses.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def build_circuit(self, factory: CircuitFactory, n_qubits: int, **kwargs: Any) -> Any: ...

    def load_qasm(self, qasm_str: str) -> Any: ...

    def compile(self, circuit: Any, backend: BenchmarkBackend, level: int = 1) -> Any: ...

    def count_ops(self, circuit: Any) -> Dict[str, int]: ...

    def gate_count(self, circuit: Any) -> int: ...

    def depth(self, circuit: Any) -> int: ...

    def n_qubits(self, circuit: Any) -> int: ...

    def bind_parameters(self, circuit: Any, values: Dict[str, float]) -> Any: ...

    def n_parameters(self, circuit: Any) -> int: ...


@runtime_checkable
class CircuitFactory(Protocol):
    """Factory for building a specific circuit family (QV, DTC, SU2, etc.).

    Each factory knows the algorithm but delegates gate calls to the SDK strategy,
    keeping the circuit specification SDK-agnostic.
    """

    @property
    def name(self) -> str: ...

    def create(self, strategy: SDKStrategy, n_qubits: int, **kwargs: Any) -> Any: ...


@runtime_checkable
class BenchmarkBackend(Protocol):
    """Target hardware abstraction for transpilation benchmarks.

    Any object with n_qubits, basis_gates, and coupling_map satisfies this.
    No provider coupling — "heavy_hex" is a topology shape, not a vendor name.
    """

    @property
    def n_qubits(self) -> int: ...

    @property
    def basis_gates(self) -> List[str]: ...

    @property
    def coupling_map(self) -> List[Tuple[int, int]]: ...


@dataclass
class WorkoutResult:
    """Result of a single benchmark test for one SDK."""

    test_name: str
    sdk_name: str
    sdk_version: str
    wall_time_s: float
    rounds: int = 1
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Collection of workout results across SDKs."""

    results: List[WorkoutResult] = field(default_factory=list)

    def add(self, result: WorkoutResult) -> None:
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to pytest-benchmark-compatible JSON schema."""
        benchmarks = []
        for r in self.results:
            benchmarks.append({
                "name": r.test_name,
                "group": r.test_name.rsplit("_", 1)[0],
                "stats": {
                    "min": r.wall_time_s,
                    "mean": r.wall_time_s,
                    "max": r.wall_time_s,
                    "rounds": r.rounds,
                },
                "extra_info": {
                    "sdk_name": r.sdk_name,
                    "sdk_version": r.sdk_version,
                    **r.extra_info,
                },
            })
        return {"benchmarks": benchmarks}
