"""
Transpilation workouts — 9 Benchpress-style tests.

These measure end-to-end transpilation: basis translation + routing
against a real hardware topology.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import numpy as np

from superfermion.benchmarks.protocols import SDKStrategy, BenchmarkBackend, WorkoutResult
from superfermion.benchmarks.topologies import TopologyFactory
from superfermion.benchmarks.workouts.construction import _timed, SEED


def _default_backend() -> BenchmarkBackend:
    return TopologyFactory.create("heavy_hex", n_qubits=127,
                                  basis_gates=["rz", "sx", "x", "ecr"])


def _transpile_workout(name: str, strategy: SDKStrategy, circuit,
                       backend: BenchmarkBackend, rounds: int) -> WorkoutResult:
    """Common transpilation timing + metric extraction."""
    result, wall = _timed(
        lambda: strategy.compile(circuit, backend, level=2),
        rounds=rounds,
    )
    ops = strategy.count_ops(result)
    gate_count_2q = sum(v for k, v in ops.items()
                        if k.upper() in ("CX", "CZ", "ECR", "CNOT", "SWAP"))
    return WorkoutResult(
        test_name=name,
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={
            "input_n_qubits": strategy.n_qubits(circuit),
            "input_gate_count": strategy.gate_count(circuit),
            "output_gate_count": strategy.gate_count(result),
            "output_gate_count_2q": gate_count_2q,
            "output_depth": strategy.depth(result),
            "output_ops": ops,
        },
    )


def _build_qft(strategy: SDKStrategy, n: int):
    """Build an n-qubit QFT circuit."""
    if strategy.name == "superfermion":
        import superfermion as sf
        qc = sf.Circuit(n)
        for i in range(n):
            qc = qc.h(i)
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                qc = qc.cp(float(angle), j, i)
        for i in range(n // 2):
            qc = qc.swap(i, n - 1 - i)
        return qc
    elif strategy.name == "qiskit":
        from qiskit.circuit.library import QFT
        return QFT(n).decompose()
    return None


def _build_bv(strategy: SDKStrategy, n: int, secret: str = ""):
    """Build an n-qubit Bernstein-Vazirani circuit."""
    if not secret:
        secret = "1" * (n - 1)
    if strategy.name == "superfermion":
        import superfermion as sf
        qc = sf.Circuit(n)
        qc = qc.x(n - 1).h(n - 1)
        for i in range(n - 1):
            qc = qc.h(i)
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                qc = qc.cx(i, n - 1)
        for i in range(n - 1):
            qc = qc.h(i)
        return qc
    elif strategy.name == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        qc.x(n - 1); qc.h(n - 1)
        for i in range(n - 1):
            qc.h(i)
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                qc.cx(i, n - 1)
        for i in range(n - 1):
            qc.h(i)
        return qc
    return None


def bench_qft_transpile(strategy: SDKStrategy, rounds: int = 3,
                        backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit QFT."""
    backend = backend or _default_backend()
    circuit = _build_qft(strategy, 100)
    return _transpile_workout("bench_qft_transpile", strategy, circuit, backend, rounds)


def bench_qv_transpile(strategy: SDKStrategy, rounds: int = 3,
                       backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit Quantum Volume."""
    backend = backend or _default_backend()
    from superfermion.benchmarks.circuits import QuantumVolumeFactory
    circuit = strategy.build_circuit(QuantumVolumeFactory(), 100, depth=10, seed=SEED)
    return _transpile_workout("bench_qv_transpile", strategy, circuit, backend, rounds)


def bench_su2_transpile(strategy: SDKStrategy, rounds: int = 3,
                        backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit EfficientSU2 (3 reps)."""
    backend = backend or _default_backend()
    from superfermion.benchmarks.circuits import EfficientSU2Factory
    circuit = strategy.build_circuit(EfficientSU2Factory(), 100, reps=3)
    if strategy.n_parameters(circuit) > 0:
        rng = np.random.default_rng(SEED)
        if strategy.name == "superfermion":
            vals = {name: float(rng.uniform(0, 2 * np.pi)) for name in circuit.parameters}
        elif strategy.name == "qiskit":
            vals = {p.name: float(rng.uniform(0, 2 * np.pi)) for p in circuit.parameters}
        else:
            vals = {}
        circuit = strategy.bind_parameters(circuit, vals)
    return _transpile_workout("bench_su2_transpile", strategy, circuit, backend, rounds)


def bench_bv_transpile(strategy: SDKStrategy, rounds: int = 3,
                       backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit Bernstein-Vazirani (all-ones secret)."""
    backend = backend or _default_backend()
    circuit = _build_bv(strategy, 100)
    return _transpile_workout("bench_bv_transpile", strategy, circuit, backend, rounds)


def bench_heisenberg_transpile(strategy: SDKStrategy, rounds: int = 3,
                               backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit square Heisenberg model circuit."""
    backend = backend or _default_backend()
    n = 100
    if strategy.name == "superfermion":
        import superfermion as sf
        qc = sf.Circuit(n)
        for q in range(n):
            qc = qc.h(q)
        for q in range(n - 1):
            qc = qc.rzz(0.5, q, q + 1).rxx(0.5, q, q + 1).ryy(0.5, q, q + 1)
        return _transpile_workout("bench_heisenberg_transpile", strategy, qc, backend, rounds)
    elif strategy.name == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for q in range(n - 1):
            qc.rzz(0.5, q, q + 1); qc.rxx(0.5, q, q + 1); qc.ryy(0.5, q, q + 1)
        return _transpile_workout("bench_heisenberg_transpile", strategy, qc, backend, rounds)
    return WorkoutResult("bench_heisenberg_transpile", strategy.name, strategy.version, 0.0)


def bench_qaoa_transpile(strategy: SDKStrategy, rounds: int = 3,
                         backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit QAOA (3 reps, linear graph)."""
    backend = backend or _default_backend()
    n, reps = 100, 3
    if strategy.name == "superfermion":
        import superfermion as sf
        qc = sf.Circuit(n)
        for q in range(n):
            qc = qc.h(q)
        for _ in range(reps):
            for q in range(n - 1):
                qc = qc.rzz(0.5, q, q + 1)
            for q in range(n):
                qc = qc.rx(0.5, q)
        return _transpile_workout("bench_qaoa_transpile", strategy, qc, backend, rounds)
    elif strategy.name == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for _ in range(reps):
            for q in range(n - 1):
                qc.rzz(0.5, q, q + 1)
            for q in range(n):
                qc.rx(0.5, q)
        return _transpile_workout("bench_qaoa_transpile", strategy, qc, backend, rounds)
    return WorkoutResult("bench_qaoa_transpile", strategy.name, strategy.version, 0.0)


def bench_simplification_transpile(strategy: SDKStrategy, rounds: int = 3,
                                   backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile a BV-like circuit that should simplify to X+Z gates."""
    backend = backend or _default_backend()
    n = 100
    if strategy.name == "superfermion":
        import superfermion as sf
        qc = sf.Circuit(n)
        for q in range(n):
            qc = qc.h(q).x(q).h(q)
        return _transpile_workout("bench_simplification_transpile", strategy, qc, backend, rounds)
    elif strategy.name == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q); qc.x(q); qc.h(q)
        return _transpile_workout("bench_simplification_transpile", strategy, qc, backend, rounds)
    return WorkoutResult("bench_simplification_transpile", strategy.name, strategy.version, 0.0)


def bench_clifford_transpile(strategy: SDKStrategy, rounds: int = 3,
                             backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 100-qubit random Clifford circuit."""
    backend = backend or _default_backend()
    from superfermion.benchmarks.circuits import RandomCliffordFactory
    circuit = strategy.build_circuit(RandomCliffordFactory(), 100, seed=SEED)
    return _transpile_workout("bench_clifford_transpile", strategy, circuit, backend, rounds)


def bench_circsu2_89_transpile(strategy: SDKStrategy, rounds: int = 3,
                               backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Transpile 89-qubit EfficientSU2 (non-power-of-2)."""
    backend = backend or _default_backend()
    from superfermion.benchmarks.circuits import EfficientSU2Factory
    circuit = strategy.build_circuit(EfficientSU2Factory(), 89, reps=3)
    if strategy.n_parameters(circuit) > 0:
        rng = np.random.default_rng(SEED)
        if strategy.name == "superfermion":
            vals = {name: float(rng.uniform(0, 2 * np.pi)) for name in circuit.parameters}
        elif strategy.name == "qiskit":
            vals = {p.name: float(rng.uniform(0, 2 * np.pi)) for p in circuit.parameters}
        else:
            vals = {}
        circuit = strategy.bind_parameters(circuit, vals)
    return _transpile_workout("bench_circsu2_89_transpile", strategy, circuit, backend, rounds)


TRANSPILATION_WORKOUTS: Dict[str, Callable] = {
    "bench_qft_transpile": bench_qft_transpile,
    "bench_qv_transpile": bench_qv_transpile,
    "bench_su2_transpile": bench_su2_transpile,
    "bench_bv_transpile": bench_bv_transpile,
    "bench_heisenberg_transpile": bench_heisenberg_transpile,
    "bench_qaoa_transpile": bench_qaoa_transpile,
    "bench_simplification_transpile": bench_simplification_transpile,
    "bench_clifford_transpile": bench_clifford_transpile,
    "bench_circsu2_89_transpile": bench_circsu2_89_transpile,
}
