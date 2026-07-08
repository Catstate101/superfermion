"""
Circuit construction workouts — 8 Benchpress-style tests.

Pattern: Template Method
Each workout is a callable that receives an SDKStrategy and returns a WorkoutResult.
The skeleton is fixed: build circuit -> record time -> validate -> report.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from superfermion.benchmarks.protocols import SDKStrategy, WorkoutResult
from superfermion.benchmarks.circuits import (
    QuantumVolumeFactory,
    DTCFactory,
    EfficientSU2Factory,
    RandomCliffordFactory,
    MultiControlXFactory,
)

SEED = 12345


def _timed(fn: Callable, rounds: int = 1) -> tuple[Any, float]:
    """Run fn() for `rounds` iterations, return (last_result, mean_time)."""
    total = 0.0
    result = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        result = fn()
        total += time.perf_counter() - t0
    return result, total / rounds


def bench_qv_build(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Build a 100-qubit, depth-100 Quantum Volume circuit from scratch."""
    factory = QuantumVolumeFactory()
    result, wall = _timed(
        lambda: strategy.build_circuit(factory, 100, depth=100, seed=SEED),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_qv_build",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"n_qubits": 100, "depth": 100,
                    "gate_count": strategy.gate_count(result)},
    )


def bench_dtc_build(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Build 100 DTC Floquet layers on 100 qubits."""
    factory = DTCFactory()
    result, wall = _timed(
        lambda: strategy.build_circuit(factory, 100, n_layers=100, seed=SEED),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_dtc_build",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"n_qubits": 100, "n_layers": 100,
                    "gate_count": strategy.gate_count(result)},
    )


def bench_clifford_build(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Build a 100-qubit random Clifford circuit."""
    factory = RandomCliffordFactory()
    result, wall = _timed(
        lambda: strategy.build_circuit(factory, 100, seed=SEED),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_clifford_build",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"n_qubits": 100,
                    "gate_count": strategy.gate_count(result)},
    )


def bench_multi_control_build(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Build a 16-controlled X gate circuit."""
    factory = MultiControlXFactory()
    result, wall = _timed(
        lambda: strategy.build_circuit(factory, 0, n_controls=16),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_multi_control_build",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"n_controls": 16,
                    "gate_count": strategy.gate_count(result)},
    )


def bench_su2_build(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Build 100-qubit EfficientSU2 ansatz with 4 reps (1000 parameters)."""
    factory = EfficientSU2Factory()
    result, wall = _timed(
        lambda: strategy.build_circuit(factory, 100, reps=4),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_su2_build",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"n_qubits": 100, "reps": 4,
                    "n_parameters": strategy.n_parameters(result),
                    "gate_count": strategy.gate_count(result)},
    )


def bench_su2_bind(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Bind 1000 parameters on a 100-qubit EfficientSU2 circuit."""
    factory = EfficientSU2Factory()
    circuit = strategy.build_circuit(factory, 100, reps=4)
    n_params = strategy.n_parameters(circuit)

    import numpy as np
    rng = np.random.default_rng(SEED)

    if strategy.name == "superfermion":
        param_names = circuit.parameters
        values = {name: float(rng.uniform(0, 2 * np.pi)) for name in param_names}
    elif strategy.name == "qiskit":
        values = {p.name: float(rng.uniform(0, 2 * np.pi))
                  for p in circuit.parameters}
    else:
        values = {f"p{i}": float(rng.uniform(0, 2 * np.pi))
                  for i in range(n_params)}

    result, wall = _timed(
        lambda: strategy.bind_parameters(circuit, values),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_su2_bind",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"n_parameters": n_params,
                    "bound_parameters": strategy.n_parameters(result)},
    )


def bench_qasm2_import(strategy: SDKStrategy, rounds: int = 3,
                       qasm_str: str = "", **kw) -> WorkoutResult:
    """Parse a QV100 QASM2 string."""
    if not qasm_str:
        qasm_str = _generate_qv_qasm(100, seed=SEED)

    result, wall = _timed(
        lambda: strategy.load_qasm(qasm_str),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_qasm2_import",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"qasm_length": len(qasm_str),
                    "gate_count": strategy.gate_count(result)},
    )


def bench_qasm2_bigint(strategy: SDKStrategy, rounds: int = 3, **kw) -> WorkoutResult:
    """Parse a QASM2 string containing large integer literals."""
    qasm_str = _generate_bigint_qasm()
    result, wall = _timed(
        lambda: strategy.load_qasm(qasm_str),
        rounds=rounds,
    )
    return WorkoutResult(
        test_name="bench_qasm2_bigint",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"qasm_length": len(qasm_str)},
    )


def _generate_qv_qasm(n: int, seed: int = 42) -> str:
    """Generate a QV-like QASM2 string for import benchmarks."""
    import numpy as np
    rng = np.random.default_rng(seed)
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{n}];',
    ]
    for _ in range(n):
        perm = rng.permutation(n).tolist()
        for j in range(0, n - 1, 2):
            q0, q1 = perm[j], perm[j + 1]
            a = float(rng.uniform(0, 6.28))
            b = float(rng.uniform(0, 6.28))
            lines.append(f'ry({a:.6f}) q[{q0}];')
            lines.append(f'rz({b:.6f}) q[{q0}];')
            lines.append(f'cx q[{q0}],q[{q1}];')
    return "\n".join(lines) + "\n"


def _generate_bigint_qasm() -> str:
    """Generate a QASM2 circuit with many gates for import stress testing."""
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        'qreg q[20];',
    ]
    for i in range(20):
        lines.append(f'h q[{i}];')
    for i in range(19):
        lines.append(f'cx q[{i}],q[{i + 1}];')
    for i in range(20):
        lines.append(f'rz(3.14159265358979323846264338327950288) q[{i}];')
    for i in range(19):
        lines.append(f'cx q[{i}],q[{i + 1}];')
    return "\n".join(lines) + "\n"


CONSTRUCTION_WORKOUTS: Dict[str, Callable] = {
    "bench_qv_build": bench_qv_build,
    "bench_dtc_build": bench_dtc_build,
    "bench_clifford_build": bench_clifford_build,
    "bench_multi_control_build": bench_multi_control_build,
    "bench_su2_build": bench_su2_build,
    "bench_su2_bind": bench_su2_bind,
    "bench_qasm2_import": bench_qasm2_import,
    "bench_qasm2_bigint": bench_qasm2_bigint,
}
