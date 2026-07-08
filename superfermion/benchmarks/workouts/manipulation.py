"""
Circuit manipulation workouts — 4 Benchpress-style tests.

These test the SDK's ability to transform existing circuits:
twirling, decomposition, and basis translation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from superfermion.benchmarks.protocols import SDKStrategy, BenchmarkBackend, WorkoutResult
from superfermion.benchmarks.workouts.construction import _timed, SEED


def bench_pauli_twirl(strategy: SDKStrategy, rounds: int = 3,
                      backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Pauli-twirl 2Q gates on a 100-qubit DTC circuit."""
    from superfermion.benchmarks.circuits import DTCFactory
    circuit = strategy.build_circuit(DTCFactory(), 100, n_layers=10, seed=SEED)

    if strategy.name == "superfermion":
        from superfermion.compiler.advanced import PauliTwirlingPass
        twirl_pass = PauliTwirlingPass(seed=SEED)
        result, wall = _timed(lambda: twirl_pass.run(circuit), rounds=rounds)
    elif strategy.name == "qiskit":
        from qiskit.transpiler.passes import PauliTwirl
        from qiskit.transpiler import PassManager
        pm = PassManager([PauliTwirl(seed=SEED)])
        result, wall = _timed(lambda: pm.run(circuit), rounds=rounds)
    else:
        result, wall = circuit, 0.0

    return WorkoutResult(
        test_name="bench_pauli_twirl",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"input_gate_count": strategy.gate_count(circuit),
                    "output_gate_count": strategy.gate_count(result)},
    )


def bench_multi_control_decompose(strategy: SDKStrategy, rounds: int = 3,
                                  backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Decompose a 16-control X gate into {rx, ry, rz, cz} basis."""
    from superfermion.benchmarks.circuits import MultiControlXFactory
    circuit = strategy.build_circuit(MultiControlXFactory(), 0, n_controls=16)
    target_basis = ["rx", "ry", "rz", "cz"]

    if strategy.name == "superfermion":
        from superfermion.compiler.passes import BasisTranslationPass
        btp = BasisTranslationPass(target_basis)
        result, wall = _timed(lambda: btp.run(circuit), rounds=rounds)
    elif strategy.name == "qiskit":
        from qiskit.compiler import transpile
        result, wall = _timed(
            lambda: transpile(circuit, basis_gates=target_basis, optimization_level=0),
            rounds=rounds,
        )
    else:
        result, wall = circuit, 0.0

    return WorkoutResult(
        test_name="bench_multi_control_decompose",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"target_basis": target_basis,
                    "output_ops": strategy.count_ops(result)},
    )


def bench_basis_change(strategy: SDKStrategy, rounds: int = 3,
                       backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Translate QV100 from {ry, rz, cx} to {sx, x, rz, cz}."""
    from superfermion.benchmarks.circuits import QuantumVolumeFactory
    circuit = strategy.build_circuit(QuantumVolumeFactory(), 100, depth=10, seed=SEED)
    target_basis = ["sx", "x", "rz", "cz"]

    if strategy.name == "superfermion":
        from superfermion.compiler.passes import BasisTranslationPass
        btp = BasisTranslationPass(target_basis)
        result, wall = _timed(lambda: btp.run(circuit), rounds=rounds)
    elif strategy.name == "qiskit":
        from qiskit.compiler import transpile
        result, wall = _timed(
            lambda: transpile(circuit, basis_gates=target_basis, optimization_level=0),
            rounds=rounds,
        )
    else:
        result, wall = circuit, 0.0

    ops = strategy.count_ops(result)
    gate_count_2q = sum(v for k, v in ops.items() if k.upper() in ("CX", "CZ", "ECR", "CNOT"))
    return WorkoutResult(
        test_name="bench_basis_change",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"target_basis": target_basis,
                    "gate_count_2q": gate_count_2q,
                    "output_ops": ops},
    )


def bench_clifford_decompose(strategy: SDKStrategy, rounds: int = 3,
                             backend: BenchmarkBackend | None = None, **kw) -> WorkoutResult:
    """Decompose a 20-qubit random Clifford into {rz, sx, x, cz}."""
    from superfermion.benchmarks.circuits import RandomCliffordFactory
    circuit = strategy.build_circuit(RandomCliffordFactory(), 20, depth=40, seed=SEED)
    target_basis = ["rz", "sx", "x", "cz"]

    if strategy.name == "superfermion":
        from superfermion.compiler.passes import BasisTranslationPass
        btp = BasisTranslationPass(target_basis)
        result, wall = _timed(lambda: btp.run(circuit), rounds=rounds)
    elif strategy.name == "qiskit":
        from qiskit.compiler import transpile
        result, wall = _timed(
            lambda: transpile(circuit, basis_gates=target_basis, optimization_level=0),
            rounds=rounds,
        )
    else:
        result, wall = circuit, 0.0

    ops = strategy.count_ops(result)
    gate_count_2q = sum(v for k, v in ops.items() if k.upper() in ("CX", "CZ", "ECR", "CNOT"))
    return WorkoutResult(
        test_name="bench_clifford_decompose",
        sdk_name=strategy.name,
        sdk_version=strategy.version,
        wall_time_s=wall,
        rounds=rounds,
        extra_info={"target_basis": target_basis,
                    "gate_count_2q": gate_count_2q},
    )


MANIPULATION_WORKOUTS: Dict[str, Callable] = {
    "bench_pauli_twirl": bench_pauli_twirl,
    "bench_multi_control_decompose": bench_multi_control_decompose,
    "bench_basis_change": bench_basis_change,
    "bench_clifford_decompose": bench_clifford_decompose,
}
