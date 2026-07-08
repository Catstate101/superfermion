"""pytest-benchmark tests for transpilation workouts.

Uses a linear 20-qubit backend and 20Q circuits for fast CI.
For full-scale benchmarking (100Q on heavy_hex 127Q), use the
``benchmark_benchpress.ipynb`` notebook or the ``BenchmarkRunner`` API.

Run:
    pytest tests/benchmarks/test_transpilation_bench.py --benchmark-only
    pytest tests/benchmarks/test_transpilation_bench.py --benchmark-json=transpile.json
"""

import pytest
import numpy as np

from superfermion.benchmarks.circuits import (
    QuantumVolumeFactory,
    EfficientSU2Factory,
    RandomCliffordFactory,
)
from superfermion.benchmarks.workouts.construction import SEED
from superfermion.benchmarks.workouts.transpilation import (
    _build_qft,
    _build_bv,
)

N = 20
ROUNDS = 3
WARMUP = 1


def test_bench_qft_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit QFT."""
    circuit = _build_qft(sf_strategy, N)
    result = benchmark.pedantic(
        sf_strategy.compile, args=(circuit, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


def test_bench_qv_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit Quantum Volume (depth 10)."""
    circuit = sf_strategy.build_circuit(QuantumVolumeFactory(), N, depth=10, seed=SEED)
    result = benchmark.pedantic(
        sf_strategy.compile, args=(circuit, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


def test_bench_su2_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit EfficientSU2 (3 reps, bound)."""
    circuit = sf_strategy.build_circuit(EfficientSU2Factory(), N, reps=3)
    rng = np.random.default_rng(SEED)
    values = {name: float(rng.uniform(0, 2 * np.pi)) for name in circuit.parameters}
    circuit = sf_strategy.bind_parameters(circuit, values)

    result = benchmark.pedantic(
        sf_strategy.compile, args=(circuit, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


def test_bench_bv_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit Bernstein-Vazirani."""
    circuit = _build_bv(sf_strategy, N)
    result = benchmark.pedantic(
        sf_strategy.compile, args=(circuit, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


def test_bench_simplification_transpile(benchmark, sf_strategy, small_backend):
    """Transpile H-X-H pattern (should simplify)."""
    import superfermion as sf
    qc = sf.Circuit(N)
    for q in range(N):
        qc = qc.h(q).x(q).h(q)

    result = benchmark.pedantic(
        sf_strategy.compile, args=(qc, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


def test_bench_clifford_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit random Clifford."""
    circuit = sf_strategy.build_circuit(RandomCliffordFactory(), N, seed=SEED)
    result = benchmark.pedantic(
        sf_strategy.compile, args=(circuit, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


@pytest.mark.slow
def test_bench_heisenberg_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit Heisenberg model circuit."""
    import superfermion as sf
    qc = sf.Circuit(N)
    for q in range(N):
        qc = qc.h(q)
    for q in range(N - 1):
        qc = qc.rzz(0.5, q, q + 1).rxx(0.5, q, q + 1).ryy(0.5, q, q + 1)

    result = benchmark.pedantic(
        sf_strategy.compile, args=(qc, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


@pytest.mark.slow
def test_bench_qaoa_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 20-qubit QAOA (3 reps)."""
    import superfermion as sf
    qc = sf.Circuit(N)
    for q in range(N):
        qc = qc.h(q)
    for _ in range(3):
        for q in range(N - 1):
            qc = qc.rzz(0.5, q, q + 1)
        for q in range(N):
            qc = qc.rx(0.5, q)

    result = benchmark.pedantic(
        sf_strategy.compile, args=(qc, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0


@pytest.mark.slow
def test_bench_circsu2_19_transpile(benchmark, sf_strategy, small_backend):
    """Transpile 19-qubit EfficientSU2 (non-power-of-2)."""
    circuit = sf_strategy.build_circuit(EfficientSU2Factory(), 19, reps=3)
    rng = np.random.default_rng(SEED)
    values = {name: float(rng.uniform(0, 2 * np.pi)) for name in circuit.parameters}
    circuit = sf_strategy.bind_parameters(circuit, values)

    result = benchmark.pedantic(
        sf_strategy.compile, args=(circuit, small_backend, 2),
        rounds=ROUNDS, warmup_rounds=WARMUP, iterations=1,
    )
    assert sf_strategy.gate_count(result) > 0
