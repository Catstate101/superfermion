"""pytest-benchmark tests for construction workouts.

Each test uses the ``benchmark`` fixture to time the core operation
with proper warmup, multiple rounds, and statistical analysis.

Run:
    pytest tests/benchmarks/test_construction_bench.py --benchmark-only
    pytest tests/benchmarks/ --benchmark-json=results.json
"""

import pytest
import numpy as np

from superfermion.benchmarks.circuits import (
    QuantumVolumeFactory,
    DTCFactory,
    EfficientSU2Factory,
    RandomCliffordFactory,
    MultiControlXFactory,
)
from superfermion.benchmarks.workouts.construction import (
    _generate_qv_qasm,
    _generate_bigint_qasm,
    SEED,
)


def test_bench_qv_build(benchmark, sf_strategy):
    """Build a 100-qubit, depth-100 Quantum Volume circuit."""
    factory = QuantumVolumeFactory()
    result = benchmark(sf_strategy.build_circuit, factory, 100, depth=100, seed=SEED)
    assert sf_strategy.gate_count(result) > 0


def test_bench_dtc_build(benchmark, sf_strategy):
    """Build 100 DTC Floquet layers on 100 qubits."""
    factory = DTCFactory()
    result = benchmark(sf_strategy.build_circuit, factory, 100, n_layers=100, seed=SEED)
    assert sf_strategy.gate_count(result) > 0


def test_bench_clifford_build(benchmark, sf_strategy):
    """Build a 100-qubit random Clifford circuit."""
    factory = RandomCliffordFactory()
    result = benchmark(sf_strategy.build_circuit, factory, 100, seed=SEED)
    assert sf_strategy.gate_count(result) > 0


def test_bench_multi_control_build(benchmark, sf_strategy):
    """Build a 16-controlled X gate circuit."""
    factory = MultiControlXFactory()
    result = benchmark(sf_strategy.build_circuit, factory, 0, n_controls=16)
    assert sf_strategy.gate_count(result) > 0


def test_bench_su2_build(benchmark, sf_strategy):
    """Build 100-qubit EfficientSU2 ansatz (4 reps, ~1000 parameters)."""
    factory = EfficientSU2Factory()
    result = benchmark(sf_strategy.build_circuit, factory, 100, reps=4)
    assert sf_strategy.n_parameters(result) > 0


def test_bench_su2_bind(benchmark, sf_strategy):
    """Bind ~1000 parameters on a 100-qubit EfficientSU2 circuit."""
    factory = EfficientSU2Factory()
    circuit = sf_strategy.build_circuit(factory, 100, reps=4)
    rng = np.random.default_rng(SEED)
    values = {name: float(rng.uniform(0, 2 * np.pi))
              for name in circuit.parameters}

    result = benchmark(sf_strategy.bind_parameters, circuit, values)
    assert sf_strategy.n_parameters(result) == 0


def test_bench_qasm2_import(benchmark, sf_strategy):
    """Parse a QV100 QASM2 string."""
    qasm_str = _generate_qv_qasm(100, seed=SEED)
    result = benchmark(sf_strategy.load_qasm, qasm_str)
    assert sf_strategy.gate_count(result) > 0


def test_bench_qasm2_bigint(benchmark, sf_strategy):
    """Parse a QASM2 string with large float literals."""
    qasm_str = _generate_bigint_qasm()
    result = benchmark(sf_strategy.load_qasm, qasm_str)
    assert sf_strategy.gate_count(result) > 0
