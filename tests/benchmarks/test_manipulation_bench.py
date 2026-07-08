"""pytest-benchmark tests for manipulation workouts.

Each test uses the ``benchmark`` fixture to time the core operation.

Run:
    pytest tests/benchmarks/test_manipulation_bench.py --benchmark-only
"""

import pytest

from superfermion.benchmarks.circuits import (
    DTCFactory,
    QuantumVolumeFactory,
    RandomCliffordFactory,
    MultiControlXFactory,
)
from superfermion.benchmarks.workouts.construction import SEED
from superfermion.compiler.passes import BasisTranslationPass


def test_bench_pauli_twirl(benchmark, sf_strategy):
    """Pauli-twirl 2Q gates on a 100-qubit DTC circuit."""
    from superfermion.compiler.advanced import PauliTwirlingPass
    circuit = sf_strategy.build_circuit(DTCFactory(), 100, n_layers=10, seed=SEED)
    twirl = PauliTwirlingPass(seed=SEED)

    result = benchmark(twirl.run, circuit)
    assert sf_strategy.gate_count(result) > 0


def test_bench_multi_control_decompose(benchmark, sf_strategy):
    """Decompose a 16-control X gate to {rx, ry, rz, cz} basis."""
    circuit = sf_strategy.build_circuit(MultiControlXFactory(), 0, n_controls=16)
    btp = BasisTranslationPass(["rx", "ry", "rz", "cz"])

    result = benchmark(btp.run, circuit)
    ops = sf_strategy.count_ops(result)
    assert all(k.upper() in ("RX", "RY", "RZ", "CZ", "BARRIER", "MEASURE")
               for k in ops), f"Non-basis gates remain: {ops}"


def test_bench_basis_change(benchmark, sf_strategy):
    """Translate QV100 from {ry, rz, cx} to {sx, x, rz, cz}."""
    circuit = sf_strategy.build_circuit(QuantumVolumeFactory(), 100, depth=10, seed=SEED)
    btp = BasisTranslationPass(["sx", "x", "rz", "cz"])

    result = benchmark(btp.run, circuit)
    ops = sf_strategy.count_ops(result)
    assert all(k.upper() in ("SX", "X", "RZ", "CZ", "BARRIER", "MEASURE")
               for k in ops), f"Non-basis gates remain: {ops}"


def test_bench_clifford_decompose(benchmark, sf_strategy):
    """Decompose a 20-qubit random Clifford to {rz, sx, x, cz}."""
    circuit = sf_strategy.build_circuit(RandomCliffordFactory(), 20, depth=40, seed=SEED)
    btp = BasisTranslationPass(["rz", "sx", "x", "cz"])

    result = benchmark(btp.run, circuit)
    ops = sf_strategy.count_ops(result)
    assert all(k.upper() in ("RZ", "SX", "X", "CZ", "BARRIER", "MEASURE")
               for k in ops), f"Non-basis gates remain: {ops}"
