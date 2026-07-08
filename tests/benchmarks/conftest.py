"""Shared fixtures for benchmark tests."""

import pytest
from superfermion.benchmarks.strategies import SuperfermionStrategy
from superfermion.benchmarks.topologies import TopologyFactory


@pytest.fixture
def sf_strategy():
    return SuperfermionStrategy()


@pytest.fixture
def heavy_hex_backend():
    return TopologyFactory.create(
        "heavy_hex", n_qubits=127,
        basis_gates=["rz", "sx", "x", "ecr"],
    )


@pytest.fixture
def small_backend():
    """A linear 20-qubit backend for fast transpilation CI tests.

    For full-scale benchmarking, use the notebook with heavy_hex 127Q.
    """
    return TopologyFactory.create(
        "linear", n_qubits=20,
        basis_gates=["rz", "sx", "x", "cx"],
    )
