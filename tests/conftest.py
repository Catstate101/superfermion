"""Shared fixtures for the Superfermion test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.parameters import param
from superfermion.experiment.protocols import TrackerProtocol


# ── Circuit fixtures ──────────────────────────────────────────────────

@pytest.fixture
def bell_circuit() -> Circuit:
    """2-qubit Bell state: (|00⟩ + |11⟩) / √2"""
    return Circuit(2).h(0).cnot(0, 1)


@pytest.fixture
def ghz_circuit():
    """Factory fixture for n-qubit GHZ states."""
    def _make(n: int = 3) -> Circuit:
        c = Circuit(n).h(0)
        for i in range(n - 1):
            c.cnot(i, i + 1)
        return c
    return _make


@pytest.fixture
def parametric_circuit() -> Circuit:
    """RX/RY circuit with symbolic parameters."""
    theta = param("theta")
    phi = param("phi")
    return Circuit(2).rx(theta, 0).ry(phi, 1).cnot(0, 1)


# ── Tracker fixtures ─────────────────────────────────────────────────

class MockTracker:
    """TrackerProtocol-satisfying mock that records all calls."""

    def __init__(self):
        self.starts: List[Dict[str, Any]] = []
        self.completions: List[Any] = []
        self.errors: List[Exception] = []

    def on_run_start(self, circuit, device, shots, metadata=None):
        self.starts.append({
            "circuit": circuit, "device": device,
            "shots": shots, "metadata": metadata,
        })

    def on_run_complete(self, result, metadata=None):
        self.completions.append({"result": result, "metadata": metadata})

    def on_run_error(self, error, metadata=None):
        self.errors.append(error)


@pytest.fixture
def mock_tracker() -> MockTracker:
    return MockTracker()


@pytest.fixture
def tmp_tracker_dir(tmp_path: Path) -> Path:
    return tmp_path / "tracker_runs"


# ── Pytest configuration ─────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "backend: backend correctness tests")
    config.addinivalue_line("markers", "domain: domain-specific tests")
    config.addinivalue_line("markers", "e2e: end-to-end tests")
    config.addinivalue_line("markers", "slow: slow tests (skipped by default)")
    config.addinivalue_line("markers", "timeout: timeout for test")
    config.addinivalue_line("markers", "timeout(timeout): per-test timeout in seconds")
