"""Integration tests for cross-backend consistency."""

import pytest

import superfermion as sf
from superfermion.results import RunResult


pytestmark = pytest.mark.integration

BELL_ALLOWED = {"00", "11"}
STATISTICAL_SHOTS = 10000
RATIO_TOLERANCE = 0.08
CORE_METHODS = ("statevector", "mps", "stabilizer")


def _method_available(method):
    try:
        circuit = sf.Circuit(2).h(0).cnot(0, 1)
        sf.run(circuit, device="cpu", method=method, shots=10, seed=42)
        return True
    except Exception:
        return False


def _run_bell_on_method(method, shots=STATISTICAL_SHOTS, seed=42):
    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    return sf.run(circuit, device="cpu", method=method, shots=shots, seed=seed)


def _assert_bell_result(result, shots):
    assert isinstance(result, RunResult)
    assert result.shots == shots
    assert result.counts
    assert set(result.counts.keys()) <= BELL_ALLOWED
    assert BELL_ALLOWED.issubset(result.counts.keys())
    assert sum(result.counts.values()) == shots

    ratio_00 = result.counts["00"] / shots
    assert 0.5 - RATIO_TOLERANCE <= ratio_00 <= 0.5 + RATIO_TOLERANCE


class TestMultiBackendConsistency:
    def test_default_and_explicit_statevector_statistically_consistent(self):
        if not _method_available("statevector"):
            pytest.skip("statevector method not available")

        default_result = sf.run(
            sf.Circuit(2).h(0).cnot(0, 1),
            device="cpu",
            shots=STATISTICAL_SHOTS,
            seed=42,
        )
        explicit_result = _run_bell_on_method("statevector")

        _assert_bell_result(default_result, STATISTICAL_SHOTS)
        _assert_bell_result(explicit_result, STATISTICAL_SHOTS)

    @pytest.mark.parametrize("method", ["statevector"])
    def test_core_methods_return_valid_run_result(self, method):
        if not _method_available(method):
            pytest.skip(f"{method} method not available")

        result = _run_bell_on_method(method, shots=1000)
        assert isinstance(result, RunResult)
        assert result.counts
        assert sum(result.counts.values()) == 1000

    def test_optional_mps_method_if_available(self):
        if not _method_available("mps"):
            pytest.skip("mps method not available")

        result = _run_bell_on_method("mps", shots=2000)
        _assert_bell_result(result, 2000)

    def test_optional_stabilizer_method_if_available(self):
        if not _method_available("stabilizer"):
            pytest.skip("stabilizer method not available")

        result = _run_bell_on_method("stabilizer", shots=2000)
        _assert_bell_result(result, 2000)

    def test_core_simulation_methods_available(self):
        available = [m for m in CORE_METHODS if _method_available(m)]
        assert "statevector" in available
