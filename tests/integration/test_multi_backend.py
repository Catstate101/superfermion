"""Integration tests for cross-backend consistency."""

import pytest

import superfermion as sf
from superfermion.backends.factory import get_backend, list_backends
from superfermion.devices.local import LocalDevice
from superfermion.results import RunResult


pytestmark = pytest.mark.integration

BELL_ALLOWED = {"00", "11"}
STATISTICAL_SHOTS = 10000
RATIO_TOLERANCE = 0.08


def _backend_available(name):
    try:
        get_backend(name)
        return True
    except Exception:
        return False


def _run_bell_on_backend(name, shots=STATISTICAL_SHOTS, seed=42):
    device = LocalDevice(name)
    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    return sf.run(circuit, device=device, shots=shots, seed=seed)


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
    def test_statevector_and_singularity_statistically_consistent(self):
        if not _backend_available("statevector"):
            pytest.skip("statevector backend not available")
        if not _backend_available("singularity"):
            pytest.skip("singularity backend not available")

        sv_result = _run_bell_on_backend("statevector")
        sg_result = _run_bell_on_backend("singularity")

        _assert_bell_result(sv_result, STATISTICAL_SHOTS)
        _assert_bell_result(sg_result, STATISTICAL_SHOTS)

    @pytest.mark.parametrize("backend_name", ["statevector", "singularity"])
    def test_core_backends_return_valid_run_result(self, backend_name):
        if not _backend_available(backend_name):
            pytest.skip(f"{backend_name} backend not available")

        result = _run_bell_on_backend(backend_name, shots=1000)
        assert isinstance(result, RunResult)
        assert result.counts
        assert sum(result.counts.values()) == 1000

    def test_optional_rust_backend_if_available(self):
        if not _backend_available("rust"):
            pytest.skip("rust backend not available")

        result = _run_bell_on_backend("rust", shots=2000)
        _assert_bell_result(result, 2000)

    def test_optional_mps_backend_if_available(self):
        if not _backend_available("mps"):
            pytest.skip("mps backend not available")

        result = _run_bell_on_backend("mps", shots=2000)
        _assert_bell_result(result, 2000)

    def test_optional_stabilizer_backend_if_available(self):
        if not _backend_available("stabilizer"):
            pytest.skip("stabilizer backend not available")

        result = _run_bell_on_backend("stabilizer", shots=2000)
        _assert_bell_result(result, 2000)

    def test_registered_backends_include_statevector(self):
        names = list_backends()
        assert "statevector" in names
        assert "singularity" in names
