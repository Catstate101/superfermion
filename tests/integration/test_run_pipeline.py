"""Integration tests for the full sf.run() execution pipeline."""

import re

import pytest

import superfermion as sf
from superfermion.devices.local import LocalDevice


pytestmark = pytest.mark.integration


def _assert_valid_bitstrings(counts, n_qubits):
    pattern = re.compile(rf"^[01]{{{n_qubits}}}$")
    for bitstring in counts:
        assert pattern.match(bitstring), f"Invalid bitstring key: {bitstring!r}"


def _assert_bell_counts(counts, shots, tolerance=0.25):
    """Bell state should produce only |00⟩ and |11⟩ with roughly equal counts."""
    assert set(counts.keys()) <= {"00", "11"}
    assert "00" in counts
    assert "11" in counts
    assert sum(counts.values()) == shots

    ratio_00 = counts["00"] / shots
    assert 0.5 - tolerance <= ratio_00 <= 0.5 + tolerance


class TestRunPipeline:
    def test_sf_run_cpu_bell_state(self, bell_circuit):
        shots = 2000
        result = sf.run(bell_circuit, device="cpu", shots=shots)

        assert result.shots == shots
        assert result.counts
        _assert_valid_bitstrings(result.counts, bell_circuit.n_qubits)
        _assert_bell_counts(result.counts, shots)

    def test_sf_run_local_device_statevector(self, bell_circuit):
        shots = 1500
        device = LocalDevice("statevector")
        result = sf.run(bell_circuit, device=device, shots=shots)

        assert result.shots == shots
        _assert_valid_bitstrings(result.counts, bell_circuit.n_qubits)
        _assert_bell_counts(result.counts, shots)

    def test_circuit_run_delegates_equivalently(self, bell_circuit):
        shots = 1000
        direct = sf.run(bell_circuit, device="cpu", shots=shots)
        via_method = bell_circuit.run(device="cpu", shots=shots)

        assert via_method.shots == direct.shots == shots
        assert set(via_method.counts.keys()) == set(direct.counts.keys())
        assert sum(via_method.counts.values()) == sum(direct.counts.values()) == shots
        _assert_bell_counts(via_method.counts, shots)

    def test_parameterized_circuit_bind_then_run(self, parametric_circuit):
        bound = parametric_circuit.bind({"theta": 0.0, "phi": 0.0})
        assert bound.n_parameters == 0

        shots = 500
        result = sf.run(bound, device="cpu", shots=shots)

        assert result.shots == shots
        assert sum(result.counts.values()) == shots
        _assert_valid_bitstrings(result.counts, bound.n_qubits)

    def test_unbound_parameters_rejected(self, parametric_circuit):
        with pytest.raises(RuntimeError, match="unbound parameter"):
            sf.run(parametric_circuit, device="cpu")

    def test_counts_keys_are_valid_bitstrings(self, ghz_circuit):
        circuit = ghz_circuit(3)
        shots = 800
        result = sf.run(circuit, device="cpu", shots=shots)

        _assert_valid_bitstrings(result.counts, circuit.n_qubits)
        assert sum(result.counts.values()) == shots

    def test_shots_match_requested(self, bell_circuit):
        for shots in (100, 500, 2000):
            result = sf.run(bell_circuit, device="cpu", shots=shots)
            assert result.shots == shots
            assert sum(result.counts.values()) == shots
