"""Amplitude estimation domain tests."""

import pytest

import superfermion as sf
from superfermion.algorithms.amplitude_estimation import amplitude_estimation


pytestmark = [pytest.mark.domain, pytest.mark.timeout(30)]


class TestAmplitudeEstimation:
    def test_iterative_ae_balanced_superposition(self):
        def state_prep(circuit: sf.Circuit) -> None:
            circuit.h(0)

        def oracle(circuit: sf.Circuit) -> None:
            circuit.z(0)

        result = amplitude_estimation(
            state_prep=state_prep,
            oracle=oracle,
            precision_bits=4,
            n_qubits=1,
            device="cpu",
            ae_method="iterative",
        )

        assert result["method"] == "iterative"
        assert abs(result["probability"] - 0.5) < 0.2

    def test_canonical_ae_balanced_superposition(self):
        def state_prep(circuit: sf.Circuit) -> None:
            circuit.h(0)

        def oracle(circuit: sf.Circuit) -> None:
            circuit.z(0)

        result = amplitude_estimation(
            state_prep=state_prep,
            oracle=oracle,
            precision_bits=3,
            n_qubits=1,
            device="cpu",
            ae_method="canonical",
        )

        assert result["method"] == "canonical"
        assert 0.0 <= result["amplitude"] <= 1.0
        assert 0.0 <= result["probability"] <= 1.0
