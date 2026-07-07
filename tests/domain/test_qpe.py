"""Quantum Phase Estimation domain tests."""

import pytest

import superfermion as sf
from superfermion.algorithms.qpe import quantum_phase_estimation


pytestmark = pytest.mark.domain


class TestQPE:
    def test_qpe_estimates_t_gate_phase(self):
        unitary = sf.Circuit(1).t(0)

        def prep_eigenstate(circuit: sf.Circuit) -> None:
            circuit.x(0)

        result = quantum_phase_estimation(
            unitary_circuit=unitary,
            eigenstate_prep=prep_eigenstate,
            precision_bits=4,
            device="cpu",
        )

        expected_phase = 0.125  # T gate: e^{2πi·(1/8)} on |1⟩
        assert abs(result["phase"] - expected_phase) < 0.1
        assert result["precision_bits"] == 4
        assert result["probability"] > 0.5

    def test_qpe_runs_and_returns_result(self):
        unitary = sf.Circuit(1).t(0)

        def prep_eigenstate(circuit: sf.Circuit) -> None:
            circuit.x(0)

        result = quantum_phase_estimation(
            unitary_circuit=unitary,
            eigenstate_prep=prep_eigenstate,
            precision_bits=4,
            device="cpu",
        )

        assert result["precision_bits"] == 4
        assert "phase" in result
        assert "phase_binary" in result
        assert "eigenvalue" in result
