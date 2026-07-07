"""End-to-end tests for statevector simulation through sf.run()."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.e2e


class TestBellState:
    def test_statevector_exact(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", shots=0)
        sv = result.statevector
        assert sv is not None
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_sampling_counts(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        total = sum(result.counts.values())
        assert total == 10000
        ratio = result.counts.get("00", 0) / total
        assert 0.4 < ratio < 0.6

    def test_metadata_populated(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", shots=100)
        assert result.metadata.get("n_qubits") == 2

    def test_probabilities_from_statevector(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", shots=0)
        probs = result.get_probabilities()
        assert abs(probs.get("00", 0) - 0.5) < 1e-10
        assert abs(probs.get("11", 0) - 0.5) < 1e-10


class TestGHZ:
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_ghz_statevector(self, ghz_circuit, n):
        circuit = ghz_circuit(n)
        result = sf.run(circuit, device="cpu", shots=0)
        sv = result.statevector
        assert len(sv) == 2**n
        probs = np.abs(sv) ** 2
        all_zero = 0
        all_one = 2**n - 1
        assert probs[all_zero] > 0.49
        assert probs[all_one] > 0.49
        assert sum(probs) == pytest.approx(1.0, abs=1e-10)

    def test_ghz_sampling(self, ghz_circuit):
        circuit = ghz_circuit(5)
        result = sf.run(circuit, device="cpu", shots=10000)
        assert set(result.counts.keys()).issubset({"00000", "11111"})


class TestParametricCircuit:
    def test_params_kwarg(self):
        qc = Circuit(2)
        qc.rx(sf.param("theta"), 0)
        qc.ry(sf.param("phi"), 1)
        qc.cx(0, 1)
        result = sf.run(qc, device="cpu", shots=0, params={"theta": 0.0, "phi": 0.0})
        sv = result.statevector
        assert abs(sv[0]) == pytest.approx(1.0, abs=1e-10)

    def test_params_pi_rotation(self):
        qc = Circuit(1)
        qc.rx(sf.param("theta"), 0)
        result = sf.run(qc, device="cpu", shots=0, params={"theta": np.pi})
        sv = result.statevector
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-10)

    def test_unbound_params_raise(self):
        qc = Circuit(1).rx(sf.param("x"), 0)
        with pytest.raises(RuntimeError, match="unbound"):
            sf.run(qc, device="cpu", shots=100)


class TestHardwareEfficientAnsatz:
    def test_he_circuit_statevector_normalization(self):
        n = 6
        qc = Circuit(n)
        for q in range(n):
            qc.ry(sf.param(f"t{q}"), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        params = {f"t{q}": 0.1 * (q + 1) for q in range(n)}
        result = sf.run(qc, device="cpu", shots=0, params=params)
        sv = result.statevector
        assert sum(np.abs(sv) ** 2) == pytest.approx(1.0, abs=1e-10)


class TestStateHandle:
    def test_run_returns_state(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", shots=0)
        assert result.state is not None
        assert result.state.n_qubits == 2
        assert result.state.method == "statevector"

    def test_simulate_returns_state_directly(self, bell_circuit):
        state = sf.simulate(bell_circuit)
        assert state.n_qubits == 2
        assert state.method == "statevector"
