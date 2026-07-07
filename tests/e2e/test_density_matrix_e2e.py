"""End-to-end tests for density matrix simulation."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.noise import NoiseModel


pytestmark = pytest.mark.e2e


class TestDensityMatrixViaRun:
    def test_bell_state_dm_sampling(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="density_matrix", shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        total = sum(result.counts.values())
        ratio = result.counts.get("00", 0) / total
        assert 0.4 < ratio < 0.6

    def test_x_gate_dm(self):
        qc = Circuit(1).x(0)
        result = sf.run(qc, device="cpu", method="density_matrix", shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_dm_metadata_has_density_matrix(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="density_matrix", shots=0)
        rho = result.metadata.get("density_matrix")
        assert rho is not None
        assert rho.shape == (4, 4)
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_dm_purity_pure_state(self, bell_circuit):
        result = sf.run(bell_circuit, device="cpu", method="density_matrix", shots=0)
        rho = result.metadata["density_matrix"]
        purity = float(np.real(np.trace(rho @ rho)))
        assert purity == pytest.approx(1.0, abs=1e-6)


class TestNoisyDensityMatrix:
    def test_noisy_bell_reduces_purity(self, bell_circuit):
        ideal_result = sf.run(bell_circuit, device="cpu", method="density_matrix", shots=0)
        nm = NoiseModel().add_depolarizing(0.1)
        noisy_result = sf.run(bell_circuit, device="cpu", method="density_matrix", shots=0, noise_model=nm)

        rho_ideal = ideal_result.metadata["density_matrix"]
        rho_noisy = noisy_result.metadata["density_matrix"]

        purity_ideal = float(np.real(np.trace(rho_ideal @ rho_ideal)))
        purity_noisy = float(np.real(np.trace(rho_noisy @ rho_noisy)))
        assert purity_ideal > purity_noisy

    def test_noisy_trace_preserved(self, bell_circuit):
        nm = NoiseModel().add_depolarizing(0.05)
        result = sf.run(bell_circuit, device="cpu", method="density_matrix", shots=0, noise_model=nm)
        rho = result.metadata["density_matrix"]
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_amplitude_damping(self):
        qc = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.5)
        result = sf.run(qc, device="cpu", method="density_matrix", shots=0, noise_model=nm)
        rho = result.metadata["density_matrix"]
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-10)
        assert rho[0, 0] > 0.1

    def test_x_gate_noiseless_dm(self):
        qc = Circuit(1).x(0)
        result = sf.run(qc, device="cpu", method="density_matrix", shots=0)
        rho = result.metadata["density_matrix"]
        expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    def test_noisy_dm_via_run_kwargs(self, bell_circuit):
        nm = NoiseModel().add_depolarizing(0.05)
        result = sf.run(
            bell_circuit, device="cpu", method="density_matrix",
            shots=0, noise_model=nm,
        )
        rho = result.metadata["density_matrix"]
        purity = float(np.real(np.trace(rho @ rho)))
        assert purity < 0.99
