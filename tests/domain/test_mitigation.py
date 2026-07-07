"""Error mitigation domain tests."""

import numpy as np
import pytest

import superfermion as sf


pytestmark = pytest.mark.domain

mitigation = pytest.importorskip(
    "superfermion.mitigation",
    reason="mitigation module unavailable",
)


def _bell_zz_observable(sv):
    """Expectation of ZZ on Bell state — ideal value is +1."""
    sv = np.asarray(sv, dtype=np.complex128).ravel()
    # ZZ eigenvalues: |00> -> +1, |11> -> +1, |01>/|10> -> -1
    probs = np.abs(sv) ** 2
    n = int(np.log2(len(sv)))
    expval = 0.0
    for i, p in enumerate(probs):
        bitstring = format(i, f"0{n}b")
        z0 = 1 if bitstring[-1] == "0" else -1
        z1 = 1 if bitstring[-2] == "0" else -1
        expval += p * z0 * z1
    return float(expval)


class TestZNE:
    def test_zne_on_bell_circuit_noiseless(self, bell_circuit):
        result = mitigation.zne(
            bell_circuit,
            _bell_zz_observable,
            scale_factors=[1, 2, 3],
            device="cpu",
        )
        assert isinstance(result, float)
        assert abs(result - 1.0) < 0.05

    def test_zne_richardson_extrapolation_two_points(self, bell_circuit):
        result = mitigation.zne(
            bell_circuit,
            _bell_zz_observable,
            scale_factors=[1, 2],
            device="cpu",
        )
        assert np.isfinite(result)

    def test_fold_circuit_amplifies_gate_count(self, bell_circuit):
        folded = mitigation._fold_circuit(bell_circuit, scale=2)
        assert folded.gate_count == 3 * bell_circuit.gate_count

    def test_zne_with_noisy_density_matrix(self, bell_circuit):
        from superfermion.noise import NoiseModel

        noise = NoiseModel()
        noise.add_depolarizing(0.01, n_qubits=1)

        def noisy_obs(sv):
            return _bell_zz_observable(sv)

        expectations = []
        for scale in [1, 2, 3]:
            folded = mitigation._fold_circuit(bell_circuit, scale)
            result = sf.run(folded, device="cpu", method="density_matrix",
                            shots=0, noise_model=noise)
            rho = result.metadata["density_matrix"]
            n = folded.n_qubits
            probs = np.real(np.diag(rho))
            expectations.append(float(np.sum(probs)))

        zne_value = mitigation._richardson_extrapolation([1, 2, 3], expectations)
        assert np.isfinite(zne_value)


class TestReadoutCorrection:
    def test_readout_correction_without_matrix_returns_unchanged(self):
        counts = {"00": 500, "11": 500}
        corrected = mitigation.readout_correction(counts)
        assert corrected == counts
