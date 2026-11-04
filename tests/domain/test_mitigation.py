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


def _depolarizing(p: float = 0.02):
    """2-qubit depolarizing model used by the MIT-1 tests."""
    from superfermion.noise import NoiseModel

    noise = NoiseModel()
    noise.add_depolarizing(p, qubits=[0, 1])
    return noise


class TestZNEUnderNoise:
    """MIT-1: folding only mitigates once the extra gates carry noise.

    ``_fold_circuit`` is unitary-exact, so on a noiseless simulator every
    scale returns the same number and the extrapolation is a structural
    no-op.  That cannot be "fixed" in the fold; it needs a noise model, and
    these tests pin the real, noise-carrying path.
    """

    def test_noiseless_extrapolation_warns_that_it_changed_nothing(
        self, bell_circuit
    ):
        with pytest.warns(RuntimeWarning, match="extrapolation changed nothing"):
            value = mitigation.zne(
                bell_circuit, _bell_zz_observable, scale_factors=[1, 2, 3]
            )
        assert abs(value - 1.0) < 0.05

    def test_noise_model_without_observable_points_at_the_fix(self, bell_circuit):
        with pytest.raises(ValueError, match="observable"):
            mitigation.zne(
                bell_circuit,
                _bell_zz_observable,
                scale_factors=[1, 2, 3],
                noise_model=_depolarizing(),
            )

    def test_folding_under_noise_degrades_the_observable(self, bell_circuit):
        raw = mitigation._folded_expectations(
            bell_circuit,
            _bell_zz_observable,
            [1, 2, 3],
            device="cpu",
            method="statevector",
            noise_model=_depolarizing(),
            observable="ZZ",
        )
        assert raw[0] < 1.0  # noise already costs accuracy at scale 1
        assert raw[0] > raw[1] > raw[2]  # and each further fold costs more

    def test_mitigation_moves_the_answer_away_from_the_plain_value(
        self, bell_circuit
    ):
        noise = _depolarizing()
        single_fold = mitigation._folded_expectations(
            bell_circuit,
            _bell_zz_observable,
            [1],
            device="cpu",
            method="statevector",
            noise_model=noise,
            observable="ZZ",
        )[0]
        mitigated = mitigation.zne(
            bell_circuit,
            _bell_zz_observable,
            scale_factors=[1, 2, 3],
            noise_model=noise,
            observable="ZZ",
        )
        # The whole point of MIT-1: it is no longer bit-for-bit the plain value.
        assert np.isfinite(mitigated)
        assert mitigated != pytest.approx(1.0, abs=1e-6)
        # ... and it recovers accuracy relative to the unmitigated fold.
        assert abs(mitigated - 1.0) < abs(single_fold - 1.0)

    def test_the_noisy_path_really_applies_the_model(self, bell_circuit):
        result = sf.run(
            bell_circuit, method="density_matrix", shots=0, noise_model=_depolarizing()
        )
        assert result.metadata["purity"] < 0.999

    def test_zne_with_calibration_reports_whether_the_model_was_used(
        self, bell_circuit
    ):
        noise = _depolarizing()
        plain = mitigation.zne_with_calibration(
            bell_circuit, _bell_zz_observable, calibration=noise, scale_factors=[1, 2, 3]
        )
        assert plain["noise_model_used"] is False
        assert abs(plain["zne_value"] - 1.0) < 1e-9

        used = mitigation.zne_with_calibration(
            bell_circuit,
            _bell_zz_observable,
            calibration=noise,
            scale_factors=[1, 2, 3],
            observable="ZZ",
        )
        assert used["noise_model_used"] is True
        assert used["raw_values"] != plain["raw_values"]


class TestFoldCircuitIntegrity:
    """The fold must reproduce the circuit, not an approximation of it."""

    def test_an_unfoldable_gate_raises_instead_of_being_dropped(self):
        from superfermion.circuit import GateRecord

        circuit = sf.Circuit(2)
        circuit.h(0)
        circuit._gates.append(GateRecord(name="MYSTERY", qubits=[0]))

        with pytest.raises(ValueError, match="MYSTERY"):
            mitigation._fold_circuit(circuit, 2)

    def test_a_matrix_only_gate_is_folded_by_its_adjoint(self):
        from superfermion.circuit import GateRecord

        phase = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
        circuit = sf.Circuit(1)
        circuit.h(0)
        circuit._gates.append(GateRecord(name="MATRIXONLY", qubits=[0], matrix=phase))

        folded = mitigation._fold_circuit(circuit, 2)
        appended = folded._gates[-3:]

        assert [gate.name for gate in appended] == ["UNITARY"] * 3
        assert np.asarray(appended[0].matrix) == pytest.approx(phase)
        assert np.asarray(appended[1].matrix) == pytest.approx(phase.conj().T)
        assert np.asarray(appended[2].matrix) == pytest.approx(phase)

    def test_the_classical_register_is_preserved(self):
        circuit = sf.Circuit(2, n_cbits=3)
        circuit.x(0)
        circuit.measure(0, 2)

        folded = mitigation._fold_circuit(circuit, 2)

        assert folded.n_cbits == 3
        assert folded._gates[-1].name == "MEASURE"
        assert folded._gates[-1].classical_bits == [2]
