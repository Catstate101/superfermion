"""Unit tests for NoiseChannel and NoiseModel construction."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("jax")

from superfermion.noise import (  # noqa: E402
    NoiseChannel,
    NoiseModel,
    ibm_eagle_noise,
    ideal_noise,
    kraus_amplitude_damping,
    kraus_depolarizing_1q,
    kraus_phase_damping,
)


def _flat(kraus_set, dim=2):
    """Row-major re/im flat form of a Kraus set (the Rust binding layout)."""
    return [
        float(v)
        for K in kraus_set
        for r in range(dim)
        for c in range(dim)
        for v in (K[r, c].real, K[r, c].imag)
    ]


class TestNoiseChannel:
    def test_gate_and_rate_aliases(self):
        channel = NoiseChannel("depolarizing", 0.01, lambda key, sv: sv)
        assert channel.gate == "depolarizing"
        assert channel.rate == 0.01


class TestNoiseModelConstruction:
    def test_empty_model(self):
        model = NoiseModel()
        assert model.single_qubit_channels == []
        assert model.two_qubit_channels == []
        assert model.readout_error == 0.0

    def test_add_depolarizing_single_qubit(self):
        model = NoiseModel().add_depolarizing(0.01)
        assert len(model.single_qubit_channels) == 1
        assert model.single_qubit_channels[0].name == "depolarizing"
        assert model.single_qubit_channels[0].error_rate == 0.01
        assert model.two_qubit_channels == []

    def test_add_depolarizing_two_qubit(self):
        model = NoiseModel().add_depolarizing(0.02, n_qubits=2)
        assert len(model.two_qubit_channels) == 1
        assert model.two_qubit_channels[0].error_rate == 0.02
        assert model.single_qubit_channels == []

    def test_add_two_qubit_depolarizing_wrapper(self):
        model = NoiseModel().add_two_qubit_depolarizing(0.03)
        assert len(model.two_qubit_channels) == 1
        assert model.two_qubit_channels[0].error_rate == 0.03

    def test_add_amplitude_damping(self):
        model = NoiseModel().add_amplitude_damping(0.005)
        assert len(model.single_qubit_channels) == 1
        assert model.single_qubit_channels[0].name == "amplitude_damping"
        assert model.single_qubit_channels[0].error_rate == 0.005

    def test_add_phase_damping(self):
        model = NoiseModel().add_phase_damping(0.001)
        assert len(model.single_qubit_channels) == 1
        assert model.single_qubit_channels[0].name == "phase_damping"
        assert model.single_qubit_channels[0].error_rate == 0.001

    def test_add_readout_error(self):
        model = NoiseModel().add_readout_error(0.02)
        assert model.readout_error == 0.02

    def test_chaining_returns_self(self):
        model = (
            NoiseModel()
            .add_depolarizing(0.01)
            .add_amplitude_damping(0.005)
            .add_phase_damping(0.001)
            .add_readout_error(0.02)
        )
        assert len(model.single_qubit_channels) == 3
        assert model.readout_error == 0.02


class TestNoiseModelFactories:
    def test_ideal_noise_is_empty(self):
        model = ideal_noise()
        assert model.single_qubit_channels == []
        assert model.two_qubit_channels == []
        assert model.readout_error == 0.0

    def test_ibm_eagle_noise_has_expected_channels(self):
        model = ibm_eagle_noise()
        assert len(model.single_qubit_channels) == 3
        assert len(model.two_qubit_channels) == 1
        assert model.readout_error == 0.01
        names = {ch.name for ch in model.single_qubit_channels}
        assert names == {"depolarizing", "amplitude_damping", "phase_damping"}


class TestNoiseModelSerialization:
    def test_to_dict_roundtrip_structure(self):
        model = (
            NoiseModel()
            .add_depolarizing(0.01)
            .add_depolarizing(0.02, n_qubits=2)
            .add_readout_error(0.03)
        )
        data = model.to_dict()
        assert data["readout_error"] == 0.03
        assert data["single_qubit_channels"] == [
            {"name": "depolarizing", "error_rate": 0.01}
        ]
        assert data["two_qubit_channels"] == [
            {"name": "depolarizing", "error_rate": 0.02}
        ]

    def test_repr(self):
        model = NoiseModel().add_depolarizing(0.01).add_readout_error(0.02)
        assert "NoiseModel" in repr(model)
        assert "readout_error=0.02" in repr(model)


class TestPlacementAndConvention:
    """NoiseModel(placement=...) and add_depolarizing(convention=...).

    Both axes default to the historical SuperFermion semantics (placement
    "touch", convention "total"); the non-default values exist purely for
    Qiskit/Aer parity and must not change the defaults.
    """

    def test_default_placement_is_touch(self):
        model = NoiseModel()
        assert model.placement == "touch"
        assert model.to_dict()["placement"] == "touch"
        assert "placement='touch'" in repr(model)

    def test_gate_placement_accepted_and_serialized(self):
        model = NoiseModel(placement="gate").add_depolarizing(0.01)
        assert model.placement == "gate"
        assert model.to_dict()["placement"] == "gate"
        assert "placement='gate'" in repr(model)

    def test_invalid_placement_raises(self):
        with pytest.raises(ValueError, match="placement"):
            NoiseModel(placement="bogus")

    def test_qiskit_convention_rescales_1q_kraus(self):
        lam = 0.05
        qiskit = NoiseModel().add_depolarizing(lam, convention="qiskit")
        total = NoiseModel().add_depolarizing(lam * 3.0 / 4.0)
        a = np.array(qiskit.to_rust_kraus_ops(1)[0][1])
        b = np.array(total.to_rust_kraus_ops(1)[0][1])
        # entries are sqrt-scaled, so only float-associativity slack is allowed
        assert np.max(np.abs(a - b)) < 1e-15
        # the nominal rate survives for introspection / serialization
        ch = qiskit.single_qubit_channels[0]
        assert ch.error_rate == lam
        assert ch.convention == "qiskit"
        assert qiskit.to_dict()["single_qubit_channels"] == [
            {"name": "depolarizing", "error_rate": lam}
        ]

    def test_qiskit_convention_rescales_2q_kraus(self):
        lam = 0.08
        a = np.array(
            NoiseModel()
            .add_two_qubit_depolarizing(lam, convention="qiskit")
            .to_rust_kraus_ops_2q()[0]
        )
        b = np.array(
            NoiseModel()
            .add_two_qubit_depolarizing(lam * 15.0 / 16.0)
            .to_rust_kraus_ops_2q()[0]
        )
        assert np.max(np.abs(a - b)) < 1e-15

    def test_qiskit_convention_matches_analytic_channel(self):
        """Ground truth: E(rho) = (1-lam) rho + lam Tr(rho) I/2."""
        lam = 0.2
        rho = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1><1|
        out = NoiseModel().add_depolarizing(lam, convention="qiskit").apply_1q(
            rho, 0, 1
        )
        expected = (1 - lam) * rho + lam * np.eye(2) / 2
        assert np.max(np.abs(out - expected)) < 1e-12
        # identical to the total convention at p = 3*lam/4
        out_total = NoiseModel().add_depolarizing(3 * lam / 4).apply_1q(rho, 0, 1)
        assert np.max(np.abs(out - out_total)) < 1e-12

    def test_total_convention_is_the_default(self):
        explicit = NoiseModel().add_depolarizing(0.01, convention="total")
        implicit = NoiseModel().add_depolarizing(0.01)
        assert np.array_equal(
            np.array(explicit.to_rust_kraus_ops(1)[0][1]),
            np.array(implicit.to_rust_kraus_ops(1)[0][1]),
        )

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="convention"):
            NoiseModel().add_depolarizing(0.01, convention="bogus")
        with pytest.raises(ValueError, match="convention"):
            NoiseModel().add_two_qubit_depolarizing(0.01, convention="bogus")


class TestTargetedChannels:
    """``add_*(..., qubits=...)`` — per-wire targeting.

    Additive on top of the historical all-qubit default, whose exact output
    format is pinned byte-for-byte here.
    """

    def test_untargeted_output_is_byte_identical_to_historical_composition(
        self,
    ):
        model = (
            NoiseModel()
            .add_depolarizing(0.03)
            .add_amplitude_damping(0.02)
            .add_phase_flip(0.01)
        )
        ops = dict(model.to_rust_kraus_ops(3))
        composite = None
        for kraus_set in model._1q_kraus:
            composite = (
                list(kraus_set)
                if composite is None
                else [L @ K for K in composite for L in kraus_set]
            )
        assert sorted(ops) == [0, 1, 2]
        assert all(ops[q] == _flat(composite) for q in range(3))

    def test_sequence_targets_deduplicate_in_order(self):
        model = NoiseModel().add_depolarizing(0.01, qubits=[2, 0, 2])
        assert model.single_qubit_channels[0].qubits == (2, 0)
        assert sorted(dict(model.to_rust_kraus_ops(3))) == [0, 2]
        assert NoiseModel().add_depolarizing(
            0.01, qubits=1
        ).single_qubit_channels[0].qubits == (1,)

    def test_only_targeted_wires_are_emitted(self):
        model = (
            NoiseModel()
            .add_depolarizing(0.3, qubits=[1])
            .add_amplitude_damping(0.2, qubits=0)
        )
        ops = dict(model.to_rust_kraus_ops(3))
        assert sorted(ops) == [0, 1]
        assert ops[0] == _flat(kraus_amplitude_damping(0.2))
        assert ops[1] == _flat(kraus_depolarizing_1q(0.3))

    def test_composition_on_one_wire_keeps_add_order(self):
        model = (
            NoiseModel()
            .add_depolarizing(0.1, qubits=1)
            .add_phase_damping(0.05, qubits=1)
        )
        ops = dict(model.to_rust_kraus_ops(2))
        expected = [
            L @ K
            for K in kraus_depolarizing_1q(0.1)
            for L in kraus_phase_damping(0.05)
        ]
        assert ops[1] == _flat(expected)
        assert 0 not in ops

    def test_out_of_range_target_raises_on_conversion(self):
        model = NoiseModel().add_depolarizing(0.01, qubits=2)
        with pytest.raises(ValueError, match="targets qubit 2"):
            model.to_rust_kraus_ops(2)
        model_2q = NoiseModel().add_depolarizing(0.01, n_qubits=2, qubits=(0, 3))
        with pytest.raises(ValueError, match=r"targets qubits \(0, 3\)"):
            model_2q.to_rust_kraus_ops(3)

    def test_invalid_target_specs_raise(self):
        with pytest.raises(ValueError, match="must not be empty"):
            NoiseModel().add_depolarizing(0.01, qubits=[])
        with pytest.raises(ValueError, match="non-negative"):
            NoiseModel().add_depolarizing(0.01, qubits=[-1])
        with pytest.raises(ValueError, match="ordered pair"):
            NoiseModel().add_depolarizing(0.01, n_qubits=2, qubits=(0, 0))
        with pytest.raises(ValueError, match="ordered pair"):
            NoiseModel().add_depolarizing(0.01, n_qubits=2, qubits=(0, 1, 2))
        with pytest.raises(TypeError, match="qubits must be"):
            NoiseModel().add_depolarizing(0.01, qubits=1.5)

    def test_apply_1q_skips_untargeted_wires(self):
        # big-endian layout of the pure-Python path: qubit 0 = MSB
        rho = np.zeros((4, 4), dtype=np.complex128)
        rho[2, 2] = 1.0  # |q0=1, q1=0>
        model = NoiseModel().add_amplitude_damping(1.0, qubits=0)
        assert np.array_equal(model.apply_1q(rho.copy(), 1, 2), rho)
        out = model.apply_1q(rho.copy(), 0, 2)
        assert abs(out[2, 2]) < 1e-15 and abs(out[0, 0] - 1.0) < 1e-15

    def test_apply_2q_matches_the_ordered_pair(self):
        rho = np.zeros((4, 4), dtype=np.complex128)
        rho[3, 3] = 1.0
        model = NoiseModel().add_depolarizing(1.0, n_qubits=2, qubits=(1, 0))
        assert np.array_equal(model.apply_2q(rho.copy(), 0, 1, 2), rho)
        assert not np.allclose(model.apply_2q(rho.copy(), 1, 0, 2), rho)

    def test_two_qubit_targets_list_and_flag(self):
        model = (
            NoiseModel()
            .add_depolarizing(0.1, n_qubits=2, qubits=(0, 1))
            .add_two_qubit_depolarizing(0.05)
        )
        assert model.has_targeted_2q is True
        assert model.to_rust_kraus_ops_2q_targets() == [(0, 1), None]
        assert len(model.to_rust_kraus_ops_2q()) == 2
        assert NoiseModel().add_two_qubit_depolarizing(0.1).has_targeted_2q is False

    def test_to_dict_qubits_key_only_when_targeted(self):
        plain = NoiseModel().add_depolarizing(0.01).to_dict()
        assert plain["single_qubit_channels"] == [
            {"name": "depolarizing", "error_rate": 0.01}
        ]
        targeted = NoiseModel().add_depolarizing(0.01, qubits=[2]).to_dict()
        assert targeted["single_qubit_channels"] == [
            {"name": "depolarizing", "error_rate": 0.01, "qubits": [2]}
        ]
        pair = NoiseModel().add_two_qubit_depolarizing(0.02, qubits=(1, 0)).to_dict()
        assert pair["two_qubit_channels"] == [
            {"name": "depolarizing", "error_rate": 0.02, "qubits": [1, 0]}
        ]


class TestThermalRelaxation:
    """``add_thermal_relaxation`` — Aer-compatible T1/T2 channel."""

    def test_kraus_equals_analytic_compose(self):
        t1, t2, t = 50.0, 70.0, 10.0
        model = NoiseModel().add_thermal_relaxation(t1, t2, t)
        g1 = 1.0 - math.exp(-t / t1)
        g_phi = 1.0 - math.exp(-2.0 * t / t2 + t / t1)
        expected = [
            L @ K
            for K in kraus_amplitude_damping(g1)
            for L in kraus_phase_damping(g_phi)
        ]
        assert len(model._1q_kraus[0]) == 4
        assert _flat(model._1q_kraus[0]) == _flat(expected)

    def test_channel_metadata(self):
        model = NoiseModel().add_thermal_relaxation(50.0, 70.0, 10.0)
        channel = model.single_qubit_channels[0]
        assert channel.name == "thermal_relaxation"
        assert channel.error_rate == 10.0  # the gate time
        assert channel.qubits is None
        assert model.to_dict()["single_qubit_channels"] == [
            {"name": "thermal_relaxation", "error_rate": 10.0}
        ]

    def test_zero_gate_time_is_the_identity_channel(self):
        model = NoiseModel().add_thermal_relaxation(50.0, 70.0, 0.0)
        rho = np.zeros((2, 2), dtype=np.complex128)
        rho[1, 1] = 1.0
        out = model.apply_1q(rho.copy(), 0, 1)
        assert np.allclose(out, rho, atol=1e-15)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match=r"2 \* T1"):
            NoiseModel().add_thermal_relaxation(50.0, 200.0, 10.0)
        with pytest.raises(ValueError, match="positive"):
            NoiseModel().add_thermal_relaxation(0.0, 70.0, 10.0)
        with pytest.raises(ValueError, match="non-negative"):
            NoiseModel().add_thermal_relaxation(50.0, 70.0, -1.0)


class TestNoiseModelJAXRuntime:
    def test_apply_to_counts_no_readout_error(self):
        import jax

        model = NoiseModel()
        counts = {"00": 10, "11": 5}
        try:
            key = jax.random.PRNGKey(0)
        except Exception as exc:
            pytest.skip(f"JAX runtime unavailable: {exc}")
        assert model.apply_to_counts(counts, key) == counts

    def test_apply_to_counts_with_readout_error(self):
        import jax

        model = NoiseModel().add_readout_error(0.5)
        counts = {"0": 100}
        try:
            key = jax.random.PRNGKey(42)
        except Exception as exc:
            pytest.skip(f"JAX runtime unavailable: {exc}")
        noisy = model.apply_to_counts(counts, key)
        assert sum(noisy.values()) == 100
        assert set(noisy.keys()).issubset({"0", "1"})
