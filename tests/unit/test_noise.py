"""Unit tests for NoiseChannel and NoiseModel construction."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("jax")

from superfermion.noise import (  # noqa: E402
    NoiseChannel,
    NoiseModel,
    ibm_eagle_noise,
    ideal_noise,
)


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
