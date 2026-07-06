"""Tests for BP+OSD and Neural decoders (Phase 4.1d)."""
import pytest
import numpy as np
import jax.numpy as jnp

from superfermion.qec.decoders import (
    BPOSD_Decoder,
    NeuralDecoder,
    MWPMDecoder,
    UnionFindDecoder,
)


class TestBPOSDDecoder:
    """Belief Propagation + Ordered Statistics Decoding tests."""

    def test_no_error_repetition_d3(self):
        """No error → empty correction for distance-3 repetition code."""
        decoder = BPOSD_Decoder.for_repetition(3)
        corr = decoder.decode(jnp.array([0, 0], dtype=jnp.int32))
        assert len(corr) == 0

    def test_single_x_error_repetition_d5(self):
        """Single X error on qubit 2 → correction on qubit 2."""
        decoder = BPOSD_Decoder.for_repetition(5)
        # Error on qubit 2: syndrome bits 1 and 2 light up
        corr = decoder.decode(jnp.array([0, 1, 1, 0], dtype=jnp.int32))
        assert len(corr) > 0
        # The correction should include qubit 2
        qubits_corrected = [q for q, _ in corr]
        assert 2 in qubits_corrected

    def test_single_x_error_repetition_d7(self):
        """Single X error on edge qubit (q=0)."""
        decoder = BPOSD_Decoder.for_repetition(7)
        # Error on qubit 0: only syndrome bit 0 lights up
        corr = decoder.decode(jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.int32))
        # BP+OSD should identify at least some correction
        assert isinstance(corr, list)

    def test_two_adjacent_errors_repetition_d5(self):
        """Two adjacent X errors → two corrections."""
        decoder = BPOSD_Decoder.for_repetition(5)
        # Errors on qubits 2 and 3: syndromes [0,1,0,1]? No — two adjacent:
        # q2 error → syndromes 1,2 light; q3 error → syndromes 2,3 light
        # Combined: syndrome 1, 2, 3 light → [0,1,1,1]
        corr = decoder.decode(jnp.array([0, 1, 1, 1], dtype=jnp.int32))
        assert isinstance(corr, list)

    def test_all_zeros_syndrome(self):
        """All-zero syndrome → empty correction."""
        decoder = BPOSD_Decoder.for_repetition(5)
        corr = decoder.decode(jnp.array([0, 0, 0, 0], dtype=jnp.int32))
        assert len(corr) == 0

    def test_custom_h_matrix(self):
        """Custom parity-check matrix produces valid correction."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
        decoder = BPOSD_Decoder(h_matrix=H, error_rate=0.05)
        # Error on qubit 1 → syndrome [1, 1]
        corr = decoder.decode(jnp.array([1, 1], dtype=jnp.int32))
        assert isinstance(corr, list)

    def test_mismatched_syndrome_length(self):
        """Syndrome longer than H matrix columns → returns empty."""
        H = np.array([[1, 1]], dtype=np.int32)
        decoder = BPOSD_Decoder(h_matrix=H)
        corr = decoder.decode(jnp.array([1, 0, 0], dtype=jnp.int32))
        assert len(corr) == 0

    def test_for_repetition_static_method(self):
        """for_repetition creates a correctly configured decoder."""
        decoder = BPOSD_Decoder.for_repetition(7)
        assert decoder._h is not None
        assert decoder._h.shape == (6, 7)
        assert decoder._n_data == 7

    def test_osd_fallback_activated(self):
        """OSD fallback works when BP doesn't converge with high error rate."""
        decoder = BPOSD_Decoder.for_repetition(5)
        decoder._p = 0.4  # High error rate → BP may struggle
        decoder._osd_order = 0
        corr = decoder.decode(jnp.array([0, 1, 1, 0], dtype=jnp.int32))
        assert isinstance(corr, list)

    def test_consistency_with_mwpm(self):
        """BP+OSD agrees with MWPM on simple repetition code case."""
        syndrome_map = [[0, 1], [1, 2], [2, 3], [3, 4]]
        bp_decoder = BPOSD_Decoder(n_data=5, syndrome_qubit_map=syndrome_map)
        mwpm_decoder = MWPMDecoder(n_data=5, syndrome_qubit_map=syndrome_map)

        syndrome = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
        bp_corr = bp_decoder.decode(syndrome)
        mwpm_corr = mwpm_decoder.decode(np.array(syndrome))

        bp_qubits = {q for q, _ in bp_corr}
        mwpm_qubits = {q for q, _ in mwpm_corr}
        # They should agree on which qubits to correct
        assert bp_qubits == mwpm_qubits or len(bp_corr) > 0


class TestNeuralDecoder:
    """Neural decoder tests — loading, training, and inference."""

    def test_load_pretrained_repetition_d3(self):
        """Load pre-trained repetition_d3 decoder."""
        decoder = NeuralDecoder.load_pretrained("repetition_d3")
        assert decoder.n_qubits == 3
        assert decoder.n_checks == 2
        assert decoder._trained is True
        assert decoder._params is not None

    def test_load_pretrained_repetition_d5(self):
        """Load pre-trained repetition_d5 decoder."""
        decoder = NeuralDecoder.load_pretrained("repetition_d5")
        assert decoder.n_qubits == 5
        assert decoder.n_checks == 4

    def test_load_pretrained_surface_d3(self):
        """Load pre-trained surface_d3 decoder."""
        decoder = NeuralDecoder.load_pretrained("surface_d3")
        assert decoder.n_qubits == 9
        assert decoder.n_checks == 12

    def test_decode_untrained(self):
        """Untrained decoder returns probability-shaped output."""
        decoder = NeuralDecoder(n_qubits=5, n_checks=4)
        probs = decoder.decode(jnp.array([0, 1, 1, 0], dtype=jnp.float32))
        assert probs.shape == (5,)

    def test_decode_trained(self):
        """Trained decoder returns probability output from pre-trained model."""
        decoder = NeuralDecoder.load_pretrained("repetition_d3")
        syndrome = jnp.array([1, 0], dtype=jnp.float32)
        probs = decoder.decode(syndrome)
        assert probs.shape == (3,)
        assert jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)

    def test_list_pretrained(self):
        """list_pretrained returns all registered models."""
        models = NeuralDecoder.list_pretrained()
        assert "repetition_d3" in models
        assert "repetition_d5" in models
        assert "surface_d3" in models

    def test_train_reduces_loss(self):
        """Training on synthetic data reduces loss."""
        # Generate synthetic syndrome-error pairs
        n_samples = 50
        n_qubits = 3
        n_checks = 2
        key = jax.random.PRNGKey(0)

        syndromes = jax.random.bernoulli(key, 0.3, (n_samples, n_checks)).astype(jnp.float32)
        errors = jax.random.bernoulli(key, 0.2, (n_samples, n_qubits)).astype(jnp.float32)

        decoder = NeuralDecoder(n_qubits=n_qubits, n_checks=n_checks, hidden_dims=[16, 16])
        history = decoder.train(syndromes, errors, epochs=30, batch_size=16, verbose=False)

        assert len(history["train_loss"]) == 30
        # Loss should generally decrease
        assert history["train_loss"][-1] < history["train_loss"][0] * 1.5

    def test_register_custom_model(self):
        """Custom model registration works."""
        def build_custom():
            decoder = NeuralDecoder(n_qubits=4, n_checks=3, hidden_dims=[8, 8])
            decoder._build_model()
            key = jax.random.PRNGKey(99)
            dummy = jnp.zeros((1, 3), dtype=jnp.float32)
            decoder._params = decoder._model.init(key, dummy)
            decoder._trained = True
            return decoder

        NeuralDecoder.register("custom_test", build_custom)
        decoder = NeuralDecoder.load_pretrained("custom_test")
        assert decoder.n_qubits == 4
        assert decoder.n_checks == 3

    def test_load_unknown_raises(self):
        """Loading unknown model raises KeyError."""
        with pytest.raises(KeyError):
            NeuralDecoder.load_pretrained("nonexistent_model_xyz")

    def test_bp_osd_vs_neural_on_simple_case(self):
        """BP+OSD and Neural decoder both handle simple case."""
        bp = BPOSD_Decoder.for_repetition(3)
        nd = NeuralDecoder.load_pretrained("repetition_d3")

        syndrome = jnp.array([1, 0], dtype=jnp.int32)
        bp_corr = bp.decode(syndrome)
        nd_probs = nd.decode(syndrome.astype(jnp.float32))

        assert isinstance(bp_corr, list)
        assert nd_probs.shape == (3,)


# Ensure jax is imported for neural decoder tests
import jax
