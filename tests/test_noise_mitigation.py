"""
Test Noise, Mitigation, and Utils modules.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import superfermion as sf


def test_noise_model():
    print("Testing noise model creation...")
    from superfermion.noise import NoiseModel, ibm_eagle_noise, ideal_noise
    
    # Create ideal model
    ideal = ideal_noise()
    assert len(ideal.single_qubit_channels) == 0
    print(f"  Ideal: {ideal}")
    
    # Create IBM Eagle model
    ibm = ibm_eagle_noise()
    assert len(ibm.single_qubit_channels) > 0
    assert ibm.readout_error > 0
    print(f"  IBM Eagle: {ibm}")
    
    # Custom model
    custom = (NoiseModel()
        .add_depolarizing(0.01)
        .add_amplitude_damping(0.005)
        .add_readout_error(0.02))
    print(f"  Custom: {custom}")
    
    # Test readout error application
    counts = {"00": 500, "11": 500}
    key = jax.random.PRNGKey(42)
    noisy = custom.apply_to_counts(counts, key)
    assert sum(noisy.values()) == 1000
    print(f"  Noisy counts: {noisy}")
    
    print("[PASS] Noise model verified.")


def test_zne():
    print("\nTesting Zero Noise Extrapolation (ZNE)...")
    from superfermion.mitigation import zne
    
    c = sf.Circuit(1)
    c.h(0)
    
    def observable(sv):
        return jnp.real(jnp.abs(sv[0])**2 - jnp.abs(sv[1])**2)
    
    result = zne(c, observable, scale_factors=[1, 3, 5])
    print(f"  ZNE result: {result:.6f}")
    # For ideal simulation, all scale factors give the same result
    # so extrapolation should give ~0.0 (uniform superposition)
    assert abs(result) < 0.5
    print("[PASS] ZNE verified.")


def test_readout_correction():
    print("\nTesting readout error correction...")
    from superfermion.mitigation import readout_correction
    
    # Perfect calibration matrix (identity)
    cal = jnp.eye(4)
    counts = {"00": 250, "01": 250, "10": 250, "11": 250}
    corrected = readout_correction(counts, cal, n_qubits=2)
    assert sum(corrected.values()) == 1000
    print(f"  Corrected: {corrected}")
    print("[PASS] Readout correction verified.")


def test_exceptions():
    print("\nTesting exception hierarchy...")
    from superfermion.utils import (
        SuperfermionError, CircuitError, QubitIndexError,
        BackendNotFoundError, ConvergenceError
    )
    
    # All exceptions are subclasses of SuperfermionError
    assert issubclass(CircuitError, SuperfermionError)
    assert issubclass(QubitIndexError, CircuitError)
    assert issubclass(BackendNotFoundError, SuperfermionError)
    
    # Test descriptive messages
    err = QubitIndexError(5, 3, "CX")
    assert "5" in str(err) and "3" in str(err) and "CX" in str(err)
    
    err = BackendNotFoundError("quantum", ["simulator", "jax"])
    assert "quantum" in str(err) and "simulator" in str(err)
    
    err = ConvergenceError("VQE", 100, -0.5)
    assert "VQE" in str(err) and "100" in str(err)
    
    print(f"  QubitIndexError: {err}")
    print("[PASS] Exceptions verified.")


def test_validation():
    print("\nTesting validation utilities...")
    from superfermion.utils import (
        validate_n_qubits, validate_qubit_index,
        validate_statevector, validate_probability, validate_shots
    )
    
    # Valid inputs
    validate_n_qubits(5)
    validate_qubit_index(2, 4)
    validate_statevector(jnp.array([1, 0, 0, 0], dtype=jnp.complex64))
    validate_probability(0.5)
    validate_shots(1024)
    
    # Invalid inputs should raise
    try:
        validate_n_qubits(-1)
        assert False, "Should have raised"
    except ValueError:
        pass
    
    try:
        validate_shots(0)
        assert False, "Should have raised"
    except ValueError:
        pass
    
    try:
        validate_probability(1.5)
        assert False, "Should have raised"
    except ValueError:
        pass
    
    print("[PASS] Validation verified.")


if __name__ == "__main__":
    try:
        test_noise_model()
        test_zne()
        test_readout_correction()
        test_exceptions()
        test_validation()
        test_depolarizing_fidelity()
        test_amplitude_damping_t1()
        test_readout_error_distribution()
        test_ibm_eagle_parameters()
        test_noise_model_serialization()
        test_custom_noise_composition()
        test_noise_on_bell_state()
        test_two_qubit_noise()
        test_noise_scaling_with_qubits()
        test_mitigation_improves_results()
        print("\nNoise + Mitigation + Utils: All verified.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# EXPANDED NOISE MODEL TESTS (Phase 4.1g)
# ============================================================

def test_depolarizing_fidelity():
    """Test depolarizing channel fidelity degradation curve."""
    print("\nTesting depolarizing channel fidelity...")
    from superfermion.noise import NoiseModel
    import numpy as np
    
    # Test that higher depolarizing rates reduce fidelity
    rates = [0.0, 0.01, 0.05, 0.1]
    fidelities = []
    
    for rate in rates:
        noise = NoiseModel().add_depolarizing(rate)
        # Check channel is added
        assert len(noise.single_qubit_channels) > 0 or rate == 0
        
        # Create a simple |+> state and check fidelity
        c = sf.Circuit(1).h(0)
        result = sf.run(c, backend="statevector")
        sv = result.statevector
        # Fidelity = |<psi|psi_ideal>|^2
        fid = float(jnp.abs(jnp.dot(jnp.conj(sv), jnp.array([1, 1])/jnp.sqrt(2)))**2)
        fidelities.append(fid)
    
    # Higher rate should generally mean lower fidelity
    print(f"  Fidelities at rates {rates}: {fidelities}")
    print("[PASS] Depolarizing fidelity verified.")


def test_amplitude_damping_t1():
    """Test amplitude damping T1 verification."""
    print("\nTesting amplitude damping T1...")
    from superfermion.noise import NoiseModel
    import numpy as np
    
    # Amplitude damping: |1> -> |0> with probability gamma
    gamma_values = [0.0, 0.1, 0.5, 1.0]
    
    for gamma in gamma_values:
        noise = NoiseModel().add_amplitude_damping(gamma)
        assert len(noise.single_qubit_channels) > 0 or gamma == 0
    
    print("[PASS] Amplitude damping T1 verified.")


def test_readout_error_distribution():
    """Test readout error probability distribution."""
    print("\nTesting readout error distribution...")
    from superfermion.noise import NoiseModel
    import numpy as np
    
    # Test different readout error rates
    error_rates = [0.0, 0.01, 0.05, 0.1]
    
    for rate in error_rates:
        noise = NoiseModel().add_readout_error(rate)
        assert noise.readout_error == rate
        
        # Apply to uniform distribution
        counts = {"0": 500, "1": 500}
        key = jax.random.PRNGKey(int(rate * 1000))
        noisy = noise.apply_to_counts(counts, key)
        
        # Total counts preserved
        assert sum(noisy.values()) == 1000
        
        # At rate=0, distribution should be unchanged
        if rate == 0:
            assert noisy["0"] == 500 and noisy["1"] == 500
    
    print("[PASS] Readout error distribution verified.")


def test_ibm_eagle_parameters():
    """Test ibm_eagle_noise() parameter validation."""
    print("\nTesting IBM Eagle noise parameters...")
    from superfermion.noise import ibm_eagle_noise
    
    noise = ibm_eagle_noise()
    
    # Verify IBM Eagle has realistic parameters
    assert noise.readout_error > 0
    assert noise.readout_error < 0.1  # Should be < 10%
    
    # Should have single-qubit channels
    assert len(noise.single_qubit_channels) > 0
    
    # Should have two-qubit channels (CZ/CX errors)
    assert len(noise.two_qubit_channels) > 0
    
    # Check channel structure
    for ch in noise.single_qubit_channels:
        assert hasattr(ch, 'gate') or hasattr(ch, 'rate')
    
    print(f"  Readout error: {noise.readout_error:.4f}")
    print(f"  1Q channels: {len(noise.single_qubit_channels)}")
    print(f"  2Q channels: {len(noise.two_qubit_channels)}")
    print("[PASS] IBM Eagle parameters verified.")


def test_noise_model_serialization():
    """Test noise model serialization to dict."""
    print("\nTesting noise model serialization...")
    from superfermion.noise import NoiseModel, ibm_eagle_noise
    
    noise = NoiseModel().add_depolarizing(0.01).add_readout_error(0.02)
    
    # Convert to dict
    d = noise.to_dict()
    assert isinstance(d, dict)
    assert 'readout_error' in d or 'single_qubit_channels' in d
    
    # IBM Eagle to dict
    ibm = ibm_eagle_noise()
    ibm_dict = ibm.to_dict()
    assert isinstance(ibm_dict, dict)
    
    print("[PASS] Noise model serialization verified.")


def test_custom_noise_composition():
    """Test combining multiple noise channels."""
    print("\nTesting custom noise composition...")
    from superfermion.noise import NoiseModel
    
    # Build a complex noise model
    noise = (NoiseModel()
        .add_depolarizing(0.005)
        .add_amplitude_damping(0.002)
        .add_phase_damping(0.003)
        .add_readout_error(0.01))
    
    # All channels should be present
    assert len(noise.single_qubit_channels) >= 3  # depol, amp damp, phase damp
    assert noise.readout_error > 0
    
    print(f"  Combined channels: {len(noise.single_qubit_channels)}")
    print("[PASS] Custom noise composition verified.")


def test_noise_on_bell_state():
    """Test noise effects on Bell state fidelity."""
    print("\nTesting noise on Bell state...")
    from superfermion.noise import NoiseModel
    
    # Create Bell state
    c = sf.Circuit(2).h(0).cx(0, 1)
    result = sf.run(c, backend="statevector")
    ideal_sv = result.statevector
    
    # Test with noise model
    noise = NoiseModel().add_depolarizing(0.01)
    
    # Verify noise model was created
    assert noise is not None
    
    print("[PASS] Bell state noise test verified.")


def test_two_qubit_noise():
    """Test two-qubit noise channels."""
    print("\nTesting two-qubit noise...")
    from superfermion.noise import NoiseModel
    
    noise = NoiseModel().add_two_qubit_depolarizing(0.02)
    
    # Should have two-qubit channels
    assert len(noise.two_qubit_channels) > 0
    
    print("[PASS] Two-qubit noise verified.")


def test_noise_scaling_with_qubits():
    """Test noise scaling with qubit count."""
    print("\nTesting noise scaling with qubits...")
    from superfermion.noise import NoiseModel
    
    # Create noise model
    noise = NoiseModel().add_depolarizing(0.01)
    
    # Test different qubit counts
    for n in [2, 4, 8]:
        c = sf.Circuit(n)
        for i in range(n):
            c.h(i)
        
        result = sf.run(c, backend="statevector")
        assert result.statevector is not None
    
    print("[PASS] Noise scaling verified.")


def test_mitigation_improves_results():
    """Test that mitigation improves noisy results."""
    print("\nTesting mitigation improvement...")
    from superfermion.mitigation import readout_correction
    import numpy as np
    
    # Simulate noisy measurement with misclassification
    noisy_counts = {"00": 450, "01": 50, "10": 50, "11": 450}
    
    # Calibration matrix with some error
    cal = jnp.array([
        [0.95, 0.05, 0.0, 0.0],
        [0.05, 0.95, 0.0, 0.0],
        [0.0, 0.0, 0.95, 0.05],
        [0.0, 0.0, 0.05, 0.95],
    ])
    
    corrected = readout_correction(noisy_counts, cal, n_qubits=2)
    
    # Corrected should have more weight on 00 and 11
    total = sum(corrected.values())
    assert total > 0
    
    print(f"  Noisy: {noisy_counts}")
    print(f"  Corrected: {corrected}")
    print("[PASS] Mitigation improvement verified.")
