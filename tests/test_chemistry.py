"""
Unit test for Quantum Chemistry module.
"""

import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.chemistry import get_molecular_hamiltonian, uccsd_ansatz

def test_h2_hamiltonian():
    H = get_molecular_hamiltonian("H2")
    assert len(H.terms) == 5
    assert "ZI" in [t.pauli_str for t in H.terms]
    assert "XX" in [t.pauli_str for t in H.terms]

def test_uccsd_vqe():
    # 1. Define H2 molecule
    H = get_molecular_hamiltonian("H2")
    
    # 2. Build UCCSD ansatz (2 qubits for minimal H2, 2 electrons)
    ansatz = uccsd_ansatz(n_qubits=2, n_electrons=2)
    
    # 3. Simulate and check energy
    # Initial state should be Hartree-Fock |11>
    sim = sf.backends.get_backend("jax")
    # All initial params zero -> |11>
    params = jnp.zeros(len(ansatz.parameters))
    state = sim.simulate(ansatz, params)
    
    # Probabilities should be 1.0 for |11> (index 3)
    probs = jnp.abs(state)**2
    assert jnp.allclose(probs[3], 1.0, atol=1e-5)
    
    # Energy at HF level
    energy_hf = H.expectation(state)
    print(f"H2 Hartree-Fock Energy: {energy_hf:.6f}")
    assert energy_hf < 0
