"""
Test script for VQE implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.algorithms.variational import VQE
from superfermion.observables.core import PauliString, Hamiltonian

def test_vqe_simple():
    print("Testing VQE on H = Z + X...")
    
    # 1. Define Hamiltonian H = Z + X
    h = Hamiltonian([
        PauliString("Z", coeffs=1.0),
        PauliString("X", coeffs=1.0)
    ])
    
    # 2. Define Ansatz: One-qubit RY rotation
    # Ground state of Z+X is in the real plane, so RY is enough
    c = sf.Circuit(1)
    c.ry(sf.param("theta"), 0)
    
    # 3. Setup and Run VQE
    vqe = VQE(c, h)
    
    # We expect convergence to -sqrt(2) approx -1.4142
    theoretical_min = -1.414214
    # 5. Minimize
    results = vqe.minimize(iterations=100)
    
    print(f"  Optimized Energy: {results.optimal_value:.6f}")
    print(f"  Theoretical Min:  {theoretical_min:.6f}")
    
    # 6. Assert
    assert jnp.isclose(results.optimal_value, theoretical_min, atol=1e-4)
    print("[PASS] VQE simple test passed!")

if __name__ == "__main__":
    try:
        test_vqe_simple()
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
