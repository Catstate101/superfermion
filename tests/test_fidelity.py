"""
Test script for Superfermion Fidelity Tracking.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from superfermion.qml.fidelity import state_fidelity

def test_fidelity_calculation():
    print("Testing JAX Fidelity Tracking...")
    
    # 1. Base states
    s0 = jnp.array([1, 0], dtype=jnp.complex64)
    s1 = jnp.array([0, 1], dtype=jnp.complex64)
    
    # Fidelity(0, 0) == 1
    f00 = state_fidelity(s0, s0)
    print(f"F(|0>, |0>) = {f00}")
    assert jnp.abs(f00 - 1.0) < 1e-6
    
    # Fidelity(0, 1) == 0
    f01 = state_fidelity(s0, s1)
    print(f"F(|0>, |1>) = {f01}")
    assert jnp.abs(f01) < 1e-6

    # 2. Superpositions
    sp = (s0 + s1) / jnp.sqrt(2.0)
    
    # Fidelity(0, +) == 0.5
    f0p = state_fidelity(s0, sp)
    print(f"F(|0>, |+>) = {f0p}")
    assert jnp.abs(f0p - 0.5) < 1e-6
    
    # Fidelity(+, +) == 1.0
    fpp = state_fidelity(sp, sp)
    print(f"F(|+>, |+>) = {fpp}")
    assert jnp.abs(fpp - 1.0) < 1e-6

    print("[PASS] Fidelity tracking verified.")

if __name__ == "__main__":
    try:
        test_fidelity_calculation()
        print("\nPhase 2: Final Milestone achieved. Monitoring engine online.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
