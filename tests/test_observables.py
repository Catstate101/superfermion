"""
Test script for Superfermion Observables.
"""

from __future__ import annotations

import superfermion as sf
from superfermion.observables.core import PauliString, Hamiltonian
import numpy as np
import jax.numpy as jnp

def test_pauli_expectation():
    print("Testing PauliString Expectations...")
    
    # |00> state
    state00 = jnp.array([1, 0, 0, 0], dtype=complex)
    
    # <00|IZ|00> should be 1
    iz = PauliString("IZ")
    exp_iz = iz.expectation(state00)
    print(f"<00|IZ|00> = {exp_iz}")
    assert abs(exp_iz - 1.0) < 1e-7
    
    # <00|ZI|00> should be 1
    zi = PauliString("ZI")
    exp_zi = zi.expectation(state00)
    print(f"<00|ZI|00> = {exp_zi}")
    assert abs(exp_zi - 1.0) < 1e-7

    # |11> state
    state11 = jnp.array([0, 0, 0, 1], dtype=complex)
    
    # <11|ZZ|11> should be (-1)*(-1) = 1
    zz = PauliString("ZZ")
    exp_zz = zz.expectation(state11)
    print(f"<11|ZZ|11> = {exp_zz}")
    assert abs(exp_zz - 1.0) < 1e-7

    # |+> state: (1/sqrt(2))(|0> + |1>)
    state_plus = jnp.array([1, 1], dtype=complex) / jnp.sqrt(2.0)
    # <+|X|+> = 1
    x = PauliString("X")
    exp_x = x.expectation(state_plus)
    print(f"<+|X|+> = {exp_x}")
    assert abs(exp_x - 1.0) < 1e-7

    print("[PASS] PauliString expectations verified.")


def test_hamiltonian():
    print("\nTesting Hamiltonian (Sum of Paulis)...")
    # H = 0.5*Z + 0.8*X
    state_0 = jnp.array([1, 0], dtype=complex)
    h = Hamiltonian([
        PauliString("Z", coeffs=0.5),
        PauliString("X", coeffs=0.8)
    ])
    # Expectation on |0>: 0.5*<0|Z|0> + 0.8*<0|X|0> = 0.5*1 + 0.8*0 = 0.5
    exp_val = h.expectation(state_0)
    print(f"<0|H|0> = {exp_val}")
    assert abs(exp_val - 0.5) < 1e-7
    
    print("[PASS] Hamiltonian expectations verified.")

if __name__ == "__main__":
    try:
        test_pauli_expectation()
        test_hamiltonian()
        print("\nObservables layer officially stabilized.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
