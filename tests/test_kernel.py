"""
Test Quantum Kernel methods.
"""

from __future__ import annotations
import jax.numpy as jnp
import superfermion as sf
from superfermion.algorithms import QuantumKernel
from superfermion.qml.encoding import iqp_encoding


def test_quantum_kernel():
    print("Testing Quantum Kernel...")
    
    # Define an encoding function for 2 qubits
    def encoding_fn(data):
        return iqp_encoding(2, data)
        
    kernel = QuantumKernel(encoding_fn)
    
    # Similarity between same point should be 1.0
    x1 = jnp.array([0.1, 0.2, 0.3])
    k_11 = kernel.evaluate(x1, x1)
    print(f"  K(x1, x1) = {k_11:.6f}")
    assert jnp.abs(k_11 - 1.0) < 1e-5
    
    # Similarity between different points
    x2 = jnp.array([0.5, 0.6, 0.7])
    k_12 = kernel.evaluate(x1, x2)
    print(f"  K(x1, x2) = {k_12:.6f}")
    assert 0 <= k_12 <= 1.0
    
    # Kernel matrix
    X = jnp.array([x1, x2])
    matrix = kernel.calculate_matrix(X)
    print(f"  Kernel Matrix:\n{matrix}")
    assert matrix.shape == (2, 2)
    assert jnp.abs(matrix[0, 0] - 1.0) < 1e-5
    assert jnp.abs(matrix[1, 1] - 1.0) < 1e-5
    
    print("[PASS] Quantum Kernel verified.")


if __name__ == "__main__":
    try:
        test_quantum_kernel()
        print("\nQuantum Kernel Verified.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
