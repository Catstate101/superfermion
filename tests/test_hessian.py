"""
Higher-Order Derivatives — Calculating Hessians with Superfermion and JAX.
"""

from __future__ import annotations

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer
import jax
import jax.numpy as jnp
from flax import linen as nn


def test_hessian_calculation():
    print("Testing Higher-Order Derivatives (Hessians)...")
    
    # 1. Setup a Parameterized Circuit
    # We use two parameters to get a 2x2 Hessian matrix
    theta1 = sf.param("theta1")
    theta2 = sf.param("theta2")
    c = sf.Circuit(1).rx(theta1, 0).ry(theta2, 0)
    
    # 2. Wrap in Flax
    model = QuantumLayer(n_qubits=1, ansatz=c)
    key = jax.random.PRNGKey(0)
    params = model.init(key)
    
    # 3. Define a Scalar Loss Function
    def loss_fn(p):
        # Probability of state '0'
        return model.apply(p)[0]
        
    # 4. Calculate Hessian
    # jax.hessian computes the matrix of second derivatives
    hessian_fn = jax.hessian(loss_fn)
    
    print(f"  Calculating Hessian for params: {params['params']['weights']}")
    h_matrix = hessian_fn(params)
    
    # Extract the nested matrix from the Flax param structure
    # Flax params are dicts, so jax.hessian returns a pytree of matrices
    matrix = h_matrix['params']['weights']['params']['weights']
    
    print("\n  Hessian Matrix (2x2):")
    print(matrix)
    
    assert matrix.shape == (2, 2)
    # The Hessian of cos(theta/2)^2 involves second-order trig terms.
    # It should be non-zero at most points.
    assert jnp.abs(matrix).sum() > 1e-5
    
    print("\n[PASS] Higher-order Jacobians (Hessians) verified!")

if __name__ == "__main__":
    try:
        test_hessian_calculation()
        print("\nAdvanced optimization infrastructure (Second-order) is ready.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
