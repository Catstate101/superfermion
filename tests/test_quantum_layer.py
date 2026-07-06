"""
Test script for Superfermion QuantumLayer (Flax integration).
"""

from __future__ import annotations

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer
import jax
import jax.numpy as jnp
from flax import linen as nn
from optax import adam


def test_flax_integration():
    print("Testing QuantumLayer (Flax) Integration...")
    
    # 1. Setup a Parameterized Circuit
    theta = sf.param("theta")
    c = sf.Circuit(1).rx(theta, 0)
    
    # 2. Wrap in Flax
    ql = QuantumLayer(n_qubits=1, ansatz=c)
    key = jax.random.PRNGKey(0)
    
    # 3. Initialize Parameters
    params = ql.init(key)
    print(f"Initialized Parameters: {params}")
    assert 'weights' in params['params']
    
    # 4. Forward Pass
    res = ql.apply(params)
    print(f"Forward pass (probs): {res}")
    assert res.shape == (2,) 
    assert jnp.allclose(jnp.sum(res), 1.0) # Probs must sum to 1

    # 5. Gradient Test (Trainability)
    def loss_fn(p):
        # Minimize probability of state '0' (forcing rotation to state '1')
        return ql.apply(p)[0]
        
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)
    print(f"Calculated Gradients: {grads}")
    
    # Check that grad is not zero
    assert jnp.abs(grads['params']['weights']).sum() > 1e-5
    
    print("[PASS] QuantumLayer is fully differentiable in Flax.")

if __name__ == "__main__":
    try:
        test_flax_integration()
        print("\nPhase 2: Hybrid Quantum-Classical ML officially unlocked.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
