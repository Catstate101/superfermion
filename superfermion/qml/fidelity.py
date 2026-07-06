"""
Fidelity Tracking — Real-time monitoring of quantum state convergence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def state_fidelity(state_a: jnp.ndarray, state_b: jnp.ndarray) -> jax.Array:
    """Calculate the squared overlap (fidelity) between two statevectors.
    
    F(ψ, φ) = |⟨ψ|φ⟩|^2
    
    Args:
        state_a: Complex JAX array representing state |ψ⟩.
        state_b: Complex JAX array representing state |φ⟩.
        
    Returns:
        Scalar JAX float (fidelity) between 0.0 and 1.0.
    """
    # ⟨ψ|φ⟩
    overlap = jnp.vdot(state_a, state_b)
    
    # |⟨ψ|φ⟩|^2
    return jnp.abs(overlap) ** 2
