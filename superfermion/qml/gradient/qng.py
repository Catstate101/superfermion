"""
Quantum Natural Gradient (QNG) — Optimization using the Fubini-Study metric.

Implements Quantum Fisher Information Matrix (QFIM) and natural gradient descent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any


def calculate_qfim(circuit_fn: Callable, params: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Quantum Fisher Information Matrix (QFIM).
    
    The QFIM G is defined as:
    G_ij = Re( <∂_i psi| ∂_j psi> - <∂_i psi|psi><psi| ∂_j psi> )
    
    Args:
        circuit_fn: A function that takes params and returns a statevector.
        params: Parameters to evaluate at.
        
    Returns:
        The QFIM matrix.
    """
    # Calculate jacobian of the statevector: ∂|psi>/∂theta
    # Since psi is complex, we need to handle it carefully.
    def real_circuit(p):
        sv = circuit_fn(p)
        return jnp.concatenate([jnp.real(sv), jnp.imag(sv)])
        
    # jacfwd or jacrev? For small number of params, jacfwd is fine.
    # But since the output is large (2^n) and params small, jacrev is better?
    # Actually, for statevector (2^n elements), jacfwd is often better 
    # if n_params < 2^n.
    jac = jax.jacfwd(circuit_fn)(params) # Shape: (2^n, n_params)
    
    # Calculate psi
    psi = circuit_fn(params)
    
    # G_ij = Re( jac_H @ jac - (jac_H @ psi) @ (psi_H @ jac) )
    # jac_H is the conjugate transpose of jac
    jac_H = jnp.conj(jac.T)
    
    # First term: <∂_i psi| ∂_j psi>
    term1 = jac_H @ jac
    
    # Second term: <∂_i psi|psi><psi| ∂_j psi>
    # <∂_i psi|psi> is (jac_H @ psi)
    overlap = jac_H @ psi  # Vector of length n_params
    term2 = jnp.outer(overlap, jnp.conj(overlap))
    
    g = jnp.real(term1 - term2)
    return g


def qng_step(loss_fn: Callable, circuit_fn: Callable, params: jnp.ndarray, 
             learning_rate: float = 0.01, regularization: float = 1e-6) -> jnp.ndarray:
    """Perform one step of Quantum Natural Gradient descent.
    
    theta = theta - eta * G^-1 * grad(L)
    """
    # 1. Calculate standard gradient
    grad = jax.grad(loss_fn)(params)
    
    # 2. Calculate QFIM
    g = calculate_qfim(circuit_fn, params)
    
    # 3. Regularize and invert
    # G might be singular, so we add a small identity term
    g_reg = g + regularization * jnp.eye(g.shape[0])
    
    # 4. Update
    # Use solve instead of inv for better numerical stability
    update = jnp.linalg.solve(g_reg, grad)
    
    return params - learning_rate * update
