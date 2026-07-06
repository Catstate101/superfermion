"""
Stochastic Reconfiguration (SR) — High-order optimization for variational circuits.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple

def calculate_qfim(ansatz_vmap: Callable, params: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Quantum Fisher Information Matrix (QFIM).
    
    Args:
        ansatz_vmap: A function that returns the statevector for given params.
        params: Circuit parameters.
        
    Returns:
        QFIM (metric tensor).
    """
    # For complex functions, we can calculate gradients of real and imag parts separately
    def real_part(p): return jnp.real(ansatz_vmap(p))
    def imag_part(p): return jnp.imag(ansatz_vmap(p))
    
    jac_real = jax.jacobian(real_part)(params)
    jac_imag = jax.jacobian(imag_part)(params)
    
    # Combined complex Jacobian: dy/dp = d_real/dp + i * d_imag/dp
    jac = jac_real + 1j * jac_imag
    
    # 2. QFIM_{ij} = Re(<di psi| dj psi> - <di psi|psi><psi|dj psi>)
    psi = ansatz_vmap(params)
    
    # <di psi| dj psi>
    term1 = jnp.real(jnp.conj(jac.T) @ jac)
    
    # <di psi|psi>
    overlap = jnp.conj(jac.T) @ psi
    term2 = jnp.real(overlap[:, None] @ jnp.conj(overlap[None, :]))
    
    return term1 - term2

def sr_update(params: jnp.ndarray, grad: jnp.ndarray, qfim: jnp.ndarray, 
              learning_rate: float, regularization: float = 1e-3) -> jnp.ndarray:
    """Stochastic Reconfiguration parameter update.
    
    Args:
        params: Current parameters.
        grad: Energy gradient.
        qfim: QFIM.
        learning_rate: LR.
        regularization: Tikhonov regularization.
        
    Returns:
        Updated parameters.
    """
    # Inverse of (QFIM + reg*I) @ grad
    identity = jnp.eye(qfim.shape[0])
    metric_inv = jnp.linalg.pinv(qfim + regularization * identity)
    
    update = jnp.dot(metric_inv, grad)
    return params - learning_rate * update
