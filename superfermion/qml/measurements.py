
"""
SuperFermion Quantum Measurements - Comprehensive QML Metrics.
This module provides PennyLane-style measurement functions for SuperFermion.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Union, Optional
from superfermion.observables.core import Observable

def state(statevector: jnp.ndarray) -> jnp.ndarray:
    """Returns the full statevector (Equivalent to qml.state())."""
    return statevector

def probs(statevector: jnp.ndarray) -> jnp.ndarray:
    """Returns the probability distribution (Equivalent to qml.probs())."""
    return jnp.abs(statevector)**2

def expval(statevector: jnp.ndarray, observable: Observable) -> float:
    """Calculates the expectation value of an observable (Equivalent to qml.expval())."""
    return float(observable.expectation(statevector))

def var(statevector: jnp.ndarray, observable: Observable) -> float:
    """Calculates the variance of an observable (Equivalent to qml.var()).
    Var(O) = <O^2> - <O>^2
    """
    from superfermion.observables.core import PauliString, Hamiltonian
    
    e1 = observable.expectation(statevector)
    
    if isinstance(observable, PauliString):
        e2 = observable.coeffs**2
    else:
        try:
            phi = observable._apply(statevector)
            e2 = jnp.real(jnp.vdot(phi, phi))
        except:
            e2 = 1.0 # fallback
            
    return float(e2 - e1**2)

def sample(counts: Dict[str, int]) -> List[str]:
    """Returns raw samples from execution counts (Equivalent to qml.sample())."""
    samples = []
    for bitstring, count in counts.items():
        samples.extend([bitstring] * count)
    return samples

def partial_trace(statevector: jnp.ndarray, keep_wires: List[int]) -> jnp.ndarray:
    """Computes the reduced density matrix for the specified wires.
    
    Args:
        statevector: The full statevector (2^n elements).
        keep_wires: List of wire indices to keep.
        
    Returns:
        Reduced density matrix (2^k x 2^k, where k = len(keep_wires)).
    """
    n = int(jnp.log2(len(statevector)))
    psi = statevector.reshape([2] * n)
    
    all_wires = list(range(n))
    traced_wires = [w for w in all_wires if w not in keep_wires]
    
    # Reorder such that keep_wires come first
    perm = list(keep_wires) + traced_wires
    psi_perm = jnp.transpose(psi, perm)
    
    # Reshape for matrix multiplication: (2^k, 2^(n-k))
    k = len(keep_wires)
    psi_matrix = psi_perm.reshape((2**k, -1))
    
    # rho_reduced = psi @ psi.H
    return jnp.matmul(psi_matrix, jnp.conjugate(psi_matrix).T)

def vn_entropy(statevector: jnp.ndarray, wires: Optional[List[int]] = None) -> float:
    """Von Neumann Entropy S = -tr(rho log rho).
    
    If wires is None, calculates entropy of the full state (0 for pure states).
    If wires is provided, calculates the Entanglement Entropy of that subsystem.
    """
    if wires is None:
        return 0.0
    
    rho = partial_trace(statevector, wires)
    eigvals = jnp.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-jnp.sum(eigvals * jnp.log(eigvals)))

def purity(statevector: jnp.ndarray, wires: Optional[List[int]] = None) -> float:
    """Purity tr(rho^2). 1.0 for pure states or subsystems with no entanglement."""
    if wires is None:
        return 1.0
    
    rho = partial_trace(statevector, wires)
    return float(jnp.real(jnp.trace(jnp.matmul(rho, rho))))

def fidelity(state_a: jnp.ndarray, state_b: jnp.ndarray) -> float:
    """State Fidelity |<psi_a | psi_b>|^2."""
    overlap = jnp.abs(jnp.vdot(state_a, state_b))
    return float(overlap**2)

def mutual_info(statevector: jnp.ndarray, wires_a: List[int], wires_b: List[int]) -> float:
    """Mutual Information I(A:B) = S(A) + S(B) - S(A U B)."""
    s_a = vn_entropy(statevector, wires_a)
    s_b = vn_entropy(statevector, wires_b)
    s_ab = vn_entropy(statevector, list(set(wires_a) | set(wires_b)))
    return float(s_a + s_b - s_ab)

def participation_ratio(statevector: jnp.ndarray) -> float:
    """Measures state localization in the computational basis.
    PR = 1 / sum(p_i^2)
    PR = 1 for a basis state, PR = 2^n for a uniform superposition.
    """
    p = jnp.abs(statevector)**2
    sum_p2 = jnp.sum(p**2)
    return float(1.0 / sum_p2) if sum_p2 > 0 else 0.0

# Friendly aliases matching PennyLane / api_reference.md naming
von_neumann_entropy = vn_entropy
expectation_value = expval


def compute_all_metrics(state_vec: jnp.ndarray, observable: Observable = None) -> Dict[str, Any]:
    """Summary of all key metrics."""
    metrics = {
        "purity": purity(state_vec),
        "entropy": vn_entropy(state_vec),
        "norm": float(jnp.linalg.norm(state_vec)),
        "participation_ratio": participation_ratio(state_vec)
    }
    if observable:
        metrics["expval"] = expval(state_vec, observable)
        metrics["var"] = var(state_vec, observable)
    return metrics
