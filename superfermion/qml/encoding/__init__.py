"""
Quantum Data Encoding — Map classical data to quantum states.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union
import jax.numpy as jnp
from superfermion.circuit import Circuit


def angle_encoding(n_qubits: int, data: jnp.ndarray, rotation: str = "RY") -> Circuit:
    """Map data to single-qubit rotations.
    
    Args:
        n_qubits: Number of qubits.
        data: Classical data vector of length n_qubits.
        rotation: Rotation gate to use ("RX", "RY", or "RZ").
        
    Returns:
        Circuit with encoding applied.
    """
    if len(data) > n_qubits:
        raise ValueError(f"Data length {len(data)} exceeds n_qubits {n_qubits}")
    
    c = Circuit(n_qubits)
    gate_fn = getattr(c, rotation.lower())
    
    for i, val in enumerate(data):
        gate_fn(val, i)
    
    return c


def basis_encoding(n_qubits: int, value: int) -> Circuit:
    """Map an integer to its binary basis state |bin(value)>.
    
    Example: value=3 (binary 11) -> |11>.
    """
    c = Circuit(n_qubits)
    binary = format(value, f'0{n_qubits}b')
    for i, bit in enumerate(binary):
        if bit == '1':
            c.x(i)
    return c


def amplitude_encoding(n_qubits: int, data: jnp.ndarray) -> Circuit:
    """Encode a normalized vector into the amplitudes of a quantum state.
    
    Requires data length == 2^n_qubits.
    Currently uses an idealized 'state_prep' instruction or manual decomposition.
    For MVP, we implement the state-vector initialization logic.
    """
    dim = 2**n_qubits
    if len(data) != dim:
        # Pad with zeros
        padded = jnp.zeros(dim)
        padded = padded.at[:len(data)].set(data)
        data = padded
        
    norm = jnp.linalg.norm(data)
    if norm > 1e-10:
        data = data / norm
        
    c = Circuit(n_qubits)
    # The actual implementation of amplitude encoding (Möttönen algorithm) 
    # is complex. For now we use the Circuit's capability to store initial states 
    # if supported, or provide a high-level gate.
    # We'll tag this circuit as 'amplitude_encoded'
    c._metadata["initial_state"] = data
    return c


def iqp_encoding(n_qubits: int, data: jnp.ndarray, reps: int = 1) -> Circuit:
    """Instantaneous Quantum Polynomial (IQP) encoding.
    
    Hadamards followed by RZ(x_i) and CZ(x_i * x_j).
    Commonly used in Quantum Kernels.
    """
    c = Circuit(n_qubits)
    
    for _ in range(reps):
        # Initial Hadamard layer
        for i in range(n_qubits):
            c.h(i)
        
        # Linear terms
        for i in range(min(n_qubits, len(data))):
            c.rz(data[i], i)
            
        # Quadratic terms (interactions)
        idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if idx < len(data):
                    # We use a simplified interaction: RZZ gate
                    # RZZ(phi) = e^{-i * phi/2 * Z \otimes Z}
                    # Can be implemented as CX, RZ, CX
                    phi = data[i] * data[j]
                    c.cx(i, j).rz(phi, j).cx(i, j)
                idx += 1
                
    return c
