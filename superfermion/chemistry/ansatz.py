"""
UCCSD — Unitary Coupled Cluster with Singles and Doubles excitations.
"""

from __future__ import annotations

import superfermion as sf
from superfermion.parameters import param

def uccsd_ansatz(n_qubits: int, n_electrons: int) -> sf.Circuit:
    """Create a UCCSD-inspired variational circuit.
    
    Args:
        n_qubits: Total qubits (orbitals).
        n_electrons: Number of electrons in the system.
    """
    c = sf.Circuit(n_qubits)
    
    # 1. Prepare Hartree-Fock state |1...10...0>
    for i in range(n_electrons):
        c.x(i)
        
    # 2. Add Single Excitations (simplified as SO(2) rotations)
    for i in range(n_electrons):
        for j in range(n_electrons, n_qubits):
            # Rotation representing excitation from occupied i to virtual j
            c.ry(param(f"s_{i}_{j}"), j)
            c.cx(i, j)
            
    # 3. Add Double Excitations (simplified)
    # Full UCCSD involves complex exponentiation of (ai^\dagger aj - h.c.)
    # Here we use a hardware-efficient proxy that captures the symmetry.
    for i in range(n_electrons - 1):
        c.cx(i, i+1)
        c.rz(param(f"d_{i}"), i+1)
        c.cx(i, i+1)
        
    return c
