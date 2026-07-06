"""
Ansätze — Variational quantum circuit templates.
"""

from __future__ import annotations

import superfermion as sf
import numpy as np


def hardware_efficient_ansatz(
    n_qubits: int, 
    layers: int = 1, 
    entanglement: str = "linear"
) -> sf.Circuit:
    """A hardware-efficient ansatz for variational algorithms.
    
    Structure:
    1. Initial layer: RY rotation on each qubit.
    2. Entanglement layer: CNOT between adjacent qubits.
    3. Rotation layer: RY rotation on each qubit. (Repeat L times)
    
    Args:
        n_qubits: Number of qubits.
        layers: Number of entangling layers.
        entanglement: Connectivity ('linear' or 'full').
        
    Returns:
        sf.Circuit with named parameters 'θ_{layer}_{qubit}'.
    """
    c = sf.Circuit(n_qubits)
    
    for l in range(layers + 1):
        # 1. Rotation Layer
        for i in range(n_qubits):
            p_name = f"theta_{l}_{i}"
            c.ry(sf.param(p_name), i)
            
        # 2. Entanglement Layer (skip after last rotation)
        if l < layers:
            if entanglement == "linear":
                for i in range(n_qubits - 1):
                    c.cx(i, i + 1)
            elif entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        c.cx(i, j)
                        
    return c
