"""
Quantum Ansatze — Variational circuit templates for QML and optimization.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union, Callable
from superfermion.circuit import Circuit
from superfermion.parameters import param


def hardware_efficient(n_qubits: int, layers: int = 1, entangler: str = "CX") -> Circuit:
    """A standard hardware-efficient ansatz (HEA).
    
    Consists of single-qubit rotations (RY, RZ) followed by entangling gates.
    """
    c = Circuit(n_qubits)
    
    for l in range(layers):
        # Rotation layer
        for i in range(n_qubits):
            c.ry(param(f"l{l}_ry{i}"), i)
            c.rz(param(f"l{l}_rz{i}"), i)
        
        # Entanglement layer (linear chain)
        for i in range(n_qubits - 1):
            if entangler.upper() == "CX":
                c.cx(i, i + 1)
            elif entangler.upper() == "CZ":
                c.cz(i, i + 1)
                
    # Final rotation layer
    for i in range(n_qubits):
        c.ry(param(f"lf_ry{i}"), i)
        
    return c


def strongly_entangling(n_qubits: int, layers: int = 1) -> Circuit:
    """Strongly Entangling Circuit (from arXiv:1804.00633).
    
    Uses RY/RZ rotations followed by a cyclic entanglement pattern.
    """
    c = Circuit(n_qubits)
    
    for l in range(layers):
        # Rotation layer
        for i in range(n_qubits):
            c.ry(param(f"l{l}_ry{i}"), i)
            c.rz(param(f"l{l}_rz{i}"), i)
            c.ry(param(f"l{l}_ry2_{i}"), i)
            
        # Entanglement layer (cyclic Shift)
        # Shift depends on the layer index
        shift = (l % (n_qubits - 1)) + 1
        for i in range(n_qubits):
            target = (i + shift) % n_qubits
            c.cx(i, target)
            
    return c


def two_local(n_qubits: int, rotation_gates: List[str] = ["RY", "RZ"], 
              entanglement_gates: str = "CX",
              reps: int = 3,
              entanglement: str = "full") -> Circuit:
    """A flexible template similar to Qiskit's TwoLocal.
    
    Args:
        n_qubits: Number of qubits.
        rotation_gates: Names of rotation gates in each layer.
        entanglement_gates: Name of entangling gate.
        reps: Number of repetitions.
        entanglement: Entanglement strategy ('full', 'linear', or 'circular').
    """
    c = Circuit(n_qubits)
    
    for r in range(reps + 1):
        # Rotation layer
        for gate_name in rotation_gates:
            for i in range(n_qubits):
                param_name = f"r{r}_{gate_name}_{i}"
                method = getattr(c, gate_name.lower())
                method(param(param_name), i)
        
        # Entanglement layer (skip in last rep)
        if r < reps:
            if entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        getattr(c, entanglement_gates.lower())(i, j)
            elif entanglement == "linear":
                for i in range(n_qubits - 1):
                    getattr(c, entanglement_gates.lower())(i, i+1)
            elif entanglement == "circular":
                for i in range(n_qubits):
                    getattr(c, entanglement_gates.lower())(i, (i + 1) % n_qubits)
                    
    return c


def u_ansatz(n_qubits: int, layers: int = 1) -> Circuit:
    """Universal ansatz template using U gates (theta, phi, lambda)."""
    c = Circuit(n_qubits)
    for l in range(layers):
        for i in range(n_qubits):
            c.u(
                param(f"l{l}_theta_{i}"),
                param(f"l{l}_phi_{i}"),
                param(f"l{l}_lam_{i}"),
                i
            )
        for i in range(n_qubits - 1):
            c.cx(i, i + 1)
    return c

# PascalCase Aliases for documentation consistency
HardwareEfficient = hardware_efficient
StronglyEntangling = strongly_entangling
TwoLocal = two_local
UAnsatz = u_ansatz
