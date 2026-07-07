"""
Pauli Observables — Helper functions for creating common operators.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from superfermion.observables.core import PauliString, Hamiltonian

def _pauli_on_qubit(p_char: str, qubit_idx: int, n_qubits: Optional[int] = None) -> PauliString:
    """Internal helper to create a Pauli op on a specific qubit."""
    # If n_qubits is not provided, we just give the character and assume 
    # the consumer knows how to handle it or we'll expand it later.
    # However, PauliString currently expects a full lattice string.
    # For now, we'll return a PauliString that represents just the target.
    # The Backend/Simulator will need to pad it with Identites if needed.
    
    # Actually, let's make it easy: if n_qubits is None, we return 
    # a PauliString with just that one char, but that might break kron.
    
    # In a real World Class SDK, we'd have a 'LatticeObservable' class.
    # For now, we'll use a string of 'I's if n_qubits is known.
    if n_qubits:
        s = ['I'] * n_qubits
        s[qubit_idx] = p_char
        return PauliString("".join(s))
    else:
        # Minimalist approach for MVP
        return PauliString(p_char)

def Z(qubit_idx: int, n_qubits: Optional[int] = None) -> PauliString:
    return _pauli_on_qubit('Z', qubit_idx, n_qubits)

def X(qubit_idx: int, n_qubits: Optional[int] = None) -> PauliString:
    return _pauli_on_qubit('X', qubit_idx, n_qubits)

def Y(qubit_idx: int, n_qubits: Optional[int] = None) -> PauliString:
    return _pauli_on_qubit('Y', qubit_idx, n_qubits)

def I(qubit_idx: int, n_qubits: Optional[int] = None) -> PauliString:
    return _pauli_on_qubit('I', qubit_idx, n_qubits)
