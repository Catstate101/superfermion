"""
UCCSD-inspired variational ansatz for quantum chemistry.

For the minimal 2-qubit H2 case, uses the standard single-parameter
ansatz from O'Malley et al. (2016). For larger systems, builds a
hardware-efficient approximation to the full UCC excitation structure.
"""

from __future__ import annotations

import superfermion as sf
from superfermion.parameters import param


def uccsd_ansatz(n_qubits: int, n_electrons: int) -> sf.Circuit:
    """Create a UCCSD-inspired variational circuit.

    Args:
        n_qubits: Total qubits (spin-orbitals).
        n_electrons: Number of electrons in the system.
    """
    if n_qubits == 2 and n_electrons == 2:
        return _h2_minimal_ansatz()

    c = sf.Circuit(n_qubits)

    for i in range(n_electrons):
        c.x(i)

    for i in range(n_electrons):
        for j in range(n_electrons, n_qubits):
            c.ry(param(f"s_{i}_{j}"), j)
            c.cx(i, j)

    for i in range(n_electrons - 1):
        c.cx(i, i + 1)
        c.rz(param(f"d_{i}"), i + 1)
        c.cx(i, i + 1)

    return c


def _h2_minimal_ansatz() -> sf.Circuit:
    """Standard 2-qubit H2 ansatz (O'Malley et al. 2016).

    The ground state lives in the {|01>, |10>} subspace.  This circuit
    prepares the HF reference |10> (one occupied spin-orbital on q0)
    then applies a Y-rotation on q1 followed by CNOT(q1, q0) to span
    cos(t/2)|10> + sin(t/2)|01>.
    """
    c = sf.Circuit(2)
    c.x(0)
    c.ry(param("theta"), 1)
    c.cx(1, 0)
    return c
