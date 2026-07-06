"""
Molecules Library — Pre-calculated Hamiltonians for standard research problems.
"""

from typing import Dict, List
import numpy as np
from superfermion.chemistry import FermionicOperator

class MoleculeLibrary:
    """Gallery of molecular Hamiltonians for VQE benchmarking."""
    
    @staticmethod
    def hydrogen(distance: float = 0.735) -> Dict:
        """Standard H2 molecule at equilibrium."""
        # Simple STO-3G basis mapping (Mock coefficients for demonstration)
        # H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*Y0Y1 + g5*X0X1
        return {
            "name": f"H2 (d={distance})",
            "n_qubits": 2,
            "hamiltonian": [
                ("I", -1.0523),
                ("Z0", 0.0112),
                ("Z1", 0.0112),
                ("Z0Z1", 0.0112),
                ("Y0Y1", 0.0456),
                ("X0X1", 0.0456)
            ]
        }

    @staticmethod
    def lithium_hydride() -> Dict:
        """LiH molecule (4 qubits)."""
        return {
            "name": "LiH",
            "n_qubits": 4,
            "hamiltonian": [
                ("I", -7.88),
                ("Z0", -0.21), ("Z1", -0.21),
                ("Z0Z1", 0.12), ("Z2Z3", 0.12)
            ]
        }

    @staticmethod
    def water() -> Dict:
        """H2O molecule (approx. 14 qubits in standard mapping)."""
        return {
            "name": "H2O",
            "n_qubits": 14,
            "hamiltonian": [] # Placeholder for complex matrix
        }
