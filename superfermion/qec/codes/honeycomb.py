"""
Honeycomb Codes — Floquet-based QEC on hexagonal lattices.
"""

from __future__ import annotations
import numpy as np

class HoneycombCode:
    """Implementations of Hastie-Haah Floquet codes."""
    
    def __init__(self, L: int):
        self.L = L # Lattice size
        self.n_data = 3 * L * L
        
    def edge_check(self, step: int):
        """Perform checks for step i mod 3."""
        pass

    def __repr__(self):
        return f"HoneycombCode(L={self.L})"
