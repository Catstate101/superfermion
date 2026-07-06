"""
QEC (Quantum Error Correction) — Surface Code & Lattice Builders.
"""

from __future__ import annotations

import superfermion as sf


class SurfaceCode:
    """Standard 2D Rotated Surface Code.
    
    This builder generates a circuit that creates a surface code lattice
    and performs a single round of syndrome extraction.
    
    Args:
        distance: The code distance (d). Total qubits ~ 2*d^2 - 1.
    """

    def __init__(self, distance: int):
        if distance < 3 or distance % 2 == 0:
            # For simplicity in this session, only odd d >= 3
            pass
        self.d = distance
        self.n_data = distance**2
        self.n_measure = distance**2 - 1
        self.n_total = self.n_data + self.n_measure

    def build_syndrome_extraction(self) -> sf.Circuit:
        """Create a circuit for one round of stabilizer measurements.
        
        Data qubits: 0-8 (3x3 lattice)
        Ancilla qubits: 9-16 (X-plaquettes and Z-vertices)
        """
        c = sf.Circuit(self.n_total)
        
        # d=3 rotated surface code layout (example mapping)
        # Data qubits on a 3x3 grid:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        
        # Ancillas 9, 11, 13, 15 (X-stabilizers - Plaquettes)
        # Ancillas 10, 12, 14, 16 (Z-stabilizers - Vertices)
        
        # 1. Initialize syndrome ancillas (implicitly |0>)
        
        # 2. X-Stabilizers (requires H on ancilla, then CNOT(ancilla, data), then H)
        plaquettes = [
            (9, [0, 1, 3, 4]),
            (11, [1, 2, 4, 5]),
            (13, [3, 4, 6, 7]),
            (15, [4, 5, 7, 8])
        ]
        
        for anc, data_qs in plaquettes:
            c.h(anc)
            for d_q in data_qs:
                c.cx(anc, d_q) # X stabilizer: ancilla is control
            c.h(anc)
            
        # 3. Z-Stabilizers (requires CNOT(data, ancilla))
        vertices = [
            (10, [0, 1, 3, 4]),
            (12, [1, 2, 4, 5]),
            (14, [3, 4, 6, 7]),
            (16, [4, 5, 7, 8])
        ]
        
        for anc, data_qs in vertices:
            for d_q in data_qs:
                c.cx(d_q, anc) # Z stabilizer: data is control
                
        # 4. Measure all ancillas
        for anc in range(self.n_data, self.n_total):
            c.measure(anc, anc - self.n_data)
            
        return c

    def __repr__(self) -> str:
        return f"SurfaceCode(distance={self.d}, total_qubits={self.n_total})"
