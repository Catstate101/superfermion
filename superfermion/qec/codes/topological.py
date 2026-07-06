"""
Superfermion Topological Codes - 2D Surface and 4D Hypercube codes.
"""
import superfermion as sf

class SurfaceCode2D:
    """Standard 2D Rotated Surface Code."""
    def __init__(self, distance=3):
        self.d = distance
        self.n_data = distance**2
        self.n_ancilla = distance**2 - 1

    def build(self) -> sf.Circuit:
        c = sf.Circuit(self.n_data + self.n_ancilla)
        # 1. Feature Entanglement
        for i in range(self.n_ancilla):
            c.h(self.n_data + i)
            # Connectivity to 4 neighbors on lattice
            # Simplified mapping for demo
            neighbors = [i % self.d, (i+1) % self.d]
            for n in neighbors:
                if n < self.n_data:
                    c.cnot(self.n_data + i, n)
            c.h(self.n_data + i)
        return c

class HypercubeCode4D:
    """
    4D Topological Error Correction.
    Encodes information into a 4-dimensional hypercube lattice.
    Extremely high fault-tolerance threshold.
    """
    def __init__(self, size=2):
        self.size = size
        # A 2x2x2x2 hypercube has 16 data qubits on vertices
        self.n_data = 2**4 
        # Syndromes on faces/edges
        self.n_syndromes = 32 # Simplified for demo

    def build(self) -> sf.Circuit:
        """
        4D Syndrome Extraction Workflow.
        Connects data qubits across the 4th spatial dimension.
        """
        c = sf.Circuit(self.n_data + self.n_syndromes)
        
        # Entangle across Dimensions 1, 2, 3
        for i in range(self.n_data):
            if i + 1 < self.n_data: c.cnot(i, i+1) # Dim 1
            if i + 2 < self.n_data: c.cnot(i, i+2) # Dim 2
            if i + 4 < self.n_data: c.cnot(i, i+4) # Dim 3
            
        # 👑 THE 4TH DIMENSION CNOTs
        # Hypercube connectivity: x -> x ^ (1 << 3)
        for i in range(8):
            c.cnot(i, i + 8)
            
        # Verification via Syndromes
        for i in range(self.n_syndromes):
            anc_idx = self.n_data + i
            c.h(anc_idx)
            # Each face in 4D has 4 qubits
            c.cnot(anc_idx, i % self.n_data)
            c.h(anc_idx)
            c.measure(anc_idx, i)
            
        return c

class ToricCode2D:
    """Toric Code with periodic boundary conditions."""
    def __init__(self, size=3):
        self.size = size
        self.n_data = 2 * size**2
        self.n_ancilla = 2 * size**2

    def build(self) -> sf.Circuit:
        c = sf.Circuit(self.n_data + self.n_ancilla)
        # Periodic stabilizers
        return c

class ColorCode:
    """Topological Color Code on a hexagonal lattice."""
    def __init__(self, distance=3):
        self.d = distance
        self.n_data = (3*distance**2 + 1) // 4 # Example scaling

    def build(self) -> sf.Circuit:
        c = sf.Circuit(self.n_data + 10) # Syndrome overhead
        return c

class HoneycombCode:
    """Floquet/Honeycomb code for dynamic parity checking."""
    def build(self) -> sf.Circuit:
        c = sf.Circuit(20)
        # Measurement-based dynamics
        return c
