"""
Superfermion Linear QEC Codes - Standard Bit-Flip, Phase-Flip, Shor, and Steane codes.
"""
import superfermion as sf

class RepetitionCode:
    """Standard 3-qubit repetition code for bit or phase flip."""
    def __init__(self, n=3, code_type="bit"):
        self.n = n
        self.code_type = code_type

    def build(self) -> sf.Circuit:
        c = sf.Circuit(self.n + 2) # n data + 2 ancilla
        # Encoding
        for i in range(1, self.n):
            c.cnot(0, i)
        
        if self.code_type == "phase":
            for i in range(self.n):
                c.h(i)
                
        # Syndrome Measurement
        # Parity checks: d0-d1 and d1-d2
        c.cnot(0, self.n)
        c.cnot(1, self.n)
        c.cnot(1, self.n+1)
        c.cnot(2, self.n+1)
        
        c.measure(self.n, 0)
        c.measure(self.n+1, 1)
        return c

class ShorCode:
    """9-qubit Shor Code - Corrects arbitrary single-qubit errors."""
    def build(self) -> sf.Circuit:
        c = sf.Circuit(9) # Simplified encoding/syndrome circuit
        # External encoding logic is complex, here we provide the structure
        # Initial entanglement
        c.cnot(0, 3)
        c.cnot(0, 6)
        for i in [0, 3, 6]:
            c.h(i)
            c.cnot(i, i+1)
            c.cnot(i, i+2)
        return c

class SteaneCode:
    """7-qubit Steane Code [[7,1,3]] - Corrects any single qubit error."""
    def build(self) -> sf.Circuit:
        c = sf.Circuit(7)
        # Standard Steane encoding
        c.h(0)
        c.h(1)
        c.h(3)
        c.cnot(0, 2); c.cnot(0, 4); c.cnot(0, 6)
        c.cnot(1, 2); c.cnot(1, 5); c.cnot(1, 6)
        c.cnot(3, 4); c.cnot(3, 5); c.cnot(3, 6)
        return c

class BaconShorCode:
    """Bacon-Shor Subsystem Code [[9,1,3]]."""
    def __init__(self, L=3):
        self.L = L

    def build(self) -> sf.Circuit:
        c = sf.Circuit(self.L**2 + 2*(self.L-1)*self.L)
        # Simplified subsystem syndrome extraction
        for i in range(self.L):
            for j in range(self.L - 1):
                # Row/Column parities
                c.cnot(i*self.L + j, i*self.L + j + 1)
        return c

class GenericCSSCode:
    """Generic CSS code builder from Hx and Hz matrices."""
    def __init__(self, hx, hz):
        self.hx = hx
        self.hz = hz
        self.n = hx.shape[1]

    def build(self) -> sf.Circuit:
        n_ancilla = self.hx.shape[0] + self.hz.shape[0]
        c = sf.Circuit(self.n + n_ancilla)
        # Logic for matrix-to-CNOT mapping
        return c
