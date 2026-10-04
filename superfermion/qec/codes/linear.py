"""
Superfermion Linear QEC Codes - Standard Bit-Flip, Phase-Flip, Shor, and Steane codes.
"""
import superfermion as sf

class RepetitionCode:
    """Standard repetition code for bit or phase flip, any length n>=2.

    Layout: n data qubits [0..n-1] + (n-1) ancillas [n..2n-2]; ancilla k
    measures parity d_k (+) d_{k+1}. Syndrome map (check -> data qubits):
    ``[[k, k+1] for k in range(n-1)]`` — Stim-compatible ordering.
    """
    def __init__(self, n=3, code_type="bit"):
        if n < 2:
            raise ValueError(f"RepetitionCode needs n>=2, got {n}.")
        self.n = n
        self.code_type = code_type

    def syndrome_map(self):
        """Parity-check map: check k touches data qubits [k, k+1]."""
        return [[k, k + 1] for k in range(self.n - 1)]

    def build(self, rounds=1) -> sf.Circuit:
        """Build encoding + syndrome-extraction circuit.

        Args:
            rounds: number of repeated syndrome-extraction rounds
                (default 1). Round r measures into fresh classical bits
                ``r*(n-1)..(r+1)*(n-1)-1``; data qubits are measured once
                at the end (into the trailing bits) for easy decoding.
        """
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}.")
        n = self.n
        n_anc = n - 1
        # data + one ancilla block reused across rounds (reset each round)
        c = sf.Circuit(2 * n - 1, n_anc * rounds + n)
        # Encoding
        for i in range(1, n):
            c.cnot(0, i)

        if self.code_type == "phase":
            for i in range(n):
                c.h(i)

        # Syndrome Measurement (repeated rounds)
        # Parity checks: d_k (+) d_{k+1} into ancilla n+k
        for r in range(rounds):
            for k in range(n - 1):
                c.cnot(k, n + k)
                c.cnot(k + 1, n + k)
            for k in range(n - 1):
                c.measure(n + k, r * n_anc + k)
            if r + 1 < rounds:
                for k in range(n - 1):
                    c.reset(n + k)
        for i in range(n):
            c.measure(i, n_anc * rounds + i)
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
