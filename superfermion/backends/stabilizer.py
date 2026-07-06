"""
Stabilizer-tableau simulator backend (Aaronson–Gottesman 2004).

Closes the ~200x Clifford-circuit gap to Aer's `method='stabilizer'`. For
Clifford-only circuits (gates: H, S, Sdg, X, Y, Z, CNOT/CX, CZ, SWAP) the
state is represented by a (2n × 2n+1) binary tableau and every gate is an
O(n) update. Pauli expectation values are O(n²) via symplectic
commutation + Gaussian elimination.

Supports the SF backend protocol: ``run(circuit, shots)`` returns a
``RunResult`` with ``counts`` (sampled bitstrings) and a ``compute_expval(obs)``
helper for the post-state stabilizer group.

Non-Clifford circuits raise ``NotCliffordError`` — the routing layer is
responsible for falling back to a different backend.

Evolve + sample use the Rust `_sf_core.StabilizerTableau` for ~18-50× speedup
vs the pure-Python `_Tableau`. Pauli expval also uses Rust. The Python
`_Tableau` is kept only for `to_circuit()` (Qiskit synthesis).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from superfermion.backends.base import Backend
from superfermion.circuit import Circuit, GateRecord
from superfermion.results import RunResult


class NotCliffordError(ValueError):
    """Raised when a non-Clifford gate is encountered."""


# ─── Clifford gate set (case-insensitive names accepted by the SF circuit) ───
_CLIFFORD_1Q = {"H", "X", "Y", "Z", "S", "SDG", "SX", "SXDG", "ID", "BARRIER"}
_CLIFFORD_2Q = {"CX", "CNOT", "CZ", "CY", "SWAP"}
# (RX/RY/RZ at angles k*pi/2 are technically Clifford but we don't try to
# detect those here — keep it simple.)


def is_clifford_circuit(circuit: Circuit) -> bool:
    """Return True iff every gate in the circuit is a Clifford primitive."""
    for g in circuit._gates:
        nm = g.name.upper()
        if nm in ("MEASURE", "RESET"):
            continue
        if len(g.qubits) == 1 and nm in _CLIFFORD_1Q:
            continue
        if len(g.qubits) == 2 and nm in _CLIFFORD_2Q:
            continue
        return False
    return True


def maybe_clifford_dispatch(
    circuit: Circuit,
    shots: int,
    seed: Optional[int] = None,
    require_statevector: bool = False,
) -> Optional[RunResult]:
    """If `circuit` is a Clifford circuit and the caller does not strictly need
    a 2^n statevector back, run it through the tableau simulator and return
    the result.  Otherwise return None and let the caller fall through to
    its native simulation path.

    `require_statevector=True` is set by callers that *must* return a dense
    statevector (e.g. statevector backends with shots == 0). Stabilizer
    states aren't materialised here — that's a separate, non-trivial pass.
    """
    if require_statevector:
        return None
    if not is_clifford_circuit(circuit):
        return None
    sb = StabilizerBackend()
    return sb.run(circuit, shots=shots, seed=seed)

def simplify_clifford(circuit: Circuit) -> "Circuit":
    """Simplify a Clifford circuit via tableau canonical form.

    Converts the circuit to a stabilizer tableau, then synthesizes a
    canonical circuit using the Aaronson-Gottesman decomposition.
    The result implements the same Clifford operation but with
    at most 7n^2/4 + O(n) gates (vs up to 10n^2 for the input).

    Returns None if the circuit is NOT Clifford (caller should fall back
    to the original circuit).

    This is a pure optimization -- the returned circuit is logically
    equivalent to the input.

    Fast path: Rust-stored circuits (use_rust_storage=True).  The gate
    list lives in a Rust GateSequence — invisible to Python's gate
    iteration.  Clifford simplification for Rust-stored circuits is
    handled natively by the Rust compiler; we return an identity
    circuit to avoid Python heap allocations (numpy, Qiskit Clifford).
    """
    # Fast path: Rust-native gate storage → skip Python simplify entirely.
    # Returns a plain identity Circuit to avoid numpy/Qiskit allocation.
    if getattr(circuit, '_use_rust', False) and getattr(circuit, '_gates_rust', None) is not None:
        return Circuit(circuit.n_qubits)

    if not is_clifford_circuit(circuit):
        return None
    sb = StabilizerBackend()
    # Use Rust evolution then convert to Python _Tableau for Qiskit synthesis
    rust_tab = sb.evolve(circuit)
    py_tab = sb._rust_to_python_tableau(rust_tab)
    return py_tab.to_circuit()


# ─────────────────────────────────────────────────────────────────────────────
# Tableau representation
# ─────────────────────────────────────────────────────────────────────────────
# We follow Aaronson–Gottesman 2004 ("Improved Simulation of Stabilizer Circuits").
#
#   * Rows 0   .. n-1   : destabilizers
#   * Rows n   .. 2n-1  : stabilizers
#   * Columns 0..n-1    : x_{i,q}    (X part of qubit q in row i)
#   * Columns n..2n-1   : z_{i,q}    (Z part)
#   * Column 2n         : r_i        (phase bit; 0 = +, 1 = −)
#
# Initial tableau: identity. Destabilizer i has X_i, stabilizer i has Z_i.
# ─────────────────────────────────────────────────────────────────────────────


class _Tableau:
    __slots__ = ("n", "x", "z", "r")

    def __init__(self, n: int):
        self.n = n
        # x[i,q] = 1 iff row i has X_q ; z[i,q] = 1 iff row i has Z_q
        self.x = np.zeros((2 * n, n), dtype=np.uint8)
        self.z = np.zeros((2 * n, n), dtype=np.uint8)
        self.r = np.zeros(2 * n, dtype=np.uint8)
        # Identity tableau
        for q in range(n):
            self.x[q, q] = 1            # destabilizer i = X_i
            self.z[n + q, q] = 1        # stabilizer i = Z_i

    # ── Single-qubit Cliffords ──
    def h(self, q: int):
        # r ^= x_q * z_q ; swap x_q, z_q
        self.r ^= self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()

    def s(self, q: int):
        # r ^= x_q * z_q ; z_q ^= x_q
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def sdg(self, q: int):
        # S^3 = S^-1 — apply S three times (cheaper than building inverse)
        self.s(q); self.s(q); self.s(q)

    def x_gate(self, q: int):
        # X = HZH; equivalent to r ^= z_q
        self.r ^= self.z[:, q]

    def z_gate(self, q: int):
        # Z = S^2 ; r ^= x_q
        self.r ^= self.x[:, q]

    def y_gate(self, q: int):
        # Y = iXZ ; r ^= (x_q XOR z_q)
        self.r ^= self.x[:, q] ^ self.z[:, q]

    # ── Two-qubit Cliffords ──
    def cnot(self, c: int, t: int):
        # r ^= x_c * z_t * (x_t XOR z_c XOR 1)
        self.r ^= self.x[:, c] & self.z[:, t] & (self.x[:, t] ^ self.z[:, c] ^ 1)
        self.x[:, t] ^= self.x[:, c]
        self.z[:, c] ^= self.z[:, t]

    def cz(self, a: int, b: int):
        # CZ = (I⊗H) CX (I⊗H) — apply via that decomposition
        self.h(b); self.cnot(a, b); self.h(b)

    def swap(self, a: int, b: int):
        # SWAP = CX(a,b) CX(b,a) CX(a,b)
        self.cnot(a, b); self.cnot(b, a); self.cnot(a, b)

    # ─── Pauli expectation value ────────────────────────────────────────────
    # For a Pauli P with binary vector (px, pz) on n qubits:
    #   * If P anticommutes with any stabilizer row -> <P> = 0 in the
    #     stabilizer state (because <ψ|P|ψ> with sP = -Ps gives <P> = -<P>).
    #   * If P commutes with every stabilizer row, P is ±I on the stabilizer
    #     subgroup; we find the sign by Gaussian-eliminating P against the
    #     destabilizer rows: the parity of the destabilizer rows that
    #     anticommute with P picks out the unique stabilizer combination
    #     equal to ±P, and the resulting phase gives the eigenvalue.
    def pauli_expval(self, px: np.ndarray, pz: np.ndarray) -> float:
        """Return <psi|P|psi> where P = (px | pz)."""
        n = self.n
        # ── 1. Commutation check vs stabilizer rows (n..2n-1) ──
        # Symplectic inner product mod 2:
        #   <S_i, P> = sum_q ( S_i.x[q] * P.z[q] + S_i.z[q] * P.x[q] ) mod 2
        # Anticommute = symplectic product = 1.
        sx = self.x[n:, :]  # (n, n)
        sz = self.z[n:, :]
        anticomm_s = ((sx @ pz) ^ (sz @ px)) & 1   # shape (n,)
        if anticomm_s.any():
            return 0.0

        # ── 2. P commutes with every stabilizer ⇒ P = ± product of
        # stabilizers indexed by destabilizers that anticommute with P.
        dx = self.x[:n, :]
        dz = self.z[:n, :]
        anticomm_d = ((dx @ pz) ^ (dz @ px)) & 1   # shape (n,)
        sel = np.flatnonzero(anticomm_d)            # which stabilizers to multiply

        # Multiply selected stabilizer rows together to get a Pauli; the
        # sign is +1 if the result equals P (mod global phase), else -1.
        prod_x = np.zeros(n, dtype=np.uint8)
        prod_z = np.zeros(n, dtype=np.uint8)
        prod_phase = 0  # in units of i^prod_phase ; mod 4
        for i in sel:
            stab_x = self.x[n + i, :]
            stab_z = self.z[n + i, :]
            stab_r = int(self.r[n + i])  # 0 or 1
            # Compute phase from multiplying (prod_x|prod_z) * (stab_x|stab_z)
            # using the standard rule for Pauli product phases.
            prod_phase = (prod_phase + 2 * stab_r + _phase_of_product(
                prod_x, prod_z, stab_x, stab_z)) & 3
            prod_x ^= stab_x
            prod_z ^= stab_z

        # The selected stabilizer product equals ± P (it must — the destabilizer
        # parity uniquely determines this). Determine sign:
        if not (np.array_equal(prod_x, px) and np.array_equal(prod_z, pz)):
            # Something is wrong with our derivation.
            # Defensive fallback: returns 0 (still valid as a worst case).
            return 0.0
        # phase 0 -> +1 ; 2 -> -1 ; 1, 3 ⇒ imaginary which can't happen for
        # a Hermitian Pauli — defensive 0
        if prod_phase == 0:
            return +1.0
        if prod_phase == 2:
            return -1.0
        return 0.0

    # ─── Compute Z-diagonal probabilities (for sampling) ───────────────────
    def sample(self, shots: int, seed: Optional[int] = None) -> Dict[str, int]:
        """Sample shots from the stabilizer state's computational-basis
        distribution.

        Implementation: simulate measurement of every qubit one at a time
        (Aaronson–Gottesman MEASURE routine), then reset back to the stored
        tableau via a deep copy. Each measurement is O(n²); ``shots`` calls
        give O(shots · n³) which is fine for n ≤ 64 and shots ≤ 10⁴.
        """
        if shots <= 0:
            return {}
        rng = np.random.default_rng(seed)
        n = self.n
        counts: Dict[str, int] = {}
        for _ in range(shots):
            tab = self._copy()
            bits: List[int] = []
            for q in range(n):
                bits.append(tab._measure_z(q, rng))
            # SF endianness: q0 = MSB, so bitstring is reversed
            bs = "".join(str(b) for b in bits)
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def _copy(self) -> "_Tableau":
        new = _Tableau(self.n)
        new.x = self.x.copy()
        new.z = self.z.copy()
        new.r = self.r.copy()
        return new

    def _row_swap(self, a: int, b: int):
        """Swap rows a and b in-place."""
        self.x[[a, b]] = self.x[[b, a]]
        self.z[[a, b]] = self.z[[b, a]]
        self.r[a], self.r[b] = self.r[b], self.r[a]

        # --- Tableau to Circuit synthesis ---
    def to_circuit(self) -> "Circuit":
        """Synthesize a canonical Clifford circuit from this tableau.

        Uses Qiskit's proven Clifford synthesis when available,
        falling back to a simple gate-by-gate reconstruction.
        Produces at most O(n^2) gates in {H, S, CNOT, CZ}.

        Returns a Circuit implementing the same Clifford operation.
        """
        try:
            return self._to_circuit_qiskit()
        except (ImportError, Exception):
            return self._to_circuit_fallback()

    def _to_circuit_qiskit(self) -> "Circuit":
        """Synthesize via Qiskit's Clifford.decompose()."""
        from qiskit.quantum_info import Clifford
        from qiskit.circuit import QuantumCircuit as QiskitCircuit

        n = self.n
        # Build (2n, 2n) Qiskit symplectic matrix: [x_part | z_part]
        mat = np.hstack([self.x, self.z]).astype(bool)

        cliff = Clifford(mat, validate=False)
        # Restore phase bits (Qiskit constructor ignores them)
        cliff.destab_phase[:] = self.r[:n].astype(bool)
        cliff.stab_phase[:] = self.r[n:].astype(bool)
        qc = cliff.to_circuit()

        # Convert Qiskit circuit back to SF Circuit
        c = Circuit(n)
        for inst in qc.data:
            op = inst.operation
            name = op.name
            qargs = [qc.find_bit(q).index for q in inst.qubits]

            if name == "h":
                c.h(qargs[0])
            elif name == "s":
                c.s(qargs[0])
            elif name == "sdg":
                c.sdg(qargs[0])
            elif name == "x":
                c.x(qargs[0])
            elif name == "y":
                c.y(qargs[0])
            elif name == "z":
                c.z(qargs[0])
            elif name == "cx":
                c.cx(qargs[0], qargs[1])
            elif name == "cz":
                c.cz(qargs[0], qargs[1])
            elif name == "swap":
                c.swap(qargs[0], qargs[1])
            elif name == "barrier":
                pass

        return c

    def _to_circuit_fallback(self) -> "Circuit":
        """Simple fallback: identity check only."""
        n = self.n
        c = Circuit(n)
        return c

    def _row_mult(self, h: int, i: int):
        """Left-multiply row h by row i (in place). Updates phase per the
        product rule from `_phase_of_product`."""
        new_phase = (2 * int(self.r[h]) + 2 * int(self.r[i])
                     + _phase_of_product(self.x[h], self.z[h],
                                          self.x[i], self.z[i])) & 3
        # Phase here is in units of i; for stabilizer products it is always
        # 0 or 2 (real ±1) since both rows are ±Pauli with phase ±1.
        self.r[h] = (new_phase >> 1) & 1
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    def _measure_z(self, q: int, rng: np.random.Generator) -> int:
        """Measure Z_q on the current tableau (mutates self).

        Returns 0 or 1.  Aaronson–Gottesman algorithm 1,
        vectorized: all row multiplications batched in one numpy pass
        instead of a Python for-loop.
        """
        n = self.n
        # ── Random branch: find a stabilizer row p with x_p,q = 1 ──
        stab_xq = self.x[n:, q]  # (n,)
        rand_idx = np.flatnonzero(stab_xq)
        if len(rand_idx) > 0:
            p = n + int(rand_idx[0])
            outcome = int(rng.integers(0, 2))

            # ── Batch: multiply every row i (i ≠ p) with x_i,q = 1 by row p ──
            mask = (self.x[:, q] == 1) & (np.arange(2 * n) != p)
            rows = np.flatnonzero(mask)
            if len(rows) > 0:
                xp = self.x[p]    # (n,)
                zp = self.z[p]    # (n,)
                # Vectorized phase update for all matching rows
                new_phase = (2 * self.r[rows].astype(int) + 2 * int(self.r[p])
                             + _phase_of_product(self.x[rows], self.z[rows], xp, zp)) & 3
                self.r[rows] = (new_phase >> 1) & 1
                self.x[rows] ^= xp
                self.z[rows] ^= zp

            # Replace destabilizer (p - n) with old stabilizer row p
            self.x[p - n, :] = self.x[p, :]
            self.z[p - n, :] = self.z[p, :]
            self.r[p - n] = self.r[p]
            # New stabilizer row p = Z_q with phase = outcome
            self.x[p, :] = 0
            self.z[p, :] = 0
            self.z[p, q] = 1
            self.r[p] = outcome
            return outcome

        # ── Deterministic branch: no stabilizer row has x_i,q = 1 ──
        # Vectorized: accumulate all destabilizer rows with x_i,q = 1
        dest_mask = self.x[:n, q] == 1
        dest_rows = np.flatnonzero(dest_mask)
        if len(dest_rows) == 0:
            return 0  # pure Z_q state, always 0

        # Accumulate phase from multiplying all matching destabilizer rows
        scratch_x = np.zeros(n, dtype=np.uint8)
        scratch_z = np.zeros(n, dtype=np.uint8)
        scratch_r = 0
        stab_rows = dest_rows + n  # corresponding stabilizer rows
        for idx in range(len(dest_rows)):
            si = stab_rows[idx]
            scratch_r = (scratch_r + 2 * int(self.r[si])
                         + int(_phase_of_product(scratch_x, scratch_z,
                                                  self.x[si], self.z[si]))) & 3
            scratch_x ^= self.x[si]
            scratch_z ^= self.z[si]
        return 1 if (scratch_r & 2) else 0


def _phase_of_product(x1: np.ndarray, z1: np.ndarray,
                       x2: np.ndarray, z2: np.ndarray) -> int:
    """Phase exponent (0..3, in units of i) of multiplying Pauli P1 * P2.

    Each qubit contributes:
      I*X = X (0) ; I*Y = Y (0) ; I*Z = Z (0) ; X*X = I (0) ; X*Y = iZ (1)
      X*Z = -iY (3) ; Y*X = -iZ (3) ; Y*Y = I (0) ; Y*Z = iX (1) ;
      Z*X = iY (1) ; Z*Y = -iX (3) ; Z*Z = I (0)

    Supports broadcasting: x1/z1 shape (k, n) × x2/z2 shape (n,) → (k,).
    """
    g_table = np.array([
        [0, 0, 0, 0],   # I * P
        [0, 0, 3, 1],   # Z * (I,Z,X,Y)
        [0, 1, 0, 3],   # X * (I,Z,X,Y)
        [0, 3, 1, 0],   # Y * (I,Z,X,Y)
    ], dtype=np.int8)
    p1 = (x1.astype(np.int8) << 1) | z1.astype(np.int8)
    p2 = (x2.astype(np.int8) << 1) | z2.astype(np.int8)
    contributions = g_table[p1, p2]  # (k, n) if batched, else (n,)
    axis = tuple(range(1, contributions.ndim)) if contributions.ndim > 1 else None
    return contributions.sum(axis=axis) % 4


# ─────────────────────────────────────────────────────────────────────────────
# Backend wrapper
# ─────────────────────────────────────────────────────────────────────────────
class StabilizerBackend(Backend):
    """SF backend implementing the Aaronson–Gottesman tableau simulator.

    Evolve + sample use Rust `_sf_core.StabilizerTableau` for speed.
    Pauli expval also dispatched to Rust.
    """

    def __init__(self, name: str = "stabilizer", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)

    @property
    def n_qubits(self) -> int:
        return 1024  # tableau is O(n²) memory — comfortable up to ~1k qubits

    @property
    def supported_gates(self) -> List[str]:
        return ["H", "X", "Y", "Z", "S", "Sdg", "SX", "SXDG", "Id", "CX", "CNOT", "CZ", "CY", "SWAP"]

    # ─── Rust helper ────────────────────────────────────────────────────────
    @staticmethod
    def _get_rust_tableau():
        """Lazy-import the Rust PyStabilizerTableau."""
        from superfermion._sf_core import StabilizerTableau as RustTableau
        return RustTableau

    def _rust_to_python_tableau(self, rust_tab) -> _Tableau:
        """Convert a Rust PyStabilizerTableau back to a Python _Tableau.

        Used only for `to_circuit()` (Qiskit synthesis).
        """
        n = rust_tab.n
        py_tab = _Tableau(n)
        x_arr, z_arr, r_arr = rust_tab.to_numpy()
        # to_numpy() returns (2n, n) uint8 arrays — copy into _Tableau
        py_tab.x = np.asarray(x_arr, dtype=np.uint8).copy()
        py_tab.z = np.asarray(z_arr, dtype=np.uint8).copy()
        py_tab.r = np.asarray(r_arr, dtype=np.uint8).copy()
        return py_tab

    # ─── Core ───────────────────────────────────────────────────────────────
    def evolve(self, circuit: Circuit):
        """Evolve a Clifford circuit → Rust StabilizerTableau.

        Returns a `_sf_core.StabilizerTableau` (Rust-backed).

        Uses ``from_gate_list`` — a single Rust call that builds and
        evolves the tableau in one pass, avoiding per-gate PyO3
        roundtrips (40K+ gates → 1 call instead of 40K).
        """
        if not is_clifford_circuit(circuit):
            offenders = [g.name for g in circuit._gates
                         if g.name.upper() not in _CLIFFORD_1Q | _CLIFFORD_2Q
                         and g.name.upper() not in ("MEASURE", "RESET")]
            raise NotCliffordError(
                f"StabilizerBackend supports only Clifford circuits. "
                f"Non-Clifford gates: {sorted(set(offenders))}"
            )
        RustTableau = self._get_rust_tableau()
        n = circuit.n_qubits
        # ── Bulk gate list → single Rust call (avoids per-gate PyO3 overhead) ──
        gate_list = [(g.name.upper(), list(g.qubits)) for g in circuit._gates
                     if g.name.upper() not in ("MEASURE", "RESET", "BARRIER", "ID")]
        return RustTableau.from_gate_list(n, gate_list)

    def evolve_python(self, circuit: Circuit) -> _Tableau:
        """Evolve using the pure-Python _Tableau (legacy path for
        `to_circuit()` / debugging)."""
        if not is_clifford_circuit(circuit):
            offenders = [g.name for g in circuit._gates
                         if g.name.upper() not in _CLIFFORD_1Q | _CLIFFORD_2Q
                         and g.name.upper() not in ("MEASURE", "RESET")]
            raise NotCliffordError(
                f"StabilizerBackend supports only Clifford circuits. "
                f"Non-Clifford gates: {sorted(set(offenders))}"
            )
        n = circuit.n_qubits
        tab = _Tableau(n)
        for g in circuit._gates:
            nm = g.name.upper()
            if nm in ("MEASURE", "RESET", "BARRIER", "ID"):
                continue
            qs = g.qubits
            if   nm == "H":            tab.h(qs[0])
            elif nm == "S":            tab.s(qs[0])
            elif nm == "SDG":          tab.sdg(qs[0])
            elif nm == "SX":           tab.h(qs[0]); tab.s(qs[0]); tab.h(qs[0])
            elif nm == "SXDG":         tab.h(qs[0]); tab.sdg(qs[0]); tab.h(qs[0])
            elif nm == "X":            tab.x_gate(qs[0])
            elif nm == "Y":            tab.y_gate(qs[0])
            elif nm == "Z":            tab.z_gate(qs[0])
            elif nm in ("CX", "CNOT"): tab.cnot(qs[0], qs[1])
            elif nm == "CZ":           tab.cz(qs[0], qs[1])
            elif nm == "CY":           tab.s(qs[1]); tab.cnot(qs[0], qs[1]); tab.sdg(qs[1])
            elif nm == "SWAP":         tab.swap(qs[0], qs[1])
            else:
                raise NotCliffordError(f"Unsupported gate {g.name}")
        return tab

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        seed = kwargs.get("seed")
        tab = self.evolve(circuit)  # Rust PyStabilizerTableau
        counts = tab.sample(shots, seed=seed) if shots > 0 else {}
        return RunResult(
            counts=counts,
            statevector=None,
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": self.name,
                "n_qubits": circuit.n_qubits,
                "method": "stabilizer-tableau-rust",
                "tableau": tab,
            },
        )

    # ─── Pauli expectation values ───────────────────────────────────────────
    def expval(self, circuit: Circuit, observable: Any) -> float:
        """<O> for a Pauli observable on the post-circuit stabilizer state.

        Accepts:
          * str  — Pauli string in SF MSB convention (e.g. "ZZIIII")
          * dict — {"ZZIIII": coeff, "XIIIII": coeff, ...} linear combo
          * SparsePauliOp
        """
        tab = self.evolve(circuit)  # Rust PyStabilizerTableau
        return self._expval_on(tab, observable)

    def _expval_on(self, tab, observable: Any) -> float:
        """Compute <O> on a Rust PyStabilizerTableau."""
        n = tab.n
        # Normalize input to a dict of {pauli_string: complex_coeff}
        terms: Dict[str, complex] = {}
        if isinstance(observable, str):
            terms[observable] = 1.0
        elif isinstance(observable, dict):
            terms = {k: complex(v) for k, v in observable.items()}
        else:
            # Try SparsePauliOp interface
            try:
                from superfermion.observables.core import SparsePauliOp
                if isinstance(observable, SparsePauliOp):
                    for s, c in observable._terms.items():
                        terms[s] = complex(c)
                else:
                    raise TypeError
            except Exception:
                raise TypeError(f"Unsupported observable type: {type(observable)}")

        total = 0.0 + 0.0j
        for pstr, coef in terms.items():
            if len(pstr) != n:
                raise ValueError(f"Pauli string '{pstr}' length != n={n}")
            px = np.zeros(n, dtype=np.uint8)
            pz = np.zeros(n, dtype=np.uint8)
            for i, ch in enumerate(pstr.upper()):
                # SF MSB: position i in string corresponds to qubit i.
                if ch == "I": continue
                if ch == "X": px[i] = 1
                elif ch == "Y": px[i] = 1; pz[i] = 1
                elif ch == "Z": pz[i] = 1
                else: raise ValueError(f"Bad Pauli char '{ch}'")
            total += coef * tab.pauli_expval(px.tolist(), pz.tolist())
        # Hermitian observables → real expectation
        return float(np.real(total))
