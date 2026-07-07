"""
Observables — measurement operators for expected values.

Hot-path expectation values are computed in Rust via ``_sf_core.hamiltonian_expval``
(MSB-convention statevector, weighted Pauli sum, single FFI call).

The NumPy ``_apply_pauli_string_np`` function is kept as a reference/fallback
and for the ``_apply`` method, but all ``_fast_expval`` paths route through Rust.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_PAULI_ENCODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}

try:
    from superfermion._sf_core import hamiltonian_expval as _rust_hamiltonian_expval
    _HAS_RUST_EXPVAL = True
except ImportError:
    _HAS_RUST_EXPVAL = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _apply_pauli_string_np(sv: np.ndarray, pauli_str: str) -> np.ndarray:
    """Apply a tensor-product Pauli string to a statevector.

    Uses index arithmetic rather than Kronecker products.
    SuperFermion convention: qubit 0 = MSB (leftmost bit in binary index).

    Returns the modified statevector (new array, does not modify sv).
    """
    n = len(pauli_str)
    result = sv.astype(np.complex128, copy=True)
    dim = len(sv)
    indices = np.arange(dim, dtype=np.int64)

    for k, p in enumerate(pauli_str):
        if p == 'I':
            continue
        # SF MSB-first: qubit k corresponds to bit position (n-1-k) in the index
        bit_pos = n - 1 - k
        if p == 'Z':
            bit_vals = (indices >> bit_pos) & 1  # 0 or 1
            result *= np.where(bit_vals == 1, -1, 1)
        elif p == 'X':
            flipped = indices ^ (1 << bit_pos)
            result = result[flipped]
        elif p == 'Y':
            # Y|0⟩ = i|1⟩,  Y|1⟩ = -i|0⟩
            # Y has matrix elements Y[0,1] = -i, Y[1,0] = +i.
            # (Yψ)[i] = sum_j Y[i_bit, j_bit] * ψ[i with bit_pos flipped to j_bit]
            #   i_bit=0: (Yψ)[i] = Y[0,1] * ψ[flipped] = -i * ψ[flipped]
            #   i_bit=1: (Yψ)[i] = Y[1,0] * ψ[flipped] = +i * ψ[flipped]
            # so factors must be: bit=0 -> -i, bit=1 -> +i.
            bit_vals = (indices >> bit_pos) & 1
            factors = np.where(bit_vals == 0, -1j, 1j)
            flipped = indices ^ (1 << bit_pos)
            result = factors * result[flipped]
        else:
            raise ValueError(f"Unknown Pauli character: '{p}'")

    return result




def expval(statevector: np.ndarray, observable) -> float:
    """Compute ⟨ψ|O|ψ⟩ for any SF observable.

    Works with plain numpy arrays (no JAX required), so it is compatible
    with all SF backends (statevector, rust, mps, singularity, etc.).

    Args:
        statevector: Complex state vector of length 2^n.
        observable:  A PauliString, SparsePauliOp, or Hamiltonian.

    Returns:
        Real expectation value (float).
    """
    sv = np.asarray(statevector, dtype=np.complex128).ravel()
    return float(np.real(observable._fast_expval(sv)))


# ── Observable base ────────────────────────────────────────────────────────────

class Observable(ABC):
    """Abstract base class for all quantum observables."""

    @abstractmethod
    def _fast_expval(self, sv: np.ndarray) -> complex:
        """Fast backend-agnostic ⟨ψ|O|ψ⟩ using numpy bit manipulation."""

    def expectation(self, statevector):
        """Compute ⟨ψ|O|ψ⟩ from a numpy statevector."""
        sv = np.asarray(statevector, dtype=np.complex128).ravel()
        return float(np.real(self._fast_expval(sv)))

    @abstractmethod
    def __repr__(self) -> str:
        pass

    # Operator overloading for building Hamiltonians
    def __add__(self, other: "Observable") -> "Hamiltonian":
        terms_self = self.terms if isinstance(self, Hamiltonian) else [self]
        terms_other = other.terms if isinstance(other, Hamiltonian) else [other]
        return Hamiltonian(terms_self + terms_other)

    def __rmul__(self, scalar: float) -> "PauliString":
        if isinstance(self, PauliString):
            return PauliString(self.pauli_str, self.coeffs * scalar)
        if isinstance(self, SparsePauliOp):
            return SparsePauliOp(self._terms)  # handled below
        raise TypeError(f"Cannot multiply {type(self)}")

    def __mul__(self, scalar: float) -> "Observable":
        return self.__rmul__(scalar)


# ── PauliString ────────────────────────────────────────────────────────────────

class PauliString(Observable):
    """A tensor product of Pauli operators, e.g. 'XIZ' with optional coefficient.

    Examples:
        >>> ps = PauliString('ZZ', coeff=0.5)
        >>> ps.expectation(bell_sv)   # returns float
    """

    def __init__(self, pauli_str: str, coeffs: float = 1.0, coeff: Optional[float] = None):
        self.pauli_str = pauli_str.upper()
        # Accept both 'coeffs' and 'coeff' keyword
        self.coeffs = coeff if coeff is not None else coeffs

    def _fast_expval(self, sv: np.ndarray) -> complex:
        if _HAS_RUST_EXPVAL:
            paulis = [_PAULI_ENCODE[ch] for ch in self.pauli_str]
            c = complex(self.coeffs)
            return _rust_hamiltonian_expval(sv, [(paulis, c.real, c.imag)])
        Opsi = _apply_pauli_string_np(sv, self.pauli_str)
        return self.coeffs * np.vdot(sv, Opsi)

    def _apply(self, statevector) -> np.ndarray:
        """Apply this Pauli string operator to a statevector.

        Returns O|psi> as a numpy array (coefficient included).
        """
        sv = np.asarray(statevector, dtype=np.complex128).ravel()
        return self.coeffs * _apply_pauli_string_np(sv, self.pauli_str)

    def __repr__(self) -> str:
        return f"PauliString('{self.pauli_str}', coeff={self.coeffs})"


# ── SparsePauliOp ──────────────────────────────────────────────────────────────

class SparsePauliOp(Observable):
    """Sparse weighted sum of Pauli strings — mirrors Qiskit's SparsePauliOp.

    Two creation styles:
        # Dict style  (most convenient)
        H = SparsePauliOp.from_dict({'ZZ': -1.0, 'XX': 0.5, 'II': 0.25})

        # List style
        H = SparsePauliOp(['ZZ', 'XX'], coeffs=[-1.0, 0.5])

    Notes:
        - Qiskit uses little-endian (qubit 0 = rightmost character).
          SF uses big-endian (qubit 0 = leftmost character).
          When converting from Qiskit's SparsePauliOp, reverse each string:
          ``sf_op = SparsePauliOp.from_qiskit(qk_spo)``
    """

    def __init__(
        self,
        paulis: Sequence[str],
        coeffs: Optional[Sequence[complex]] = None,
    ):
        if coeffs is None:
            coeffs = [1.0] * len(paulis)
        self._terms: List[Tuple[str, complex]] = [
            (p.upper(), complex(c)) for p, c in zip(paulis, coeffs)
        ]

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict[str, complex]) -> "SparsePauliOp":
        paulis = list(d.keys())
        coeffs = list(d.values())
        return cls(paulis, coeffs)

    @classmethod
    def from_qiskit(cls, qk_op) -> "SparsePauliOp":
        """Convert Qiskit SparsePauliOp → SF SparsePauliOp.

        Qiskit uses little-endian (qubit 0 = rightmost); SF uses big-endian.
        Reversal is applied to each Pauli string.
        """
        paulis = [str(p)[::-1] for p in qk_op.paulis]
        coeffs = [complex(c) for c in qk_op.coeffs]
        return cls(paulis, coeffs)

    @classmethod
    def from_pennylane(cls, pl_op) -> "SparsePauliOp":
        """Convert a PennyLane Hamiltonian / Sum / SProd to SF SparsePauliOp.

        PennyLane uses big-endian (wire 0 = leftmost); no reversal needed.
        Accepts ``qml.Hamiltonian``, ``qml.ops.Sum``, or ``qml.ops.SProd``.
        """
        import pennylane as qml
        terms, coeffs = qml.Hamiltonian(*_pl_to_lists(pl_op)).terms()
        paulis = [_pl_term_to_str(t) for t in terms]
        return cls(paulis, coeffs)

    @classmethod
    def from_string(cls, s: str, n_qubits: Optional[int] = None) -> "SparsePauliOp":
        """Parse a human-readable Pauli observable string.

        Supported formats:
            'Z0'       → Z on qubit 0, I elsewhere
            'Z0Z1'     → Z on qubits 0 and 1
            'X0Y1Z2'   → X on 0, Y on 1, Z on 2
            'ZZ'       → Z on qubits 0 and 1 (condensed, length = n_qubits)
            'IXI'      → X on qubit 1 (explicit identity positions)

        Args:
            s:        Observable string.
            n_qubits: Total qubit count.  Inferred from string if omitted.
        """
        import re

        s = s.strip().upper()

        # Format 1: Indexed pairs like "Z0X1Z2"
        matches = re.findall(r'([XYZI])(\d+)', s)
        if matches:
            max_q = max(int(q) for _, q in matches)
            n = n_qubits if n_qubits is not None else max_q + 1
            pauli = ['I'] * n
            for op, q in matches:
                pauli[int(q)] = op
            return cls([''.join(pauli)], [1.0])

        # Format 2: Condensed like 'ZZ', 'IXI', 'XYZX'
        if all(ch in 'IXYZ' for ch in s):
            n = n_qubits if n_qubits is not None else len(s)
            # Pad with I if shorter than n_qubits
            pauli = list(s) + ['I'] * (n - len(s))
            return cls([''.join(pauli[:n])], [1.0])

        raise ValueError(
            f"Cannot parse observable string: '{s}'. "
            f"Use formats like 'Z0', 'Z0Z1', 'XYZX', or 'IXI'."
        )

    # ── Expectation ────────────────────────────────────────────────────────────

    def _fast_expval(self, sv: np.ndarray) -> complex:
        if _HAS_RUST_EXPVAL:
            rust_terms = []
            for pauli_str, coeff in self._terms:
                paulis = [_PAULI_ENCODE[ch] for ch in pauli_str]
                c = complex(coeff)
                rust_terms.append((paulis, c.real, c.imag))
            return _rust_hamiltonian_expval(sv, rust_terms)
        total: complex = 0.0
        for pauli_str, coeff in self._terms:
            if set(pauli_str) == {'I'}:
                total += coeff * float(np.vdot(sv, sv).real)
            else:
                Opsi = _apply_pauli_string_np(sv, pauli_str)
                total += coeff * np.vdot(sv, Opsi)
        return total


    # ── Arithmetic ─────────────────────────────────────────────────────────────

    def __add__(self, other: "SparsePauliOp") -> "SparsePauliOp":
        if isinstance(other, SparsePauliOp):
            return SparsePauliOp(
                [t[0] for t in self._terms + other._terms],
                [t[1] for t in self._terms + other._terms],
            )
        return NotImplemented

    def __mul__(self, scalar) -> "SparsePauliOp":
        return SparsePauliOp(
            [t[0] for t in self._terms],
            [t[1] * scalar for t in self._terms],
        )

    def __rmul__(self, scalar) -> "SparsePauliOp":
        return self.__mul__(scalar)

    @property
    def terms(self) -> List[PauliString]:
        """Compatibility with Hamiltonian.terms access pattern."""
        return [PauliString(p, coeff=c) for p, c in self._terms]

    def __repr__(self) -> str:
        parts = [f"{c:+.4f}*{p}" for p, c in self._terms]
        return "SparsePauliOp(" + " ".join(parts) + ")"


# ── Hamiltonian ────────────────────────────────────────────────────────────────

class Hamiltonian(Observable):
    """A linear combination of PauliStrings (list-based, legacy compatible)."""

    def __init__(self, terms: List[PauliString]):
        self.terms = terms

    def _fast_expval(self, sv: np.ndarray) -> complex:
        if _HAS_RUST_EXPVAL:
            rust_terms = []
            for term in self.terms:
                paulis = [_PAULI_ENCODE[ch] for ch in term.pauli_str]
                c = complex(term.coeffs)
                rust_terms.append((paulis, c.real, c.imag))
            return _rust_hamiltonian_expval(sv, rust_terms)
        total: complex = 0.0
        for term in self.terms:
            total += term._fast_expval(sv)
        return total

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Convert to SparsePauliOp for interoperability."""
        return SparsePauliOp(
            [t.pauli_str for t in self.terms],
            [t.coeffs for t in self.terms],
        )

    def __repr__(self) -> str:
        return f"Hamiltonian(terms={len(self.terms)})"


# ── PennyLane conversion helpers ───────────────────────────────────────────────

def _pl_to_lists(pl_op):
    """Return (coeffs, ops) from a PennyLane Hamiltonian/Sum."""
    try:
        coeffs, ops = pl_op.coeffs, pl_op.ops
        return list(coeffs), ops
    except AttributeError:
        # qml.ops.Sum / qml.ops.SProd (PennyLane ≥ 0.36)
        if hasattr(pl_op, 'operands'):
            cs, os = [], []
            for op in pl_op.operands:
                c, o = _pl_to_lists(op)
                cs.extend(c)
                os.extend(o)
            return cs, os
        # SProd: scalar * op
        if hasattr(pl_op, 'scalar'):
            return [pl_op.scalar], [pl_op.base]
        return [1.0], [pl_op]


def _pl_term_to_str(pl_term) -> str:
    """Convert a single PennyLane Pauli product to a Pauli string."""
    import pennylane as qml
    if isinstance(pl_term, qml.Identity):
        return 'I'
    name = pl_term.name  # 'PauliX', 'PauliZ', etc.
    char = name[-1] if len(name) > 1 else name
    return char
