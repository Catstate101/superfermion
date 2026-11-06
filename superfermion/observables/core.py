"""
Observables — measurement operators for expected values.

Hot-path expectation values are computed in Rust via ``_sf_core.hamiltonian_expval``
(little-endian statevector, weighted Pauli sum, single FFI call).

The NumPy ``_apply_pauli_string_np`` function is kept as a reference/fallback
and for the ``_apply`` method, but all ``_fast_expval`` paths route through Rust.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from superfermion._limits import validate_bond_dim

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
    SuperFermion convention: qubit q lives at bit position q (little-endian),
    so qubit 0 is the least-significant bit of the binary index.

    Returns the modified statevector (new array, does not modify sv).
    """
    result = sv.astype(np.complex128, copy=True)
    dim = len(sv)
    indices = np.arange(dim, dtype=np.int64)

    for k, p in enumerate(pauli_str):
        if p == 'I':
            continue
        # SF little-endian: qubit k corresponds to bit position k in the index
        bit_pos = k
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


class _TruncationReport:
    """Minimal shim so ``_check_mps_truncation`` can be reused verbatim.

    That helper expects an object with a ``truncation_report()`` method, which
    is what ``run(method="mps").state`` provides. ``mps_expval`` gets the
    report as a plain dict from a one-shot Rust call, so it wraps it here
    rather than duplicating the threshold logic, the wording and the
    ``SF_MPS_WARN_WEIGHT`` / ``SF_MPS_STRICT`` environment handling.
    """

    __slots__ = ("_report",)

    def __init__(self, report):
        self._report = report

    def truncation_report(self):
        return self._report


def mps_expval(
    circuit,
    observable,
    bond_dim: int = 64,
    params=None,
    warn_truncation: bool = True,
) -> float:
    """Compute ⟨ψ|O|ψ⟩ directly on the MPS tensor network.

    Evolves the circuit once (bond_dim cap) and evaluates every Pauli term
    in a single batched Rust pass — the state is never densified into 2^n
    amplitudes, so this stays cheap where ``run(method="mps").state.numpy()``
    would OOM (e.g. GHZ-30 in milliseconds).

    Args:
        circuit: ``sf.Circuit`` (parameters must be bound, or pass ``params``).
        observable: str, dict, SparsePauliOp, Hamiltonian-like, PauliString,
            or Rust term list ``[(paulis, coef_re, coef_im)]``. In a Pauli
            **string** the leftmost character is qubit 0, i.e. position ``k``
            carries weight ``2**k`` — qubit 0 is the *least* significant bit.
            This is the same order ``expval()`` and ``RunResult.expectation``
            use, and it is the **opposite** of a counts key, where the
            leftmost character is qubit ``n-1``.
        bond_dim: max MPS bond dimension (default 64).
        params: optional dict of symbolic-parameter bindings.
        warn_truncation: emit the same ``RuntimeWarning`` (and honour the
            same ``SF_MPS_WARN_WEIGHT`` / ``SF_MPS_STRICT`` environment
            switches) that ``run(method="mps")`` raises, so a value computed
            from a heavily truncated state is not reported as if it were
            exact. Set ``False`` to silence it.

    Returns:
        Real expectation value (float).

    Raises:
        RuntimeError: if truncation occurs and ``SF_MPS_STRICT=1``.
    """
    from superfermion.qml.gradient.adjoint import _observable_to_rust_terms

    bond_dim = validate_bond_dim(bond_dim)
    if params is not None:
        circuit = circuit.bind(params)
    n = circuit.n_qubits
    if isinstance(observable, (list, tuple)):
        terms = list(observable)
    else:
        terms = _observable_to_rust_terms(observable, n)
    paulis = []
    coefs = []
    for t in terms:
        plist = [int(x) for x in t[0]]
        if len(plist) != n:
            raise ValueError(
                f"Pauli term length {len(plist)} != n_qubits {n}. "
                "Use one character per qubit, e.g. 'ZIII' (leftmost char = qubit 0)."
            )
        paulis.append(plist)
        coefs.append(complex(t[1], t[2]))
    ir = circuit.to_ir()
    get_report = getattr(ir, "simulate_mps_pauli_expval_batch_report", None)
    if warn_truncation and get_report is not None:
        vals, report = get_report(bond_dim, paulis)
        from superfermion.devices.rust_device import _check_mps_truncation

        _check_mps_truncation(_TruncationReport(report), bond_dim)
    else:
        vals = ir.simulate_mps_pauli_expval_batch(bond_dim, paulis)
    total = sum((c * complex(v[0], v[1]) for c, v in zip(coefs, vals)), start=0j)
    return float(total.real)


def _numba_site_step():
    """Compile (once, cached by caller) the parallel sweep kernel.

    Returns None if numba is unavailable.
    """
    try:
        import numba as _nb
    except ImportError:
        return None

    @_nb.njit(parallel=True, fastmath=True)
    def _step(M, T, OUT, p, dl, dr):
        # Row layout: new bit s at position idx -> row s*p + i (NOT i*2+s,
        # which would append at bit 0 and bit-reverse the output; GHZ-symmetric
        # states cannot catch that bug, asymmetric ones can).
        for i in _nb.prange(p):
            for s in range(2):
                for r in range(dr):
                    acc = 0j
                    toff = s * dl
                    for l in range(dl):
                        acc += M[i, l] * T[toff + l, r]
                    OUT[s * p + i, r] = acc

    return _step


_NUMBA_SITE_STEP = None
_NUMBA_PROBED = False


def _get_numba_site_step():
    """Module-cached kernel getter (compiles on first use only)."""
    global _NUMBA_SITE_STEP, _NUMBA_PROBED
    if not _NUMBA_PROBED:
        _NUMBA_SITE_STEP = _numba_site_step()
        _NUMBA_PROBED = True
    return _NUMBA_SITE_STEP


def mps_todense(circuit, bond_dim: int = 64, params=None) -> "np.ndarray":
    """Densify an MPS evolution to a full 2^n statevector via a numba-parallel sweep.

    Evolves with ``bond_dim`` cap, then contracts left-to-right with one
    numba-parallel kernel per site (no per-amplitude allocation, SIMD inner
    loops). Same little-endian convention as everywhere (qubit q at bit q).

    Typically ~2x faster than the Rust single-threaded densify path on
    large states (GHZ-24: ~0.5s vs ~1.5s); trails threaded-BLAS tensor
    libraries (~0.3s) on tall-skinny shapes.

    Requires ``numba`` (a quimb dependency — present wherever MPS work
    happens); falls back to ``sf.run(method="mps").state.numpy()`` without it.

    Args:
        circuit: ``sf.Circuit`` (parameters must be bound, or pass ``params``).
        bond_dim: max MPS bond dimension (default 64).
        params: optional dict of symbolic-parameter bindings.

    Returns:
        Complex statevector as numpy array.

    ``bond_dim`` is SF's per-bond cap and is not numerically comparable to
    another framework's cap of the same name (equal caps differ in accuracy);
    compare fidelity-vs-cap curves instead.
    """
    if not hasattr(circuit, "to_ir"):
        raise TypeError(
            f"mps_todense expects an sf.Circuit; got {type(circuit).__name__}. "
            "To densify an IR, use ir.evolve_mps(bond_dim).to_statevector()."
        )
    import numpy as _np

    bond_dim = validate_bond_dim(bond_dim)
    if params is not None:
        circuit = circuit.bind(params)
    _step = _get_numba_site_step()
    if _step is None:
        return _np.asarray(
            __import__("superfermion", fromlist=["run"]).run(
                circuit, method="mps", shots=0, bond_dim=bond_dim
            ).state.numpy()
        )
    tensors = [
        _np.ascontiguousarray(t)
        for t in circuit.to_ir().evolve_mps(bond_dim).tensors()
    ]

    M = _np.ones((1, 1), dtype=_np.complex128)
    for t in tensors:
        dl = t.shape[0] // 2
        dr = t.shape[1]
        p = M.shape[0]
        OUT = _np.empty((2 * p, dr), dtype=_np.complex128)
        _step(M, t, OUT, p, dl, dr)
        M = OUT
    return M[:, 0].copy()


def _apply_observable(sv: np.ndarray, observable) -> np.ndarray:
    """Return O|ψ⟩ as a numpy array for PauliString / SparsePauliOp / Hamiltonian."""
    if isinstance(observable, PauliString):
        return observable._apply(sv)
    if isinstance(observable, SparsePauliOp):
        out = np.zeros_like(sv)
        for pauli_str, coeff in observable._terms:
            out += PauliString(pauli_str, coeffs=complex(coeff))._apply(sv)
        return out
    if isinstance(observable, Hamiltonian):
        out = np.zeros_like(sv)
        for term in observable.terms:
            out += term._apply(sv)
        return out
    raise TypeError(
        f"Unsupported observable type: {type(observable).__name__}. "
        "Expected PauliString, SparsePauliOp, or Hamiltonian."
    )


def variance(statevector: np.ndarray, observable) -> float:
    """Compute Var(O) = ⟨O²⟩ − ⟨O⟩² for any SF observable.

    For Hermitian observables (real coefficients) the result is exact:
    ⟨O²⟩ = ‖O|ψ⟩‖² and ⟨O⟩ = Re⟨Oψ|ψ⟩ are both read off a single
    application O|ψ⟩. Works with plain numpy arrays (no JAX required),
    compatible with all SF backends. PennyLane analogue: ``qml.var``.

    Args:
        statevector: Complex state vector of length 2^n.
        observable:  A PauliString, SparsePauliOp, or Hamiltonian.

    Returns:
        Real variance (float).
    """
    sv = np.asarray(statevector, dtype=np.complex128).ravel()
    opsi = _apply_observable(sv, observable)
    mean = float(np.vdot(opsi, sv).real)
    mean_sq = float(np.vdot(opsi, opsi).real)
    return mean_sq - mean * mean


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


# ── State.expectation observable compatibility ────────────────────────────────
#
# ``State`` is a PyO3 class whose ``expectation`` only accepted the raw Rust
# term list ``[(paulis, coef_re, coef_im)]``. Every Python-side observable form
# raised a PyO3 conversion error instead -- ``Can't extract `str` to `Vec```
# for a plain string, ``'dict' object is not an instance of 'Sequence'`` for a
# dict, a PauliString or a SparsePauliOp (CH-6), and the messages said nothing
# about the observable (CH-11). The wrapper below coerces the observable first
# and validates the shape, so a bad term list names the offending qubit count
# instead of leaking a Rust extraction message.

_STATE_EXPECTATION_ORIGINAL = None


def _normalise_rust_terms(terms, n_qubits: int, source: str):
    """Validate ``[(paulis, coef_re, coef_im)]`` and return the canonical form."""
    if isinstance(terms, (str, bytes)) or not isinstance(terms, (list, tuple)):
        raise TypeError(
            f"{source}: expected a Pauli observable (str, dict, PauliString, "
            f"SparsePauliOp, Hamiltonian) or a Rust term list "
            f"[(paulis, coef_re, coef_im)], got {type(terms).__name__}."
        )
    out = []
    for index, term in enumerate(terms):
        if isinstance(term, (str, bytes)) or not isinstance(term, (list, tuple)):
            raise TypeError(
                f"{source}: term {index} must be (paulis, coef_re, coef_im), "
                f"got {type(term).__name__}. Did you pass Pauli letters without "
                f"converting them? Use observables.core's helper or a Pauli "
                f"string such as 'ZIII'."
            )
        if len(term) < 3:
            raise TypeError(
                f"{source}: term {index} needs 3 entries "
                f"(paulis, coef_re, coef_im), got {len(term)}."
            )
        paulis, coef_re, coef_im = term[0], term[1], term[2]
        pauli_list = [int(p) for p in paulis]
        if n_qubits and len(pauli_list) != n_qubits:
            raise ValueError(
                f"{source}: term {index} covers {len(pauli_list)} qubits but the "
                f"state has {n_qubits}. Use one entry per qubit, e.g. 'ZIII' "
                f"for a 4-qubit state (leftmost character = qubit 0)."
            )
        out.append((pauli_list, float(np.real(coef_re)), float(np.real(coef_im))))
    return out


def _state_expectation(self, observable):
    """``State.expectation`` that accepts every documented observable form."""
    n_qubits = getattr(self, "n_qubits", 0)
    if isinstance(observable, list) and observable and isinstance(observable[0], tuple):
        # Already a Rust term list. Keep the original call, but reject a term
        # whose length disagrees with the state first: the Rust side silently
        # returned 0.0 for a short term and raised "MethodError invalid state
        # dimension or Pauli string" for a long one.
        for index, term in enumerate(observable):
            if len(term) >= 1 and isinstance(term[0], (list, tuple)) and n_qubits:
                if len(term[0]) != n_qubits:
                    raise ValueError(
                        f"State.expectation: term {index} covers {len(term[0])} "
                        f"qubits but the state has {n_qubits}."
                    )
        return _STATE_EXPECTATION_ORIGINAL(self, observable)
    from superfermion.qml.gradient.adjoint import _observable_to_rust_terms

    terms = _observable_to_rust_terms(observable, n_qubits)
    terms = _normalise_rust_terms(terms, n_qubits, "State.expectation")
    try:
        return _STATE_EXPECTATION_ORIGINAL(self, terms)
    except TypeError as exc:  # pragma: no cover - defensive, should not trigger
        raise TypeError(
            f"State.expectation: could not evaluate {type(observable).__name__} "
            f"({exc}). The underlying Rust call expects "
            f"[(paulis, coef_re, coef_im)] with one entry per qubit."
        ) from exc


def install_state_expectation_compat() -> bool:
    """Patch ``State.expectation`` to share ``RunResult.expectation``'s API.

    Returns True when the patch is installed (idempotent). PyO3 classes allow
    attribute assignment in this build; if a future build forbids it the
    function returns False and the raw behaviour is left untouched.
    """
    global _STATE_EXPECTATION_ORIGINAL
    try:
        from superfermion._sf_core import State
    except ImportError:
        return False
    if _STATE_EXPECTATION_ORIGINAL is not None:
        return True
    original = getattr(State, "expectation", None)
    if original is None:
        return False
    try:
        State.expectation = _state_expectation
    except (AttributeError, TypeError):
        return False
    _STATE_EXPECTATION_ORIGINAL = original
    return True


STATE_EXPECTATION_COMPAT = install_state_expectation_compat()
