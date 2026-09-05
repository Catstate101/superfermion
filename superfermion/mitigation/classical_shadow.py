"""Classical shadows: randomized Pauli-basis snapshots for expectation values.

Implements the classical-shadow protocol (Huang-Kueng-Preskill) on top of
an exact statevector:

1. ``classical_shadow(state, shots, seed)`` draws ``shots`` snapshots.
   In each snapshot every qubit is measured in a uniformly random Pauli
   basis (X, Y, or Z); a snapshot is stored as an outcome ``bits`` row and
   a ``recipes`` row holding the basis that was measured on each qubit.

2. ``shadow_expval`` (or ``ClassicalShadow.expval``) estimates ``<O>``
   from the snapshots with the standard unbiased estimator: a snapshot
   contributes ``3**|S|`` times the product of the measured eigenvalues
   when it measured *all* of ``O``'s support qubits in the matching Pauli
   basis, and contributes 0 otherwise.  The mean over snapshots is the
   estimate (median of ``k`` chunk means for ``k > 1``).

Conventions:
- Pauli codes follow ``observables.core._PAULI_ENCODE``: I=0, X=1, Y=2, Z=3.
- ``recipes`` codes: 1 = measured in X basis, 2 = Y basis, 3 = Z basis.
- A snapshot outcome bit 0 maps to eigenvalue +1, bit 1 to eigenvalue -1
  of the measured basis operator.
- Observable inputs: ``PauliString`` / ``SparsePauliOp`` / ``Hamiltonian``,
  or the raw term format used by ``State.expectation``: a list of
  ``(paulis, real, imag)`` tuples where ``paulis`` is a per-qubit code
  list (or a Pauli string).  Labels are q0-leftmost ("ZI" = Z on qubit 0);
  statevectors are little-endian (qubit q lives at bit q).
- Pure-state (statevector-representable) states only.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from superfermion.observables.core import _PAULI_ENCODE

__all__ = ["ClassicalShadow", "classical_shadow", "shadow_expval"]

# Single-qubit rotations mapping the target Pauli basis to Z.
#   U X U^dag = Z with U = H;  U Y U^dag = Z with U = H * S^dag.
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2)
_UY = np.array([[1.0, -1j], [1.0, 1j]], dtype=np.complex128) / np.sqrt(2)
_I2 = np.eye(2, dtype=np.complex128)
_ROT = {1: _H, 2: _UY, 3: _I2}

# ── helpers ────────────────────────────────────────────────────────────────


def _statevector_of(state) -> np.ndarray:
    """Extract an exact statevector from a Circuit, State, or ndarray."""
    if hasattr(state, "_gates"):
        # Circuit: run the exact CPU statevector backend (no shots).
        import superfermion as sf

        result = sf.run(state, device="cpu", method="statevector", shots=0)
        return np.asarray(result.statevector, dtype=np.complex128).ravel()
    if hasattr(state, "statevector") and not isinstance(state, np.ndarray):
        return np.asarray(state.statevector, dtype=np.complex128).ravel()
    if hasattr(state, "numpy") and not isinstance(state, np.ndarray):
        # Rust State (or duck-typed state) with a numpy() accessor.
        return np.asarray(state.numpy(), dtype=np.complex128).ravel()
    return np.asarray(state, dtype=np.complex128).ravel()


def _validate_shots(shots) -> int:
    if not isinstance(shots, (int, np.integer)):
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if int(shots) < 1:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    return int(shots)


def _rotate(sv: np.ndarray, n: int, recipe_row) -> np.ndarray:
    """Rotate ``sv`` so a computational-basis measurement implements the
    Pauli-basis measurement described by ``recipe_row`` (codes 1/2/3)."""
    arr = sv.reshape((2,) * n)
    for q in range(n):
        m = _ROT[int(recipe_row[q])]
        # Statevectors are little-endian (wire q at bit q), so in the
        # C-order ravel the wire q lives on tensor axis n-1-q, not axis q
        # (axis q carries bit n-1-q = wire n-1-q).  SUP-25.
        axis = n - 1 - q
        arr = np.tensordot(m, arr, axes=([1], [axis]))
        arr = np.moveaxis(arr, 0, axis)
    return arr.ravel()


def _sample_snapshots(sv: np.ndarray, shots: int, seed) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``shots`` classical-shadow snapshots of the exact statevector.

    Returns ``(bits, recipes)`` with shape (shots, n_qubits): ``bits`` is
    the outcome (0/1, little-endian per qubit) and ``recipes`` holds the
    basis code (1=X, 2=Y, 3=Z) that was measured on each qubit.
    """
    dim = sv.size
    n = int(round(np.log2(dim)))
    if 2 ** n != dim or n < 1:
        raise ValueError(
            f"statevector must be a pure state of 2**n amplitudes, got length {dim}"
        )
    rng = np.random.default_rng(seed)
    recipes = rng.integers(1, 4, size=(shots, n)).astype(np.int8)
    # Group snapshots by identical basis choice: at most 3**n groups, so we
    # rotate + sample once per group instead of once per snapshot.
    uniq, inverse, counts = np.unique(
        recipes, axis=0, return_inverse=True, return_counts=True
    )
    bits = np.zeros((shots, n), dtype=np.uint8)
    for row_i in range(uniq.shape[0]):
        rotated = _rotate(sv, n, uniq[row_i])
        probs = np.abs(rotated) ** 2
        probs = probs / probs.sum()
        idx = rng.choice(dim, size=int(counts[row_i]), p=probs)
        snapshots = np.flatnonzero(inverse == row_i)
        for q in range(n):
            bits[snapshots, q] = (idx >> q) & 1
    return bits, recipes


def _normalize_term(paulis, n: int) -> List[int]:
    """Normalize one term's pauli specification to per-qubit codes of
    length ``n`` (q0-leftmost, trailing identities padded)."""
    if isinstance(paulis, str):
        codes = [_PAULI_ENCODE[ch] for ch in paulis.upper()]
    else:
        codes = [int(x) for x in paulis]
    if len(codes) > n:
        raise ValueError(
            f"pauli term {paulis!r} acts on more qubits than the shadow has ({n})"
        )
    if any(c < 0 or c > 3 for c in codes):
        raise ValueError(f"invalid pauli codes in term {paulis!r}")
    return codes + [0] * (n - len(codes))


def _iter_terms(observable, n: int):
    """Yield ``(codes, coef)`` for every term of any SF observable input."""
    if isinstance(observable, str):
        # Parse indexed ('Z0', 'X0Z1') and condensed ('ZZ', 'IXI') labels.
        from superfermion.observables.core import SparsePauliOp

        op = SparsePauliOp.from_string(observable, n_qubits=n)
        for label, coef in op._terms:
            yield _normalize_term(label, n), complex(coef)
        return
    if hasattr(observable, "_terms"):
        # SparsePauliOp: list of (label, complex coef)
        for label, coef in observable._terms:
            yield _normalize_term(label, n), complex(coef)
        return
    if hasattr(observable, "pauli_str"):
        # PauliString
        yield _normalize_term(observable.pauli_str, n), complex(observable.coeffs)
        return
    if hasattr(observable, "terms"):
        # Hamiltonian: list of PauliString
        for term in observable.terms:
            yield _normalize_term(term.pauli_str, n), complex(term.coeffs)
        return
    # Raw term format: either a list of terms, each (paulis, real, imag) or
    # (paulis, coef) like State.expectation, or a single bare term tuple.
    items = [observable] if isinstance(observable, tuple) else list(observable)
    for item in items:
        if isinstance(item, str):
            from superfermion.observables.core import SparsePauliOp

            op = SparsePauliOp.from_string(item, n_qubits=n)
            for label, coef in op._terms:
                yield _normalize_term(label, n), complex(coef)
            continue
        head = item[0]
        codes = _normalize_term(head, n)
        if len(item) >= 3:
            coef = complex(float(item[1]), float(item[2]))
        elif len(item) == 2:
            coef = complex(item[1])
        else:
            coef = 1.0 + 0.0j
        yield codes, coef


# ── public surface ─────────────────────────────────────────────────────────


class ClassicalShadow:
    """A classical shadow: random single-qubit Pauli-basis snapshots.

    Args:
        bits: outcome array, shape (n_snapshots, n_qubits), values 0/1.
        recipes: basis array, same shape; 1 = X, 2 = Y, 3 = Z.

    Examples:
        >>> shadow = ClassicalShadow.from_statevector(sv, shots=1000, seed=7)
        >>> est = shadow.expval(SparsePauliOp.from_string("Z0", n_qubits=2))
    """

    def __init__(self, bits, recipes):
        bits = np.asarray(bits)
        recipes = np.asarray(recipes)
        if bits.ndim != 2 or recipes.ndim != 2 or bits.shape != recipes.shape:
            raise ValueError(
                "bits and recipes must be 2-D arrays of the same shape "
                "(n_snapshots, n_qubits)"
            )
        if bits.size and not np.all(np.isin(bits, [0, 1])):
            raise ValueError("bits must only contain 0/1 outcomes")
        if bits.size and not np.all(np.isin(recipes, [1, 2, 3])):
            raise ValueError("recipes must only contain basis codes 1 (X), 2 (Y), 3 (Z)")
        self._bits = np.ascontiguousarray(bits, dtype=np.uint8)
        self._recipes = np.ascontiguousarray(recipes, dtype=np.int8)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def bits(self) -> np.ndarray:
        """Snapshot outcomes, shape (n_snapshots, n_qubits)."""
        return self._bits

    @property
    def recipes(self) -> np.ndarray:
        """Snapshot basis choices, shape (n_snapshots, n_qubits)."""
        return self._recipes

    @property
    def n_qubits(self) -> int:
        """Number of qubits in each snapshot."""
        return self._bits.shape[1]

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots (shots) in the shadow."""
        return self._bits.shape[0]

    # ── constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_statevector(cls, statevector, shots: int, seed=None) -> "ClassicalShadow":
        """Sample snapshots from an exact statevector.

        Args:
            statevector: complex statevector of length 2**n (or any object
                exposing a statevector, e.g. a ``State``).
            shots: number of snapshots to draw.
            seed: RNG seed for reproducible snapshots.
        """
        sv = _statevector_of(statevector)
        shots = _validate_shots(shots)
        bits, recipes = _sample_snapshots(sv, shots, seed)
        return cls(bits, recipes)

    @classmethod
    def from_circuit(cls, circuit, shots: int, seed=None) -> "ClassicalShadow":
        """Sample snapshots from the exact statevector of a Circuit."""
        shots = _validate_shots(shots)
        sv = _statevector_of(circuit)
        bits, recipes = _sample_snapshots(sv, shots, seed)
        return cls(bits, recipes)

    # ── estimation ──────────────────────────────────────────────────────────

    def expval(self, observable, k: int = 1) -> float:
        """Estimate ``<observable>`` from the snapshots.

        Args:
            observable: PauliString / SparsePauliOp / Hamiltonian / raw term
                iterable ``(paulis, real, imag)``.
            k: number of chunks for the median-of-means aggregation.
                k=1 (default) uses the plain mean over all snapshots.

        Returns:
            Real estimate of the expectation value.
        """
        if not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        k = int(k)
        if k > self.n_snapshots:
            raise ValueError(
                f"k={k} exceeds the number of snapshots ({self.n_snapshots})"
            )
        bits, recipes = self._bits, self._recipes
        total = np.zeros(self.n_snapshots, dtype=np.complex128)
        for codes, coef in _iter_terms(observable, self.n_qubits):
            codes = np.asarray(codes, dtype=np.int64)
            support = np.flatnonzero(codes != 0)
            if support.size == 0:
                total += coef  # identity term: contributes exactly <I> = 1
                continue
            matched = np.all(recipes[:, support] == codes[support], axis=1)
            eigenvalues = np.where(bits[:, support] == 0, 1.0, -1.0).prod(axis=1)
            total += np.where(
                matched, coef * (3.0 ** support.size) * eigenvalues, 0.0
            )
        estimates = np.real(total)
        if k == 1:
            return float(np.mean(estimates))
        chunks = np.array_split(estimates, k)
        return float(np.median([np.mean(c) for c in chunks]))


def classical_shadow(state, shots: int, seed=None) -> ClassicalShadow:
    """Draw a classical shadow of the exact state of ``state``.

    Args:
        state: Circuit, State, or numpy statevector (pure states only).
        shots: number of snapshots (randomized X/Y/Z measurements).
        seed: RNG seed for reproducible snapshots.

    Returns:
        A ClassicalShadow of ``state`` with ``shots`` snapshots.
    """
    shots = _validate_shots(shots)
    sv = _statevector_of(state)
    bits, recipes = _sample_snapshots(sv, shots, seed)
    return ClassicalShadow(bits, recipes)


def shadow_expval(state, observable, shots: int = 1000, seed=None, k: int = 1) -> float:
    """Estimate ``<observable>`` of ``state`` via a classical shadow.

    Args:
        state: Circuit, State, or numpy statevector (pure states only).
        observable: PauliString / SparsePauliOp / Hamiltonian / raw terms.
        shots: number of shadow snapshots to draw.
        seed: RNG seed for reproducible snapshots.
        k: median-of-means chunks (k=1 = plain mean).

    Returns:
        Real estimate of the expectation value.
    """
    return classical_shadow(state, shots, seed=seed).expval(observable, k=k)
