"""Determinant-basis tools: exact subspace matrices, RDMs, and solvers.

Several methods in this package need the same primitive: the Hamiltonian
matrix over a *chosen set* of Slater determinants, in the interleaved
spin-orbital basis ``P = 2p + s`` that :func:`_spin_orbital_integrals` uses.
Full CI, SQD, QSCI, DMET's impurity solver and the RDM-based entropies all
reduce to it, so it lives here once.

Determinants are integers: bit ``P`` set means spin orbital ``P`` is occupied
and has weight ``+1`` in the state's sign convention.  That matches the
character order of :class:`PauliString`, so a bitmask can be compared directly
against ``reference_occupation``.

Matrix elements are obtained by *applying* each second-quantized term to each
determinant with explicit fermionic signs, rather than by implementing the
Slater-Condon rules.  It is the slower route (``O(n_terms * n_dets)``) but it
reuses ``FermionicOperator.from_coeffs``'s own term construction, so there is
no second place for an index or sign convention to drift.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from superfermion.chemistry.hamiltonians import FermionicOperator
from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals

__all__ = [
    "popcount",
    "det_occupation",
    "all_determinants",
    "hamiltonian_matrix",
    "fermionic_terms",
    "ground_state",
    "rdm1",
    "rdm2",
    "exact_energy",
    "pauli_matrix",
]


def popcount(value: int) -> int:
    """Number of set bits."""
    return bin(value).count("1")


def det_occupation(det: int, n_qubits: int) -> List[int]:
    """Occupation list of ``det``, ordered like a Pauli string."""
    return [(det >> p) & 1 for p in range(n_qubits)]


def all_determinants(
    n_orbitals: int,
    n_electrons: int,
    *,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
) -> List[int]:
    """All determinants with the requested electrons, as sorted bitmasks.

    Args:
        n_orbitals: Spatial orbitals; the spin-orbital count is ``2 * n``.
        n_electrons: Total electrons, used when the spin split is not given.
        n_alpha: Alpha electrons to fix (optional).
        n_beta: Beta electrons to fix (optional).

    Returns:
        Ascending list of bitmasks.
    """
    n_so = 2 * n_orbitals
    if n_alpha is None or n_beta is None:
        if n_electrons > n_so:
            raise ValueError(
                f"{n_electrons} electrons do not fit in {n_so} spin orbitals"
            )
        return [det for det in range(1 << n_so) if popcount(det) == n_electrons]

    alpha_bits = [2 * p for p in range(n_orbitals)]
    beta_bits = [2 * p + 1 for p in range(n_orbitals)]
    from itertools import combinations

    results = []
    for a_occ in combinations(alpha_bits, n_alpha):
        base = 0
        for bit in a_occ:
            base |= 1 << bit
        for b_occ in combinations(beta_bits, n_beta):
            det = base
            for bit in b_occ:
                det |= 1 << bit
            results.append(det)
    return sorted(results)


def apply_operators(det: int, ops: Sequence[Tuple[int, int]]) -> Tuple[int, Optional[int]]:
    """Apply ``((site, create), ...)`` to ``det``.

    The tuple is written left-to-right as the product appears in the
    Hamiltonian, so the rightmost operator acts first.

    Returns:
        ``(sign, new_det)``, with ``sign == 0`` when the term annihilates.
    """
    sign = 1
    current = det
    for site, create in reversed(ops):
        occupied = (current >> site) & 1
        if create:
            if occupied:
                return 0, None
        else:
            if not occupied:
                return 0, None
        # Jordan-Wigner parity: (-1) raised to the occupations below `site`
        if (current & ((1 << site) - 1)).bit_count() & 1:
            sign = -sign
        if create:
            current |= 1 << site
        else:
            current &= ~(1 << site)
    return sign, current


def fermionic_terms(
    h1: np.ndarray, h2: Optional[np.ndarray]
) -> Dict[Tuple[Tuple[int, int], ...], complex]:
    """Spin-orbital integral arrays -> fermionic term dictionary.

    ``h1`` is ``(2n, 2n)`` and ``h2`` must already be in the layout
    ``FermionicOperator.from_coeffs`` expects (see
    :func:`_spin_orbital_integrals`), so this is a thin wrapper that exists to
    keep the convention in one place.
    """
    return FermionicOperator.from_coeffs(h1, h2).terms


def hamiltonian_matrix(
    terms: Dict[Tuple[Tuple[int, int], ...], complex],
    dets: Sequence[int],
    *,
    sparse: bool = False,
):
    """Hamiltonian matrix in the given determinant basis.

    Args:
        terms: Term dictionary from :func:`fermionic_terms`.
        dets: Determinants spanning the subspace.
        sparse: Return ``scipy.sparse.csr_matrix`` instead of a dense array.

    Returns:
        Square matrix of side ``len(dets)`` (real when the imaginary parts
        cancel, which they do for any Hermitian molecular Hamiltonian).
    """
    index = {det: row for row, det in enumerate(dets)}
    size = len(dets)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[complex] = []

    for ops, coeff in terms.items():
        for det in dets:
            sign, new_det = apply_operators(det, ops)
            if sign == 0:
                continue
            row = index.get(new_det)
            if row is None:
                continue
            rows.append(row)
            cols.append(index[det])
            vals.append(sign * coeff)

    if sparse:
        import scipy.sparse as sp

        matrix = sp.coo_matrix(
            (vals, (rows, cols)), shape=(size, size), dtype=complex
        ).tocsr()
        matrix.sum_duplicates()
        return matrix

    matrix = np.zeros((size, size), dtype=complex)
    np.add.at(matrix, (rows, cols), np.asarray(vals, dtype=complex))
    if np.max(np.abs(matrix.imag)) < 1e-12:
        return matrix.real
    return matrix


def ground_state(
    terms: Dict[Tuple[Tuple[int, int], ...], complex],
    dets: Sequence[int],
    *,
    k: int = 1,
    sparse: bool = True,
    which: str = "SA",
) -> Tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` eigenpairs of the Hamiltonian restricted to ``dets``.

    ``sparse=True`` asks ARPACK, which needs ``k < n - 1``; requesting the
    ground state of a one- or two-determinant space raises ``TypeError`` from
    ``eigsh`` rather than returning the answer, so small spaces fall back to
    ``numpy.linalg.eigh``.  Iterative schemes such as QSCI reach such spaces
    on their first pass, so the fallback is not an edge case.

    Returns:
        ``(energies, vectors)`` with ``vectors[:, i]`` the eigenvector for
        ``energies[i]``, indexed like ``dets``.
    """
    dets = list(dets)
    size = len(dets)
    if size == 0:
        raise ValueError("cannot diagonalise an empty determinant set")
    k = min(int(k), size)
    use_sparse = sparse and size > max(2, 2 * k)
    matrix = hamiltonian_matrix(terms, dets, sparse=use_sparse)

    if use_sparse:
        import scipy.sparse.linalg as sla

        values, vectors = sla.eigsh(matrix, k=k, which=which)
        order = np.argsort(values.real)
        return values.real[order], vectors[:, order]

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    values, vectors = np.linalg.eigh(np.asarray(matrix))
    order = np.argsort(values.real)
    return values.real[order][:k], vectors[:, order][:, :k]


def _coeff_vector(coefficients: np.ndarray, dets: Sequence[int]) -> Dict[int, complex]:
    return {
        det: complex(coefficients[i])
        for i, det in enumerate(dets)
        if abs(coefficients[i]) > 1e-14
    }


def rdm1(
    vector: np.ndarray,
    dets: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """Spin-orbital 1-RDM ``<a^dag_P a_Q>`` of a determinant-basis state.

    ``rho[P, Q] = sum_{d,d'} conj(c_d') c_d <d'| a^dag_P a_Q |d>``, so the
    bra amplitude is looked up by the *determinant* that the term lands on
    (``coeffs[new_det]``), not by its position in ``dets``.  Using the
    position silently returns a matrix for the wrong state whenever the
    determinants are not ``0, 1, 2, ...``.
    """
    coeffs = _coeff_vector(vector, dets)
    rho = np.zeros((n_qubits, n_qubits), dtype=complex)

    for det, amp in coeffs.items():
        for p in range(n_qubits):
            for q in range(n_qubits):
                sign, new_det = apply_operators(det, ((p, 1), (q, 0)))
                if sign == 0:
                    continue
                bra = coeffs.get(new_det)
                if bra is None:
                    continue
                rho[p, q] += sign * np.conj(bra) * amp
    return rho


def rdm2(
    vector: np.ndarray,
    dets: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """Spin-orbital 2-RDM ``<a^dag_P a^dag_Q a_R a_S>``.

    Index order is ``(P, Q, R, S)`` for the operator product
    ``a^dag_P a^dag_Q a_S a_R``, i.e. the standard chemistry ordering with the
    annihilation indices transposed relative to the array index order.
    """
    coeffs = _coeff_vector(vector, dets)
    rho = np.zeros((n_qubits,) * 4, dtype=complex)

    for det, amp in coeffs.items():
        for p in range(n_qubits):
            for q in range(n_qubits):
                for r in range(n_qubits):
                    for s in range(n_qubits):
                        ops = ((p, 1), (q, 1), (s, 0), (r, 0))
                        sign, new_det = apply_operators(det, ops)
                        if sign == 0:
                            continue
                        bra = coeffs.get(new_det)
                        if bra is None:
                            continue
                        rho[p, q, r, s] += sign * np.conj(bra) * amp
    return rho


_PARITY_CACHE: Dict[int, np.ndarray] = {}


def _sign_table(n_qubits: int) -> np.ndarray:
    """``(-1) ** popcount(i)`` for every basis index of an ``n``-qubit register.

    Cached per width: one table serves every term of a Hamiltonian and every
    sector of a reduction.
    """
    signs = _PARITY_CACHE.get(n_qubits)
    if signs is None:
        signs = np.ones(1 << n_qubits, dtype=float)
        remaining = np.arange(1 << n_qubits, dtype=np.int64)
        while np.any(remaining):
            signs = np.where(remaining & 1, -signs, signs)
            remaining >>= 1
        _PARITY_CACHE[n_qubits] = signs
    return signs


def pauli_matrix(hamiltonian) -> np.ndarray:
    """Dense matrix of a Pauli-sum Hamiltonian.

    Character ``q`` of a pauli string is qubit ``q``, so the matrix is the
    Kronecker product in character order with axis 0 the leftmost character
    -- the same convention the statevector simulator and
    :func:`det_occupation` use.  Only for small qubit counts; it exists so
    encodings can be checked against a spectrum that does not depend on the
    encoding.

    Each term is assembled from its action on a basis state rather than as a
    Kronecker product: ``P|b> = phase(b) |b xor mask>``, where ``mask`` marks
    the ``X``/``Y`` positions and ``phase`` is ``+-1`` from the ``Z`` positions
    times ``i`` per ``Y``.  That is ``O(2**n)`` per term instead of ``O(4**n)``,
    which is the difference between minutes and milliseconds on a twelve-qubit
    sector; the matrix is the same to rounding.
    """
    terms = list(hamiltonian.terms)
    if not terms:
        return np.zeros((1, 1), dtype=complex)
    n_qubits = len(terms[0].pauli_str)
    dimension = 1 << n_qubits
    basis = np.arange(dimension, dtype=np.int64)
    signs = _sign_table(n_qubits)
    out = np.zeros((dimension, dimension), dtype=complex)
    for term in terms:
        flip = 0
        parity = 0
        n_y = 0
        for position, character in enumerate(term.pauli_str):
            bit = 1 << (n_qubits - 1 - position)
            if character in ("X", "Y"):
                flip |= bit
            if character in ("Y", "Z"):
                parity |= bit
            if character == "Y":
                n_y += 1
        phase = signs[basis & parity]
        remainder = n_y % 4
        if remainder == 1:
            phase = phase * 1j
        elif remainder == 2:
            phase = -phase
        elif remainder == 3:
            phase = phase * -1j
        out[basis ^ flip, basis] += complex(term.coeffs) * phase
    return out


def exact_energy(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    n_orbitals: int,
    n_electrons: int,
    *,
    nuclear_repulsion: float = 0.0,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
) -> float:
    """Full CI energy in the given space, from spin-free MO integrals.

    ``h1`` is ``(n, n)`` and ``h2`` is ``(n, n, n, n)`` in chemist order, i.e.
    exactly what :class:`MolecularData` stores.  This is the oracle the
    approximate solvers in this package are checked against.
    """
    h1 = np.asarray(h1, dtype=float)
    if h2 is not None:
        h1_so, h2_so = _spin_orbital_integrals(h1, np.asarray(h2, dtype=float))
    else:
        h1_so = np.zeros((2 * n_orbitals, 2 * n_orbitals))
        for s in (0, 1):
            h1_so[s::2, s::2] = h1
        h2_so = None

    terms = fermionic_terms(h1_so, h2_so)
    dets = all_determinants(
        n_orbitals, n_electrons, n_alpha=n_alpha, n_beta=n_beta
    )
    energies, _ = ground_state(terms, dets, sparse=False)
    return float(energies[0] + nuclear_repulsion)
