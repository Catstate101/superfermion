"""Excited states, spin labelling and quantum subspace expansion.

Three separate things live here because they answer three different questions
and a caller usually wants all of them at once:

* :func:`excited_states` diagonalises the Hamiltonian in an explicit
  determinant space and keeps the lowest ``n_roots`` eigenpairs.  This is the
  exact answer for that space -- no circuit, no ansatz, no optimiser -- and it
  is the reference every approximate excited-state method in this package is
  measured against.
* :func:`qse_excited` is the quantum subspace expansion of McClean et al.,
  *Phys. Rev. A* 95, 042308 (2017): given one state (a VQE or QSCI vector) and
  a pool of excitation operators, diagonalise ``H`` and the overlap matrix in
  the span of ``{O_i |psi>}``.  It is the variant that runs on a
  determinant-space solver rather than a circuit, so the operators are applied
  exactly and the only approximation is the pool itself plus whatever the pool
  pushes outside the space (reported, not hidden).
* :func:`spin_squared` labels the result.  Without it an "excited state" list
  is a list of energies that may mix singlets and triplets, and a method that
  cannot tell them apart cannot be validated against a spectroscopic
  measurement.

**Spin labelling is built as an operator, not as a formula over RDMs.**  ``S^2``
is assembled as a normal-ordered term dictionary and contracted with the same
:func:`~superfermion.chemistry.subspace.hamiltonian_matrix` machinery the
Hamiltonian uses, so it is correct by construction rather than by an index
convention that has to be right in four places at once.  The RDM contraction
:func:`spin_squared_from_rdms` exists because large spaces have an RDM without
an explicit vector, and it is checked against the operator version in the
tests -- two independent implementations that have to agree.

``S^2 = S_z^2 + S_z + S_- S_+`` with, summing over *spatial* ``p, q`` and
writing ``P`` for spin orbitals (``2p`` alpha, ``2p+1`` beta):

    S_z^2  = 1/4 sum_PQ eps_P eps_Q a^dag_P a_P a^dag_Q a_Q
    S_-S_+ = sum_p a^dag_pbeta a_pbeta
             - sum_pq a^dag_qbeta a^dag_palpha a_qalpha a_pbeta

The second term of ``S_-S_+`` is where implementations usually go wrong: the
annihilators are ``a_qalpha a_pbeta``, i.e. the beta operator is on the right
and acts first, so the corresponding 2-RDM element is ``<a^dag_qbeta
a^dag_palpha a_pbeta a_qalpha>`` and *not* the ``<a^dag a^dag a a>`` element
with the same index order reversed.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.chemistry.subspace import (
    all_determinants,
    apply_operators,
    fermionic_terms,
    ground_state,
    hamiltonian_matrix,
)

__all__ = [
    "spin_squared_terms",
    "spin_squared_matrix",
    "spin_squared",
    "spin_squared_from_rdms",
    "spin_of",
    "excitation_pool",
    "excited_states",
    "exact_excited_states",
    "qse_excited",
]

#: An operator term: a tuple of ``(site, create)`` pairs, left to right.
Term = Tuple[Tuple[int, int], ...]


def spin_squared_terms(n_spatial: int) -> Dict[Term, float]:
    """``S^2`` as a normal-ordered fermionic term dictionary.

    Args:
        n_spatial: Spatial orbitals.  The spin-orbital count is ``2 n`` with
            ``2p`` alpha and ``2p+1`` beta, matching
            :func:`~superfermion.chemistry.subspace.all_determinants`.

    Returns:
        Term dictionary suitable for :func:`hamiltonian_matrix`.

    The returned operator is correct for any state that never occupies sites
    outside ``0 .. 2 n - 1``; over-counting ``n_spatial`` beyond what the
    determinants reach adds only terms that annihilate, so an upper bound is
    harmless (see :func:`qse_excited`).
    """
    n_spatial = int(n_spatial)
    if n_spatial < 1:
        raise ValueError(f"n_spatial must be >= 1, got {n_spatial}")
    n_so = 2 * n_spatial
    eps = np.where(np.arange(n_so) % 2 == 0, 1.0, -1.0)

    terms: Dict[Term, float] = {}

    def add(ops: Term, coeff: float) -> None:
        terms[ops] = terms.get(ops, 0.0) + float(coeff)

    # S_z = 1/2 sum_P eps_P a^dag_P a_P
    for p in range(n_so):
        add(((p, 1), (p, 0)), 0.5 * eps[p])
        # the a^dag_pbeta a_pbeta piece of S_-S_+ (beta sites are odd)
        if p % 2:
            add(((p, 1), (p, 0)), 1.0)

    # S_z^2 = 1/4 sum_PQ eps_P eps_Q a^dag_P a_P a^dag_Q a_Q
    for p in range(n_so):
        for q in range(n_so):
            add(
                ((p, 1), (p, 0), (q, 1), (q, 0)),
                0.25 * eps[p] * eps[q],
            )

    # -sum_pq a^dag_qbeta a^dag_palpha a_qalpha a_pbeta
    #
    # The last site is ``2p+1``, not ``2p``: the rightmost annihilator is the
    # *beta* partner of the alpha one that precedes it.  Writing ``2p`` there
    # collapses the operator onto a^dag_qbeta a^dag_palpha a_qalpha a_palpha,
    # which annihilates every closed-shell determinant -- so this term silently
    # vanished and S^2 came out as N/2 (5.0 for water) for every state.
    for q in range(n_spatial):
        for p in range(n_spatial):
            add(
                ((2 * q + 1, 1), (2 * p, 1), (2 * q, 0), (2 * p + 1, 0)),
                -1.0,
            )
    return {ops: coeff for ops, coeff in terms.items() if coeff != 0.0}


def spin_squared_matrix(
    determinants: Sequence[int], n_spatial: Optional[int] = None
) -> np.ndarray:
    """``S^2`` in the determinant basis, as a dense real matrix.

    ``n_spatial`` defaults to the smallest value that covers every
    determinant, which is enough because the operator annihilates any site it
    is given that the determinants never reach.
    """
    dets = list(determinants)
    if not dets:
        raise ValueError("spin_squared_matrix needs at least one determinant")
    if n_spatial is None:
        n_spatial = (max(dets).bit_length() + 1) // 2
    terms = spin_squared_terms(int(n_spatial))
    matrix = hamiltonian_matrix(terms, dets, sparse=False)
    return np.asarray(np.real_if_close(matrix, tol=1000), dtype=float)


def spin_squared(
    vector: np.ndarray,
    determinants: Sequence[int],
    n_spatial: Optional[int] = None,
) -> float:
    """``<psi|S^2|psi>`` for a determinant-space state vector."""
    matrix = spin_squared_matrix(determinants, n_spatial)
    coeffs = np.asarray(vector, dtype=complex).ravel()
    if coeffs.size != matrix.shape[0]:
        raise ValueError(
            f"vector has {coeffs.size} amplitudes for "
            f"{matrix.shape[0]} determinants"
        )
    return float(np.real(coeffs.conj() @ matrix @ coeffs))


def spin_squared_from_rdms(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """``<S^2>`` from the spin-orbital 1- and 2-RDMs alone.

    ``rho1[P, Q] = <a^dag_P a_Q>`` and
    ``rho2[P, Q, R, S] = <a^dag_P a^dag_Q a_S a_R>`` -- the layout
    :func:`~superfermion.chemistry.subspace.rdm1` and ``rdm2`` produce.

    ``O(n^2)`` and no state vector, which is what a large active space can
    afford; :func:`spin_squared` is the definition it must agree with.
    """
    rho1 = np.asarray(rho1)
    rho2 = np.asarray(rho2)
    n_so = rho1.shape[0]
    if rho1.shape != (n_so, n_so):
        raise ValueError(f"rho1 must be square, got {rho1.shape}")
    if rho2.shape != (n_so,) * 4:
        raise ValueError(f"rho2 must be {(n_so, n_so, n_so, n_so)}")
    eps = np.where(np.arange(n_so) % 2 == 0, 1.0, -1.0)
    diag = np.real(np.diag(rho1))

    n_elec = float(np.sum(diag))
    sz = 0.5 * float(np.dot(eps, diag))
    # sum_PQ eps_P eps_Q <a^dag_P a_P a^dag_Q a_Q>
    #   = sum_PQ eps_P eps_Q rho2[P, Q, Q, P] with the annihilators NOT swapped
    sz2 = 0.25 * n_elec - 0.25 * float(
        np.real(np.einsum("p,q,pqqp->", eps, eps, rho2))
    )
    # S_-S_+ = N_beta - sum_pq rho2[qbeta, palpha, pbeta, qalpha]
    n_beta = float(np.sum(diag[1::2]))
    cross = float(np.real(np.einsum("qppq->", rho2[1::2, 0::2, 1::2, 0::2])))
    return sz2 + sz + n_beta - cross


def spin_of(s2: float, *, tol: float = 1e-6) -> Tuple[float, int]:
    """``S`` and the multiplicity ``2S+1`` implied by ``<S^2>``.

    ``S(S+1) = <S^2>`` gives ``S = (-1 + sqrt(1 + 4<S^2>))/2``.  A value that
    is not consistent with any half-integer spin is returned as-is with the
    multiplicity rounded, and the caller can compare: a state that is not an
    ``S^2`` eigenstate reports ``<S^2>`` between two eigenvalues, and rounding
    it to a "label" would be the whole failure mode this function exists to
    avoid.  Compare the rounded multiplicity against
    ``2S+1`` recomputed from the returned ``S`` when it matters.
    """
    s2 = float(s2)
    if s2 < -tol:
        raise ValueError(f"<S^2> = {s2} is negative beyond tolerance")
    spin_s = 0.5 * (np.sqrt(1.0 + 4.0 * max(s2, 0.0)) - 1.0)
    return float(spin_s), int(round(2.0 * spin_s + 1.0))


def excitation_pool(
    n_spatial: int,
    reference: int,
    *,
    rank: int = 2,
    preserve_spin: bool = True,
    max_operators: Optional[int] = None,
) -> List[Term]:
    """Single and double excitation operators relative to ``reference``.

    Args:
        n_spatial: Spatial orbitals.
        reference: Reference determinant bitmask; its occupations define the
            occupied/virtual split.
        rank: ``1`` for singles only, ``2`` for singles and doubles, ``3`` to
            add triples, and so on.  The rank is a *ceiling*, not a request: a
            system with fewer virtual than occupied-to-virtual spin orbitals
            simply produces no operator of that size.  Water/STO-3G has two
            virtual spatial orbitals, so its triples are 2-alpha-1-beta and
            1-alpha-2-beta moves and nothing larger until rank 4.
        preserve_spin: Restrict to alpha->alpha and beta->beta moves (default).
            A spin-flipping operator maps the state out of its ``S_z`` sector,
            so mixing one into the pool adds a column that is entirely
            ``leaked`` and, in the QSE diagonalisation, looks like a rank
            deficiency rather than the mistake it is.  Set ``False`` only when
            the determinant space really does span several ``S_z`` values.
        max_operators: Truncate the list (singles survive first, so the
            truncation is reproducible) instead of silently generating a pool
            too large to diagonalise.

    Returns:
        List of operator terms: singles, alpha-alpha doubles, beta-beta
        doubles, then alpha-beta doubles.
    """
    n_so = 2 * int(n_spatial)
    reference = int(reference)
    occupied = [s for s in range(n_so) if (reference >> s) & 1]
    virtual = [s for s in range(n_so) if not (reference >> s) & 1]

    occ_a = [s for s in occupied if s % 2 == 0]
    occ_b = [s for s in occupied if s % 2 == 1]
    vir_a = [s for s in virtual if s % 2 == 0]
    vir_b = [s for s in virtual if s % 2 == 1]
    if not preserve_spin:
        occ_a = occ_b = occupied
        vir_a = vir_b = virtual

    pool: List[Term] = []
    for targets, sources in ((vir_a, occ_a), (vir_b, occ_b)):
        for a in targets:
            for i in sources:
                pool.append(((a, 1), (i, 0)))
    if rank >= 2:
        for targets, sources in ((vir_a, occ_a), (vir_b, occ_b)):
            for a_pair in combinations(targets, 2):
                for i_pair in combinations(sources, 2):
                    pool.append(
                        (
                            (a_pair[0], 1),
                            (a_pair[1], 1),
                            (i_pair[1], 0),
                            (i_pair[0], 0),
                        )
                    )
        for a1 in vir_a:
            for a2 in vir_b:
                for i1 in occ_a:
                    for i2 in occ_b:
                        pool.append(
                            ((a1, 1), (a2, 1), (i2, 0), (i1, 0))
                        )
    if rank >= 3:
        # Higher ranks, in the same convention as the doubles above: creations
        # ascending by spin orbital, annihilations descending.  Keeping one
        # convention at every rank is what makes the signs of a triple
        # consistent with the signs of the doubles it is built from.
        seen = set(pool)
        for total in range(3, int(rank) + 1):
            for n_alpha in range(total + 1):
                n_beta = total - n_alpha
                if n_alpha > len(occ_a) or n_alpha > len(vir_a):
                    continue
                if n_beta > len(occ_b) or n_beta > len(vir_b):
                    continue
                for out_a in combinations(vir_a, n_alpha):
                    for in_a in combinations(occ_a, n_alpha):
                        for out_b in combinations(vir_b, n_beta):
                            for in_b in combinations(occ_b, n_beta):
                                term = tuple(
                                    (site, 1) for site in sorted(out_a + out_b)
                                ) + tuple(
                                    (site, 0)
                                    for site in sorted(in_a + in_b, reverse=True)
                                )
                                if term not in seen:
                                    seen.add(term)
                                    pool.append(term)

    if max_operators is not None and len(pool) > int(max_operators):
        pool = pool[: int(max_operators)]
    return pool


def _active_space_terms(
    mol: Any,
    active_indices: Optional[Sequence[int]],
    n_frozen_core: int,
    n_alpha: Optional[int],
    n_beta: Optional[int],
    max_determinants: int,
) -> Dict[str, Any]:
    """Shared setup: contracted integrals, determinant list, energy offset."""
    from math import comb

    from superfermion.chemistry.pyscf_bridge import (
        MolecularData,
        _spin_orbital_integrals,
    )

    if not isinstance(mol, MolecularData):
        raise TypeError("excited_states expects a MolecularData")
    active = None if active_indices is None else [int(i) for i in active_indices]
    h1_as, h2_as, n_elec, n_orbs, e_fc = mol._apply_active_space(
        None, active_indices=active, frozen_core=n_frozen_core
    )
    if n_alpha is None and n_beta is None:
        n_alpha = (n_elec + int(mol.spin)) // 2
        n_beta = n_elec - n_alpha
    elif n_alpha is None or n_beta is None:
        raise ValueError("give both n_alpha and n_beta, or neither")
    if n_alpha + n_beta != n_elec:
        raise ValueError(
            f"n_alpha + n_beta = {n_alpha + n_beta} != {n_elec} active electrons"
        )
    size = comb(n_orbs, n_alpha) * comb(n_orbs, n_beta)
    if size > max_determinants:
        raise ValueError(
            f"a CAS({n_elec},{n_orbs}) in this spin sector has {size} "
            f"determinants, above max_determinants={max_determinants}"
        )
    h1_so, h2_so = _spin_orbital_integrals(h1_as, h2_as)
    terms = fermionic_terms(h1_so, h2_so)
    dets = all_determinants(n_orbs, n_elec, n_alpha=n_alpha, n_beta=n_beta)
    return {
        "terms": terms,
        "determinants": dets,
        "n_spatial": n_orbs,
        "n_active_electrons": n_elec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "energy_offset": float(mol.nuclear_repulsion) + float(e_fc),
    }


def excited_states(
    mol: Any,
    *,
    n_roots: int = 4,
    active_indices: Optional[Sequence[int]] = None,
    n_frozen_core: int = 0,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    max_determinants: int = 20000,
) -> Dict[str, Any]:
    """Lowest ``n_roots`` eigenpairs of the Hamiltonian in a determinant space.

    Exact for that space: the returned energies are what
    ``fci.FCI(nroots=n_roots)`` returns on the same integrals.  Excited states
    are not variational in a targeted sense -- they are simply the next
    eigenvalues -- so the numbers are exact where a QEOM/VQD estimate is not.

    Args:
        mol: :class:`~superfermion.chemistry.pyscf_bridge.MolecularData`.
        n_roots: Number of eigenpairs to keep.
        active_indices: Active orbitals, or ``None`` for all of them.
        n_frozen_core: Lowest orbitals treated as a closed-shell core.
        n_alpha, n_beta: Spin sector; default from ``n_elec`` and ``mol.spin``.
        max_determinants: Refuse larger spaces rather than trying to build
            them.

    Returns:
        Dict with ``energies`` (absolute, nuclear repulsion and frozen-core
        energy included), ``spin_squared``, ``multiplicity``, ``spin``,
        ``vectors`` (columns, indexed like ``determinants``), ``terms`` and
        ``determinants`` (so :func:`qse_excited` can be chained), plus
        ``energy_offset``, ``n_determinants``, ``n_active_electrons``,
        ``n_active_orbitals`` and ``reference_energy`` (the lowest root).
    """
    setup = _active_space_terms(
        mol, active_indices, n_frozen_core, n_alpha, n_beta, max_determinants
    )
    terms = setup["terms"]
    dets = setup["determinants"]
    k = int(n_roots)
    if k < 1:
        raise ValueError(f"n_roots must be >= 1, got {n_roots}")
    if k > len(dets):
        raise ValueError(
            f"n_roots={k} exceeds the {len(dets)} determinants in the space"
        )
    values, vectors = ground_state(terms, dets, k=k, sparse=True)
    s2_matrix = spin_squared_matrix(dets, setup["n_spatial"])
    s2_values = np.array(
        [float(np.real(vectors[:, i].conj() @ s2_matrix @ vectors[:, i]))
         for i in range(values.size)]
    )
    spins = [spin_of(value) for value in s2_values]
    offset = setup["energy_offset"]
    energies = np.asarray(values, dtype=float) + offset
    return {
        "energies": energies,
        "spin_squared": s2_values,
        "spin": np.array([s for s, _ in spins]),
        "multiplicity": np.array([m for _, m in spins], dtype=int),
        "vectors": np.asarray(vectors),
        "terms": terms,
        "determinants": dets,
        "energy_offset": offset,
        "n_determinants": len(dets),
        "n_roots": int(values.size),
        "n_active_electrons": setup["n_active_electrons"],
        "n_active_orbitals": setup["n_spatial"],
        "n_alpha": setup["n_alpha"],
        "n_beta": setup["n_beta"],
        "reference_energy": float(energies[0]),
    }


def _apply_to_state(
    vector: np.ndarray,
    determinants: Sequence[int],
    index: Dict[int, int],
    ops: Term,
) -> Tuple[np.ndarray, float]:
    """``ops |vector>`` inside the determinant space, plus the leaked weight."""
    out = np.zeros_like(vector)
    leaked = 0.0
    for position, det in enumerate(determinants):
        amplitude = vector[position]
        if amplitude == 0.0:
            continue
        sign, new_det = apply_operators(det, ops)
        if sign == 0:
            continue
        target = index.get(new_det)
        if target is None:
            leaked += abs(amplitude) ** 2
            continue
        out[target] += sign * amplitude
    return out, leaked


def qse_excited(
    terms: Dict[Term, complex],
    determinants: Sequence[int],
    reference: np.ndarray,
    *,
    pool: Optional[Sequence[Term]] = None,
    n_spatial: Optional[int] = None,
    n_roots: int = 3,
    energy_offset: float = 0.0,
    threshold: float = 1e-10,
) -> Dict[str, Any]:
    """Quantum subspace expansion around one state.

    Builds the subspace ``{|psi>, O_1|psi>, O_2|psi>, ...}``, orthonormalises
    it by SVD (the expanded set is generically rank deficient -- ``O_i|psi>``
    can be zero or a repeat), then diagonalises ``H`` in it.  Because the
    operators are applied exactly to a determinant-space vector, the only
    error is the pool and any amplitude driven outside ``determinants``, which
    is measured per operator and returned in ``leaked``.

    Args:
        terms: Hamiltonian term dictionary.
        determinants: Determinants spanning the solver space.
        reference: State vector indexed like ``determinants`` (e.g. the
            ground state from :func:`excited_states`).  Pass a single bitmask
            to start from that determinant.
        pool: Operator terms.  Defaults to singles and doubles relative to
            the largest-amplitude determinant of ``reference``, which needs
            ``n_spatial``.
        n_spatial: Spatial orbitals; inferred when omitted.
        n_roots: Eigenpairs to keep.
        energy_offset: Added to every returned energy (nuclear repulsion plus
            frozen-core energy when chaining from :func:`excited_states`).
        threshold: Relative singular-value cutoff for the subspace.

    Returns:
        Dict with ``energies``, ``vectors`` (columns, indexed like
        ``determinants``), ``coefficients`` (expansion in the orthonormal
        subspace basis), ``spin_squared``, ``spin``, ``multiplicity``,
        ``pool_size``, ``subspace_dimension``, ``leaked`` (list, one per
        operator, of the norm-squared driven out of the space) and
        ``reference_energy``.
    """
    dets = list(determinants)
    if not dets:
        raise ValueError("qse_excited needs at least one determinant")
    index = {det: row for row, det in enumerate(dets)}
    if n_spatial is None:
        n_spatial = (max(dets).bit_length() + 1) // 2
    n_spatial = int(n_spatial)

    if isinstance(reference, (int, np.integer)):
        vector = np.zeros(len(dets), dtype=complex)
        position = index.get(int(reference))
        if position is None:
            raise ValueError(
                f"reference determinant {reference} is not in the space"
            )
        vector[position] = 1.0
    else:
        # Keep the state complex.  ``ground_state`` returns whatever dtype
        # ARPACK produced, and casting a complex vector to float raises a
        # ComplexWarning while silently discarding exactly the part that says
        # whether the vector was real in the first place.
        vector = np.asarray(reference, dtype=complex).ravel()
        if vector.size != len(dets):
            raise ValueError(
                f"reference has {vector.size} amplitudes for {len(dets)} "
                f"determinants"
            )
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("reference state has zero norm")
    vector = vector / norm

    if pool is None:
        anchor = dets[int(np.argmax(np.abs(vector)))]
        pool = excitation_pool(n_spatial, anchor)
    pool = list(pool)

    columns: List[np.ndarray] = [vector]
    leaked: List[float] = []
    for ops in pool:
        column, escaped = _apply_to_state(vector, dets, index, ops)
        columns.append(column)
        leaked.append(escaped)

    basis = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    cutoff = max(threshold * (singular[0] if singular.size else 0.0), 0.0)
    keep = singular > cutoff
    if not np.any(keep):
        raise ValueError("the QSE pool spans nothing")
    rank = int(np.count_nonzero(keep))
    orthonormal = left[:, keep]

    matrix = hamiltonian_matrix(terms, dets, sparse=False)
    matrix = np.asarray(np.real_if_close(matrix, tol=1000))
    if not np.iscomplexobj(matrix) or np.max(np.abs(matrix.imag)) < 1e-10:
        matrix = np.real(matrix)
    projected = orthonormal.conj().T @ matrix @ orthonormal
    projected = 0.5 * (projected + projected.conj().T)
    values, coeffs = np.linalg.eigh(projected)
    order = np.argsort(values.real)
    values, coeffs = values.real[order], coeffs[:, order]
    k = min(int(n_roots), values.size)
    values, coeffs = values[:k], coeffs[:, :k]
    vectors = orthonormal @ coeffs

    s2_matrix = spin_squared_matrix(dets, n_spatial)
    s2_values = np.array(
        [
            float(np.real(vectors[:, i].conj() @ s2_matrix @ vectors[:, i]))
            for i in range(values.size)
        ]
    )
    spins = [spin_of(value) for value in s2_values]
    return {
        "energies": values + float(energy_offset),
        "vectors": vectors,
        "coefficients": coeffs,
        "spin_squared": s2_values,
        "spin": np.array([s for s, _ in spins]),
        "multiplicity": np.array([m for _, m in spins], dtype=int),
        "pool_size": len(pool),
        "subspace_dimension": rank,
        "leaked": leaked,
        "reference_energy": float(values[0] + float(energy_offset)),
    }


def exact_excited_states(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Alias for :func:`excited_states`, kept for callers that spell it out."""
    return excited_states(*args, **kwargs)
