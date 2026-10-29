"""Orbital-basis reduction: optimise the one-body basis before mapping.

Darmawan et al., *EPJ Quantum Technol.* (2025).  A molecular Hamiltonian is
not tied to the canonical SCF orbitals.  Any orthogonal rotation ``U`` of the
orbital basis gives the same spectrum but a *different* Pauli sum, and some
bases produce dramatically fewer terms.  Since the mapping cost -- circuit
depth, term count, measurement budget -- is a property of the Pauli sum and
not of the physics, choosing the basis is free accuracy.

This module rotates the integrals and measures the result directly:

    h1' = U^T h1 U,      h2'_{ijkl} = sum U_pi U_qj U_rk U_sl h2_{pqrs}

with ``U = exp(kappa)`` for antisymmetric ``kappa``.  The optimisation
minimises the L1 mass of the rotated integrals, which is a convex-sparsity
surrogate for "how many entries are non-negligible" -- the Frobenius norm
would be invariant under the rotation and therefore useless as an objective.

Two invariants make the result checkable rather than plausible:

* the exact (full CI) energy must not change, because an orthogonal rotation
  is a similarity transformation of the Hamiltonian;
* the L1 objective must not go up -- and when the search cannot improve it,
  the SCF basis is returned unchanged instead of a rotation that merely
  redistributes magnitudes.  The reported Pauli term counts are measured
  after mapping, not predicted.

The random restarts are what the name refers to: the objective is
non-convex, so several random ``kappa`` seeds are tried and the best is kept.

The surrogate is *not* even in ``kappa`` in general, and ``kappa = 0`` is not a
stationary point of it.  Measured on a synthetic four-orbital model the
finite-difference slope at the origin is ``-5.42`` along a random direction,
``f(+kappa) - f(-kappa)`` is ``-1.1e-3`` at ``|kappa| = 1e-4``, and BFGS started
exactly at zero still descends from ``39.79`` to ``21.07``.  Evenness is a
property of symmetric *inputs*, not of the surrogate: on H2/STO-3G there is a
single rotation parameter and a diagonal ``h1``, so ``f(+kappa) - f(-kappa)`` is
below ``1e-15`` at every step size tried, and the origin is a kink of the
``|.|`` sum rather than a smooth stationary point -- the slope is ``+3.02``
from both sides, so every direction raises the objective and no start finds an
improvement.  The SCF basis is then correctly returned unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "rotation_matrix",
    "rotate_integrals",
    "rotate_by_unitary",
    "basis_objective",
    "cisd_determinants",
    "pool_energy",
    "pauli_term_stats",
    "optimise_orbital_basis",
    "COST_FUNCTIONS",
]

#: Accepted values of ``optimise_orbital_basis(cost=...)``.
COST_FUNCTIONS = ("sparsity", "fci", "cisd", "custom")


def rotation_matrix(kappa: np.ndarray) -> np.ndarray:
    """``U = exp(kappa)`` for an antisymmetric generator.

    Raises:
        ValueError: if ``kappa`` is not square and antisymmetric.  A symmetric
            generator would not preserve orthonormality, and the resulting
            integrals would describe a different molecule.
    """
    k = np.asarray(kappa, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError(f"kappa must be square, got shape {k.shape}")
    if not np.allclose(k, -k.T, atol=1e-12):
        raise ValueError("kappa must be antisymmetric")
    # expm of a real antisymmetric matrix is orthogonal; expm is imported
    # lazily because it is the only thing this module needs from scipy.
    from scipy.linalg import expm

    return expm(k)


def rotate_by_unitary(
    h1: np.ndarray, h2: Optional[np.ndarray], U: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rotate the integrals by an explicit orthogonal matrix.

    ``rotate_integrals`` builds ``U`` from a generator; this takes the matrix
    directly, which is what a natural-orbital or an AVAS transformation needs.

    Args:
        h1: ``(n, n)`` one-body integrals.
        h2: ``(n, n, n, n)`` chemist ERIs, or ``None``.
        U: ``(n, n)`` orthogonal matrix; column ``j`` is new orbital ``j``.

    Raises:
        ValueError: if ``U`` is not square/orthogonal, or its size does not
            match ``h1``.
    """
    U = np.asarray(U, dtype=float)
    h1 = np.asarray(h1, dtype=float)
    n = h1.shape[0]
    if U.shape != (n, n):
        raise ValueError(f"U must be {(n, n)}, got {U.shape}")
    if not np.allclose(U.T @ U, np.eye(n), atol=1e-10):
        raise ValueError("U must be orthogonal: U^T U = I")
    h1_new = U.T @ h1 @ U
    if h2 is None:
        return h1_new, None
    h2 = np.asarray(h2, dtype=float)
    h2_new = np.einsum("pi,qj,rk,sl,pqrs->ijkl", U, U, U, U, h2, optimize=True)
    return h1_new, h2_new


def rotate_integrals(
    h1: np.ndarray, h2: Optional[np.ndarray], kappa: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rotate the integrals by ``exp(kappa)``."""
    return rotate_by_unitary(h1, h2, rotation_matrix(kappa))


def basis_objective(
    h1: np.ndarray, h2: Optional[np.ndarray], *, h2_weight: float = 1.0
) -> float:
    """Sparsity surrogate: off-diagonal one-body L1 plus two-body L1.

    The one-body diagonal is excluded because it is the orbital energies,
    which the rotation is entitled to change but not to shrink.
    """
    h1 = np.asarray(h1, dtype=float)
    total = float(np.sum(np.abs(h1)) - np.sum(np.abs(np.diag(h1))))
    if h2 is not None:
        total += h2_weight * float(np.sum(np.abs(h2)))
    return total


def cisd_determinants(
    n_orbitals: int,
    *,
    n_alpha: int,
    n_beta: int,
    reference: Optional[int] = None,
) -> List[int]:
    """Reference, singles and doubles within the ``(n_alpha, n_beta)`` sector.

    Built by excitation rank, not by enumerating ``C(n, k)`` determinants and
    filtering: the pool is ``O(n_occ^2 n_virt^2)`` while the sector is
    exponential, which is the whole reason the CISD cost is affordable where an
    FCI cost is not.

    The reference defaults to the aufbau occupancy of the orbital order the
    integrals are currently expressed in.  Note what that means for
    :func:`optimise_orbital_basis`: it resolves this pool once, before the
    search, so the objective is the ground state of a *fixed* set of index
    patterns in a rotating basis, not the CISD space of the rotated aufbau
    reference.  From a canonical Hartree-Fock start that objective is
    stationary regardless -- singles do not couple to the reference
    (Brillouin) -- so the search reports no improvement, which is the correct
    answer rather than a stall.

    Args:
        n_orbitals: Spatial orbitals.
        n_alpha, n_beta: Occupation of the reference, i.e. the sector.
        reference: Explicit reference bitmask. Must contain exactly
            ``n_alpha`` alpha and ``n_beta`` beta modes.

    Returns:
        Ascending list of determinant bitmasks.
    """
    from itertools import combinations

    n = int(n_orbitals)
    if not (0 <= n_alpha <= n and 0 <= n_beta <= n):
        raise ValueError(
            f"n_alpha={n_alpha}, n_beta={n_beta} do not fit in {n} orbitals"
        )
    if reference is None:
        occ_a = list(range(n_alpha))
        occ_b = list(range(n_beta))
    else:
        bits = int(reference)
        occ_a = [p for p in range(n) if (bits >> (2 * p)) & 1]
        occ_b = [p for p in range(n) if (bits >> (2 * p + 1)) & 1]
        if len(occ_a) != n_alpha or len(occ_b) != n_beta:
            raise ValueError(
                f"reference {reference} has ({len(occ_a)}, {len(occ_b)}) "
                f"electrons, not ({n_alpha}, {n_beta})"
            )
    vir_a = [p for p in range(n) if p not in set(occ_a)]
    vir_b = [p for p in range(n) if p not in set(occ_b)]

    def det(a_occ, b_occ) -> int:
        value = 0
        for p in a_occ:
            value |= 1 << (2 * p)
        for p in b_occ:
            value |= 1 << (2 * p + 1)
        return value

    pool = {det(occ_a, occ_b)}
    # singles: one alpha OR one beta moved from an occupied to a virtual mode
    for i in occ_a:
        for a in vir_a:
            pool.add(det([a if p == i else p for p in occ_a], occ_b))
    for i in occ_b:
        for a in vir_b:
            pool.add(det(occ_a, [a if p == i else p for p in occ_b]))
    # doubles: alpha-alpha, beta-beta and alpha-beta
    for i, j in combinations(occ_a, 2):
        for a, b in combinations(vir_a, 2):
            pool.add(
                det([a if p == i else b if p == j else p for p in occ_a], occ_b)
            )
    for i, j in combinations(occ_b, 2):
        for a, b in combinations(vir_b, 2):
            pool.add(
                det(occ_a, [a if p == i else b if p == j else p for p in occ_b])
            )
    for i in occ_a:
        for j in occ_b:
            for a in vir_a:
                for b in vir_b:
                    pool.add(
                        det(
                            [a if p == i else p for p in occ_a],
                            [b if p == j else p for p in occ_b],
                        )
                    )
    return sorted(pool)


def pool_energy(
    h1: np.ndarray, h2: Optional[np.ndarray], determinants: Sequence[int]
) -> float:
    """Lowest eigenvalue of H restricted to ``determinants``.

    Thin wrapper over :mod:`superfermion.chemistry.subspace` so a cost function
    and a reference calculation cannot disagree about the layout.
    """
    from superfermion.chemistry.subspace import ground_state, fermionic_terms
    from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals

    h1 = np.asarray(h1, dtype=float)
    if h2 is None:
        n = h1.shape[0]
        h1_so = np.zeros((2 * n, 2 * n))
        for s in (0, 1):
            h1_so[s::2, s::2] = h1
        h2_so = None
    else:
        h1_so, h2_so = _spin_orbital_integrals(h1, np.asarray(h2, dtype=float))
    terms = fermionic_terms(h1_so, h2_so)
    energies, _ = ground_state(terms, list(determinants), sparse=True)
    return float(energies[0])


def pauli_term_stats(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    *,
    threshold: float = 1e-10,
) -> Dict[str, Any]:
    """Map the integrals to a Pauli sum and report its size.

    Returns:
        Dict with ``n_qubits``, ``n_terms`` (terms above ``threshold``),
        ``max_weight`` (the largest Pauli support) and ``l1_norm``.
    """
    from superfermion.chemistry.hamiltonians import FermionicOperator
    from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals

    n = np.asarray(h1).shape[0]
    h1_so, h2_so = _spin_orbital_integrals(
        np.asarray(h1, dtype=float),
        None if h2 is None else np.asarray(h2, dtype=float),
    )
    op = FermionicOperator.from_coeffs(h1_so, h2_so)
    mapped = op.jordan_wigner(2 * n)
    terms = list(mapped.terms)
    kept = [t for t in terms if abs(complex(t.coeffs)) > threshold]
    return {
        "n_qubits": 2 * n,
        "n_terms": len(kept),
        "max_weight": max(
            (sum(1 for ch in t.pauli_str if ch != "I") for t in kept), default=0
        ),
        "l1_norm": float(sum(abs(complex(t.coeffs)) for t in kept)),
    }


def _random_kappa(n: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    raw = rng.normal(size=(n, n)) * scale
    return raw - raw.T


def optimise_orbital_basis(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    *,
    n_starts: int = 4,
    seed: int = 0,
    scale: float = 0.5,
    first_start_scale: float = 0.05,
    maxiter: int = 300,
    h2_weight: float = 1.0,
    tolerance: float = 1e-10,
    cost: str = "sparsity",
    cost_function: Optional[Callable[[np.ndarray, Optional[np.ndarray]], float]] = None,
    n_electrons: Optional[int] = None,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
) -> Dict[str, Any]:
    """Search for an orbital basis with a sparser Hamiltonian or lower cost.

    Args:
        h1, h2: MO integrals in the chemist layout used everywhere else.
        n_starts: Number of restarts. ``1`` runs a single local search from a
            small perturbation of the SCF basis.
        seed: RNG seed, so a reported reduction is reproducible.
        scale: Standard deviation of the random generator entries.
        first_start_scale: Standard deviation of the first start's generator.
            Small, so that start is a local search out of the SCF basis; the
            remaining starts use ``scale``.  Keeping it non-zero avoids asking
            BFGS for a derivative at the origin, where the ``|.|`` sum can be
            even in a rotation parameter and therefore kinked.
        maxiter: BFGS iteration cap per start.
        h2_weight: Weight of the two-body L1 term (``cost='sparsity'`` only).
        tolerance: Objective improvement below which a start is abandoned.
        cost: Which objective to minimise.

            * ``'sparsity'`` -- the L1 surrogate above (the historical
              behaviour, still the default).
            * ``'fci'`` -- the exact ground-state energy of the rotated
              integrals.  **This objective is flat**: the spectrum of a
              unitary rotation is unchanged, so this exists as a correctness
              check on :func:`rotate_integrals` (it must not move) and as the
              reference against which the truncated costs are judged.  It
              cannot, on its own, select a better basis.
            * ``'cisd'`` -- the ground-state energy of a singles+doubles pool
              built once around the aufbau determinant of the starting
              ordering.  Not rotation-invariant, so it is a candidate
              objective -- but from a canonical Hartree-Fock start it is
              stationary (singles do not couple to the reference: Brillouin),
              so a search from there reports zero improvement by construction.
            * ``'custom'`` -- any ``cost_function(h1, h2) -> float``, the hook
              an external solver (a DMRG or MPS energy, a hardware estimate)
              plugs into.
        cost_function: Required for ``cost='custom'``; must accept the rotated
            integrals and return a scalar to minimise.
        n_electrons, n_alpha, n_beta: Electron count and spin sector for the
            ``'fci'`` and ``'cisd'`` costs.  ``n_alpha``/``n_beta`` default to
            an even split of ``n_electrons`` when only the count is given.

    Returns:
        Dict with ``h1``, ``h2`` (rotated integrals), ``kappa``,
        ``objective_before``, ``objective_after``, ``improvement``,
        ``n_starts``, ``history`` (objective per restart), ``cost`` and, for the
        determinant-space costs, ``n_determinants``.
    """
    from scipy.optimize import minimize

    h1 = np.asarray(h1, dtype=float)
    h2 = None if h2 is None else np.asarray(h2, dtype=float)
    n = h1.shape[0]
    if cost not in COST_FUNCTIONS:
        raise ValueError(
            f"cost={cost!r} is not one of {COST_FUNCTIONS}"
        )

    # Resolve the determinant pool once: it depends only on the electron
    # count and the aufbau ordering, not on the rotation being searched.
    determinants: Optional[List[int]] = None
    if cost in ("fci", "cisd"):
        if n_electrons is None:
            raise ValueError(f"cost={cost!r} needs n_electrons=")
        if n_alpha is None and n_beta is None:
            if n_electrons % 2:
                raise ValueError(
                    "an odd electron count needs an explicit n_alpha/n_beta "
                    "split (S_z is not implied)"
                )
            n_alpha = n_beta = n_electrons // 2
        elif n_alpha is None or n_beta is None:
            raise ValueError("give both n_alpha and n_beta, or neither")
        if n_alpha + n_beta != n_electrons:
            raise ValueError(
                f"n_alpha + n_beta = {n_alpha + n_beta} != "
                f"n_electrons = {n_electrons}"
            )
        if cost == "fci":
            from superfermion.chemistry.subspace import all_determinants

            determinants = all_determinants(
                n, n_electrons, n_alpha=n_alpha, n_beta=n_beta
            )
        else:
            determinants = cisd_determinants(
                n, n_alpha=n_alpha, n_beta=n_beta
            )

    def rotate_and_score(x: np.ndarray) -> float:
        r1, r2 = rotate_integrals(h1, h2, unpack(x))
        if cost == "sparsity":
            return basis_objective(r1, r2, h2_weight=h2_weight)
        if cost == "custom":
            return float(cost_function(r1, r2))
        return pool_energy(r1, r2, determinants)

    tri = np.triu_indices(n, k=1)

    def unpack(x: np.ndarray) -> np.ndarray:
        k = np.zeros((n, n))
        k[tri] = x
        return k - k.T

    if cost == "custom" and cost_function is None:
        raise ValueError("cost='custom' needs cost_function= to be a callable")

    if n < 2:
        identity = np.zeros((n, n))
        before = rotate_and_score(np.zeros(len(tri[0])))
        return {
            "h1": h1,
            "h2": h2,
            "kappa": identity,
            "objective_before": before,
            "objective_after": before,
            "improvement": 0.0,
            "n_starts": 0,
            "history": [],
            "cost": cost,
            "n_determinants": 0 if determinants is None else len(determinants),
        }

    def objective(x: np.ndarray) -> float:
        return rotate_and_score(x)

    rng = np.random.default_rng(seed)
    starts = [_random_kappa(n, rng, first_start_scale)[tri]]
    starts.extend(_random_kappa(n, rng, scale)[tri] for _ in range(max(0, n_starts - 1)))

    best = None
    history = []
    for x0 in starts:
        result = minimize(
            objective, x0, method="BFGS", options={"maxiter": maxiter}
        )
        history.append(float(result.fun))
        if best is None or result.fun < best.fun - tolerance:
            best = result

    kappa = unpack(best.x)
    before = objective(np.zeros(len(tri[0])))
    if best.fun >= before - tolerance:
        # Nothing was gained.  An unchanged Hamiltonian is a usable answer;
        # a rotation chosen by a tie is not, so return the SCF basis.
        kappa = np.zeros((n, n))
        h1_new, h2_new = h1, h2
        after = before
    else:
        h1_new, h2_new = rotate_integrals(h1, h2, kappa)
        after = objective(best.x)
    return {
        "h1": h1_new,
        "h2": h2_new,
        "kappa": kappa,
        "objective_before": before,
        "objective_after": after,
        "improvement": before - after,
        "n_starts": len(starts),
        "history": history,
        "cost": cost,
        "n_determinants": 0 if determinants is None else len(determinants),
        "optimiser_nit": int(getattr(best, "nit", 0)),
    }
