"""SQD and QSCI: diagonalise the Hamiltonian in a sampled determinant space.

Robledo-Moreno et al., arXiv:2405.05068 (2024) for sample-based quantum
diagonalization; Kanno et al., arXiv:2302.11320 (2023) for quantum-selected
configuration interaction.  Both methods have the same shape:

    produce a set of Slater determinants, then diagonalise H inside it.

The quantum part is only the *proposal*; the accuracy comes from the classical
eigensolver, which is why an exact subspace matrix (see
:mod:`superfermion.chemistry.subspace`) is the right thing to build this on.

Two selection rules are provided, and they are not interchangeable:

* **QSCI** ranks determinants by ``|<d|H|psi>|`` -- how strongly a determinant
  couples to the current state -- and re-selects after each diagonalisation,
  so a determinant that only matters at the second order is still found.
* **SQD** takes the determinants that were actually sampled from a quantum
  state, which weights them by ``|<d|psi>|^2`` instead.  Sampling concentrates
  on high-amplitude determinants; coupling concentrates on high-energy-gain
  ones.  With a complete determinant set both are exact, which is the only
  case where they are comparable.

``sample_determinants`` uses the exact amplitudes as the sampling distribution.
That is a stand-in for hardware with honest statistics: the sampler really
draws bitstrings, and the energy really is computed from whatever came out, so
a poor sample set gives a poor energy rather than the reference answer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.chemistry import subspace as sub
from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals

__all__ = [
    "subspace_energy",
    "sqd_energy",
    "sample_determinants",
    "qsci",
]

_DEFAULT_POOL_LIMIT = 40000


def _spin_terms(h1: np.ndarray, h2: Optional[np.ndarray]):
    h1 = np.asarray(h1, dtype=float)
    if h2 is None:
        n = h1.shape[0]
        h1_so = np.zeros((2 * n, 2 * n))
        for s in (0, 1):
            h1_so[s::2, s::2] = h1
        return h1_so, None
    return _spin_orbital_integrals(h1, np.asarray(h2, dtype=float))


def subspace_energy(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    determinants: Sequence[int],
    *,
    sparse: bool = True,
    k: int = 1,
) -> Tuple[float, np.ndarray]:
    """Lowest eigenvalues of H restricted to ``determinants``.

    Returns:
        ``(energies, vectors)``; ``vectors[:, i]`` belongs to ``energies[i]``
        and is indexed like ``determinants``.
    """
    dets = sorted({int(d) for d in determinants})
    if not dets:
        raise ValueError("no determinants given")
    h1_so, h2_so = _spin_terms(h1, h2)
    terms = sub.fermionic_terms(h1_so, h2_so)
    energies, vectors = sub.ground_state(terms, dets, k=k, sparse=sparse)
    return energies, vectors


def sqd_energy(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    determinants: Sequence[int],
    *,
    k: int = 1,
    return_vector: bool = False,
    max_seniority: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample-based quantum diagonalization for an explicit determinant set.

    Args:
        h1, h2: Spin-free MO integrals (chemist layout).
        determinants: The determinant bitmasks to diagonalise in. Duplicates
            and ordering do not matter.
        k: Number of lowest eigenpairs to return.
        return_vector: Also return the ground-state vector.
        max_seniority: Drop determinants whose seniority exceeds this before
            diagonalising. Sampled pools routinely contain high-seniority
            determinants that carry no weight; removing them shrinks the matrix
            without moving the energy. ``None`` keeps every determinant.

    Returns:
        Dict with ``energy`` (lowest), ``energies`` (all ``k``),
        ``n_determinants``, ``determinants`` (the sorted, deduplicated list),
        ``n_dropped`` and optionally ``vector``.
    """
    dets = sorted({int(d) for d in determinants})
    if max_seniority is not None:
        from superfermion.chemistry.seniority import filter_by_seniority

        kept = filter_by_seniority(dets, max_seniority)
        if not kept:
            raise ValueError(
                f"max_seniority={max_seniority} removes all {len(dets)} "
                f"determinants"
            )
        n_dropped = len(dets) - len(kept)
        dets = kept
    else:
        n_dropped = 0
    energies, vectors = subspace_energy(h1, h2, dets, k=k)
    result: Dict[str, Any] = {
        "energy": float(energies[0]),
        "energies": np.asarray(energies, dtype=float),
        "n_determinants": len(dets),
        "n_dropped": n_dropped,
        "determinants": dets,
        "max_seniority": max_seniority,
    }
    if return_vector:
        result["vector"] = vectors[:, 0]
    return result


def sample_determinants(
    amplitudes: np.ndarray,
    determinants: Sequence[int],
    n_samples: int,
    *,
    seed: int = 0,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    n_orbitals: Optional[int] = None,
) -> Dict[str, Any]:
    """Draw ``n_samples`` determinants from ``|<d|psi>|^2``.

    Args:
        amplitudes: Coefficient per entry of ``determinants``.
        determinants: Pool the amplitudes index into.
        n_samples: Number of draws *with* replacement, as on hardware.
        seed: RNG seed so a reported energy is reproducible.
        n_alpha, n_beta, n_orbitals: When given, samples whose electron count
            or spin projection is wrong are dropped and counted, which is what
            post-selection does on real devices. Without them every draw is
            kept and the caller is responsible for the sector.

    Returns:
        Dict with ``determinants`` (unique, sorted), ``counts`` (draws per
        unique determinant), ``n_unique``, ``n_discarded`` and
        ``n_requested``.
    """
    amps = np.asarray(amplitudes)
    dets = [int(d) for d in determinants]
    if amps.shape[0] != len(dets):
        raise ValueError(
            f"{amps.shape[0]} amplitudes for {len(dets)} determinants"
        )
    weights = np.abs(amps) ** 2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("the amplitude vector is zero; nothing to sample")
    probabilities = weights / total

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dets), size=int(n_samples), p=probabilities)

    counts: Dict[int, int] = {}
    discarded = 0
    for index in picks:
        det = dets[int(index)]
        if n_alpha is not None and n_beta is not None and n_orbitals is not None:
            alpha = sum(1 for p in range(n_orbitals) if (det >> (2 * p)) & 1)
            beta = sum(1 for p in range(n_orbitals) if (det >> (2 * p + 1)) & 1)
            if alpha != n_alpha or beta != n_beta:
                discarded += 1
                continue
        counts[det] = counts.get(det, 0) + 1

    return {
        "determinants": sorted(counts),
        "counts": counts,
        "n_unique": len(counts),
        "n_discarded": discarded,
        "n_requested": int(n_samples),
    }


def qsci(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    n_orbitals: int,
    n_electrons: int,
    *,
    n_alpha: int,
    n_beta: int,
    n_select: int,
    max_iter: int = 8,
    energy_tol: float = 1e-10,
    reference: Optional[Sequence[int]] = None,
    pool: Optional[Sequence[int]] = None,
    pool_limit: int = _DEFAULT_POOL_LIMIT,
    max_seniority: Optional[int] = None,
) -> Dict[str, Any]:
    """Quantum-selected CI with coupling-based selection.

    Starting from a reference determinant, each iteration

    1. projects the Hamiltonian onto the current determinant set,
    2. diagonalises it,
    3. ranks every determinant in the pool by ``|<d|H|psi>|`` and keeps the
       ``n_select`` strongest, always retaining those already selected.

    Args:
        h1, h2: Spin-free MO integrals.
        n_orbitals, n_electrons: Size of the CI space.
        n_alpha, n_beta: Spin projection that defines the pool.
        n_select: Determinants to keep. At ``n_select >= len(pool)`` the
            method is exact and the result equals full CI.
        max_iter: Iteration cap.
        energy_tol: Stop when the energy changes by less than this.
        reference: Starting determinant (bitmask). Defaults to the
            Hartree-Fock occupation of the given spin split.
        pool: Determinant pool to select from. Defaults to every determinant
            with ``(n_alpha, n_beta)``.
        pool_limit: Refuse to enumerate a pool larger than this.
        max_seniority: Build the default pool from determinants whose seniority
            is at most this, which is what makes the pool affordable for a
            large active space. Ignored when ``pool`` is given explicitly, so
            a caller who supplies a pool sees exactly that pool.

    Returns:
        Dict with ``energy``, ``determinants``, ``history`` (energy per
        iteration), ``iterations``, ``pool_size``, ``max_seniority`` and
        ``exact`` (whether the selected set covers the pool).
    """
    if pool is None:
        size = _binomial(n_orbitals, n_alpha) * _binomial(n_orbitals, n_beta)
        if size > pool_limit:
            raise ValueError(
                f"the (n_alpha={n_alpha}, n_beta={n_beta}) pool has {size} "
                f"determinants, above pool_limit={pool_limit}; pass an "
                f"explicit pool= or a smaller active space"
            )
        if max_seniority is None:
            pool_dets = sub.all_determinants(
                n_orbitals, n_electrons, n_alpha=n_alpha, n_beta=n_beta
            )
        else:
            from superfermion.chemistry.seniority import filter_by_seniority

            pool_dets = filter_by_seniority(
                sub.all_determinants(
                    n_orbitals, n_electrons, n_alpha=n_alpha, n_beta=n_beta
                ),
                max_seniority,
            )
            if not pool_dets:
                raise ValueError(
                    f"max_seniority={max_seniority} empties the "
                    f"(n_alpha={n_alpha}, n_beta={n_beta}) pool"
                )
    else:
        pool_dets = sorted({int(d) for d in pool})

    if reference is None:
        start = 0
        for p in range(n_alpha):
            start |= 1 << (2 * p)
        for p in range(n_beta):
            start |= 1 << (2 * p + 1)
    else:
        start = int(reference)

    h1_so, h2_so = _spin_terms(h1, h2)
    terms = sub.fermionic_terms(h1_so, h2_so)
    full = sub.hamiltonian_matrix(terms, pool_dets, sparse=True)

    selected = [start] if start in pool_dets else [pool_dets[0]]
    n_select = min(int(n_select), len(pool_dets))
    pool_index = {det: i for i, det in enumerate(pool_dets)}
    history: List[float] = []

    # Seed the first iteration with the reference and its strongest direct
    # couplings.  Beginning from a single determinant would ask for the
    # ground state of a 1x1 operator and spend an iteration that carries no
    # correlation at all.
    if 1 < n_select:
        seed_column = np.zeros(len(pool_dets), dtype=complex)
        seed_column[pool_index[selected[0]]] = 1.0
        chosen = set(selected)
        for i in np.argsort(-np.abs(full @ seed_column)):
            if len(chosen) >= n_select:
                break
            chosen.add(pool_dets[int(i)])
        selected = sorted(chosen)

    energy = None
    for _ in range(max_iter):
        energies, vectors = sub.ground_state(terms, selected, sparse=True, k=1)
        energy = float(energies[0])
        history.append(energy)

        # <d|H|psi> over the whole pool: one sparse matvec, not a loop.
        column = np.zeros(len(pool_dets), dtype=complex)
        for local, det in enumerate(selected):
            column[pool_index[det]] = vectors[local, 0]
        scores = np.abs(full @ column)

        chosen = set(selected)
        order = np.argsort(-scores)
        for i in order:
            if len(chosen) >= n_select:
                break
            chosen.add(pool_dets[int(i)])
        new_selected = sorted(chosen)
        if new_selected == selected:
            break
        selected = new_selected

    return {
        "energy": float(energy),
        "determinants": selected,
        "history": history,
        "iterations": len(history),
        "pool_size": len(pool_dets),
        "max_seniority": max_seniority,
        "exact": len(selected) == len(pool_dets),
    }


def _binomial(n: int, k: int) -> int:
    from math import comb

    return comb(n, k) if 0 <= k <= n else 0
