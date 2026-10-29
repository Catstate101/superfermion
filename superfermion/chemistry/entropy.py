"""Orbital-entropy diagnostics and AutoCAS-style active-space selection.

The single-orbital entropy ``s_i`` measures how entangled spatial orbital ``i``
is with the rest of the system, so it answers the question that a fixed
window cannot: *which* orbitals actually need correlating.  ``s_i = 0`` for a
determinant; ``s_i = ln 4`` for a maximally mixed orbital.

Reduced density matrices are obtained by reshaping the coefficient tensor and
contracting the modes that are traced out.  Keeping the modes in their natural
order is what makes this sign-free: fermionic signs are already inside the
amplitudes, and the partial trace of ``|psi><psi|`` is a linear operation on
the coefficient tensor.  Factorising into a different mode order is where this
computation usually goes wrong.

Conventions match :mod:`superfermion.chemistry.subspace`: a determinant is an
integer over spin orbitals in the interleaved order ``P = 2p + s``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "orbital_entropy",
    "single_orbital_entropy",
    "mutual_information",
    "autocas_selection",
    "entropies_from_integrals",
]


def _coefficient_tensor(
    vector: np.ndarray, dets: Sequence[int], n_qubits: int
) -> np.ndarray:
    """Dense amplitude array indexed by determinant bitmask."""
    psi = np.zeros(1 << n_qubits, dtype=complex)
    for amp, det in zip(np.asarray(vector, dtype=complex), dets):
        psi[int(det)] += amp
    # reshape [2]*n maps flat index to bits with axis 0 = most significant,
    # so mode p sits on axis (n - 1 - p).
    return psi.reshape([2] * n_qubits)


def _reduced_density_matrix(
    psi: np.ndarray, modes: Sequence[int], n_qubits: int
) -> np.ndarray:
    """Partial trace over everything except ``modes``."""
    modes = list(modes)
    keep = [n_qubits - 1 - p for p in modes]
    perm = keep + [i for i in range(n_qubits) if i not in keep]
    psi = psi.transpose(perm).reshape(2 ** len(modes), -1)
    return psi @ psi.conj().T


def _von_neumann(rho: np.ndarray, tol: float = 1e-12) -> float:
    eigenvalues = np.linalg.eigvalsh((rho + rho.conj().T) / 2).real
    eigenvalues = eigenvalues[eigenvalues > tol]
    if eigenvalues.size == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def orbital_entropy(
    vector: np.ndarray, dets: Sequence[int], n_orbitals: int
) -> np.ndarray:
    """Per-orbital entropy ``s_i`` for a state in a determinant basis.

    ``s_i`` is the von Neumann entropy of the 4x4 reduced density matrix of
    the two spin orbitals belonging to spatial orbital ``i``, so it lies in
    ``[0, ln 4]``.

    Args:
        vector: Coefficients, indexed like ``dets``.
        dets: Determinants (bitmask over ``2 * n_orbitals`` spin orbitals).
        n_orbitals: Spatial orbital count.

    Returns:
        Array of shape ``(n_orbitals,)``.
    """
    n_qubits = 2 * n_orbitals
    psi = _coefficient_tensor(vector, dets, n_qubits)
    return np.array(
        [
            _von_neumann(_reduced_density_matrix(psi, (2 * i, 2 * i + 1), n_qubits))
            for i in range(n_orbitals)
        ]
    )


# Alias kept for the name used in the literature and in the report.
single_orbital_entropy = orbital_entropy


def mutual_information(
    vector: np.ndarray, dets: Sequence[int], n_orbitals: int
) -> np.ndarray:
    """Two-orbital mutual information ``I_ij``.

    ``I_ij = 0.5 * (s_i + s_j - s_ij)`` where ``s_ij`` is the von Neumann
    entropy of the 16x16 reduced density matrix of the two orbitals.  It is
    the standard companion to ``s_i``: ``s_i`` says an orbital matters,
    ``I_ij`` says *what it is entangled with*.

    Returns:
        Symmetric ``(n_orbitals, n_orbitals)`` array, zero on the diagonal.
    """
    n_qubits = 2 * n_orbitals
    psi = _coefficient_tensor(vector, dets, n_qubits)
    s_i = orbital_entropy(vector, dets, n_orbitals)
    out = np.zeros((n_orbitals, n_orbitals), dtype=float)
    for i in range(n_orbitals):
        for j in range(i + 1, n_orbitals):
            rho_ij = _reduced_density_matrix(
                psi, (2 * i, 2 * i + 1, 2 * j, 2 * j + 1), n_qubits
            )
            s_ij = _von_neumann(rho_ij)
            value = 0.5 * (s_i[i] + s_i[j] - s_ij)
            out[i, j] = out[j, i] = max(0.0, value)
    return out


def autocas_selection(
    vector: np.ndarray,
    dets: Sequence[int],
    n_orbitals: int,
    *,
    threshold: float = 0.05,
    max_orbitals: Optional[int] = None,
    use_mutual_information: bool = False,
) -> List[int]:
    """Rank orbitals by entropy and keep those above ``threshold``.

    This is the AutoCAS rule of Stein and Reiher (*JCTC* 12, 1760 (2016)):
    an orbital belongs in the active space if ``s_i`` exceeds a threshold,
    with ``I_ij`` available as a cross-check that the selected set is
    connected rather than a collection of unrelated hot orbitals.

    Args:
        vector, dets, n_orbitals: As in :func:`orbital_entropy`.
        threshold: Keep orbitals with ``s_i > threshold``.
        max_orbitals: Optional cap; the highest-entropy orbitals win.
        use_mutual_information: Rank by ``sum_j I_ij`` instead of ``s_i``.
            The two rankings usually agree; they differ when one orbital is
            entangled with many weakly-coupled partners.

    Returns:
        Ascending orbital indices.
    """
    s_i = orbital_entropy(vector, dets, n_orbitals)
    if use_mutual_information:
        score = mutual_information(vector, dets, n_orbitals).sum(axis=1)
        cut = 0.0
    else:
        score = s_i
        cut = threshold

    order = np.argsort(-score)
    selected = [int(i) for i in order if score[i] > cut]
    if max_orbitals is not None:
        selected = selected[: int(max_orbitals)]
    return sorted(selected)


def entropies_from_integrals(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    n_orbitals: int,
    n_electrons: int,
    *,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact FCI in the given space, then entropies of its ground state.

    Convenience wrapper so a caller does not have to hold the determinant
    basis and the eigenvector separately.

    Returns:
        ``(s_i, mutual_information, energy)``.
    """
    from superfermion.chemistry import subspace as sub

    dets = sub.all_determinants(
        n_orbitals, n_electrons, n_alpha=n_alpha, n_beta=n_beta
    )
    h1 = np.asarray(h1, dtype=float)
    if h2 is not None:
        h1_so, h2_so = sub._spin_orbital_integrals(h1, np.asarray(h2, dtype=float))
    else:
        h1_so = np.zeros((2 * n_orbitals, 2 * n_orbitals))
        for s in (0, 1):
            h1_so[s::2, s::2] = h1
        h2_so = None

    terms = sub.fermionic_terms(h1_so, h2_so)
    energies, vectors = sub.ground_state(terms, dets, sparse=False)
    vector = vectors[:, 0]
    return (
        orbital_entropy(vector, dets, n_orbitals),
        mutual_information(vector, dets, n_orbitals),
        float(energies[0]),
    )
