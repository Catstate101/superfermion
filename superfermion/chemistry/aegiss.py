"""AEGISS: an AVAS-plus-entropy workflow for active-space selection.

Tarocco et al., arXiv:2508.10671.  The published method is a packaged,
semi-automated workflow rather than a new variational principle: a chemical
target set proposes a space (AVAS), a correlated density decides which of
those orbitals actually carry correlation (single-orbital entropy), and the
survivors are handed to the solver.

What this module implements is exactly that two-stage loop, with exact CI in
place of the reference's DMRG-lite density:

1. AVAS proposes a contiguous window from an AO target set
   (:func:`superfermion.chemistry.avas.avas_active_space`).
2. Exact CI in that window gives the 1-RDM and hence ``s_i``.
3. Orbitals with ``s_i`` below ``entropy_threshold`` are dropped -- but only
   when their average occupation is near ``0`` or ``2``.  An orbital with
   ``s_i ~ 0`` and ``n_i ~ 1`` is an unpaired electron, not a dead orbital;
   removing it would change the molecule, so that case is kept and reported.

Every number needed to check the claim is returned in the result dict: both
energies, all entropies, all occupations, and the orbital indices in the
*original* MO basis so the choice can be replayed against
``MolecularData.to_hamiltonian(active_indices=...)``.

Both energies are *total molecular energies* (nuclear repulsion included).
That is not a detail: running CI on the bare window integrals without the
inactive Fock potential and the inactive energy reports a number with no
relation to the molecule.  Measured on water/STO-3G with the O 2p window the
correct total is -74.9788 Ha; dropping both terms gives -23.6862 Ha, dropping
only the potential -95.1146 Ha, and dropping only the core energy -3.5503 Ha.
:func:`_fold_core` does the folding, and the pruned energy folds the removed
orbitals the same way rather than dropping their potential on the floor.

The refinement can only restrict the space, so ``E_pruned >= E_avas``
necessarily; a large gap is the signal that the entropy threshold is too
aggressive, and it is reported rather than hidden.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.chemistry import entropy as ent
from superfermion.chemistry import subspace as sub
from superfermion.chemistry.avas import avas_active_space
from superfermion.chemistry.pyscf_bridge import MolecularData

__all__ = ["aegiss_selection"]

# Dense diagonalisation stops being affordable above a few thousand
# determinants; past this the sparse path builds the same matrix and asks
# ARPACK for the lowest eigenpair.
_DENSE_LIMIT = 4000


def _fold_core(
    mol: MolecularData,
    active: Sequence[int],
    occupations: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Dress an active window with the inactive potential and core energy.

    Args:
        mol: Molecule whose ``h1``/``h2`` live in the basis being sliced.
        active: Spatial orbital indices of the window.
        occupations: Spin-summed occupation of every spatial orbital of
            ``mol``, in that basis.  Contracted against the inactive ones:
            exact for a closed-shell inactive set (entries ``0`` or ``2``),
            the standard mean-field approximation otherwise.

    Returns:
        ``(h1_eff, h2_eff, e_core)``.  A CI energy over these integrals plus
        ``mol.nuclear_repulsion + e_core`` is the total molecular energy.
    """
    h1 = np.asarray(mol.h1, dtype=float)
    h2 = None if mol.h2 is None else np.asarray(mol.h2, dtype=float)
    active = [int(i) for i in active]
    occupations = np.asarray(occupations, dtype=float)
    keep = set(active)
    inactive = [p for p in range(h1.shape[0]) if p not in keep]

    if inactive:
        fock, e_core = mol.reference_inactive_fock(
            inactive, active, density=np.diag(occupations[inactive])
        )
    else:
        fock, e_core = np.zeros((len(active), len(active)), dtype=float), 0.0

    h1_eff = h1[np.ix_(active, active)] + fock
    h2_eff = None if h2 is None else h2[np.ix_(active, active, active, active)]
    return h1_eff, h2_eff, float(e_core)


def _spin_split(spin: int, n_elec: int) -> Tuple[int, int]:
    delta = int(spin)
    if delta > n_elec or (n_elec - delta) % 2:
        raise ValueError(
            f"cannot split {n_elec} electrons into an (n_alpha, n_beta) pair "
            f"with 2S={delta}"
        )
    return (n_elec + delta) // 2, (n_elec - delta) // 2


def _exact_state(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    n_orbitals: int,
    n_electrons: int,
    n_alpha: int,
    n_beta: int,
) -> Tuple[float, np.ndarray, List[int]]:
    """Lowest eigenpair of the Hamiltonian in the full CI space of the window."""
    from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals

    h1 = np.asarray(h1, dtype=float)
    if h2 is not None:
        h1_so, h2_so = _spin_orbital_integrals(h1, np.asarray(h2, dtype=float))
    else:
        h1_so = np.zeros((2 * n_orbitals, 2 * n_orbitals))
        for s in (0, 1):
            h1_so[s::2, s::2] = h1
        h2_so = None

    terms = sub.fermionic_terms(h1_so, h2_so)
    dets = sub.all_determinants(
        n_orbitals, n_electrons, n_alpha=n_alpha, n_beta=n_beta
    )
    dense = len(dets) <= _DENSE_LIMIT
    energies, vectors = sub.ground_state(terms, dets, sparse=not dense, k=1)
    return float(energies[0]), vectors[:, 0], dets


def aegiss_selection(
    mol: MolecularData,
    *,
    target_aos: Optional[Sequence[int]] = None,
    target_atoms: Optional[Sequence[object]] = None,
    target_shells: Optional[Sequence[str]] = None,
    avas_threshold: float = 0.2,
    max_orbitals: Optional[int] = None,
    entropy_threshold: float = 0.05,
    occupation_tol: float = 0.05,
    active_space: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Propose a space with AVAS, then refine it with orbital entropies.

    Args:
        mol: PySCF-built ``MolecularData``.
        target_aos, target_atoms, target_shells: AVAS target set. One of them
            must be given unless ``active_space`` is passed instead.
        avas_threshold: Projector-eigenvalue threshold for AVAS.
        max_orbitals: Cap on selected orbitals per AVAS block.
        entropy_threshold: Drop orbitals with ``s_i`` at or below this.
        occupation_tol: Pruning additionally requires the average occupation
            to be within this distance of 0 or 2. Protects open-shell
            orbitals, whose entropy can be small through symmetry alone.
        active_space: Skip AVAS and refine this ``(n_elec, n_orb)`` window of
            ``mol`` directly.

    Returns:
        Dict with ``energy_avas``, ``energy_pruned`` (both total molecular
        energies), ``entropies``, ``occupations``, ``mutual_information``,
        ``kept``, ``pruned``, ``active_indices`` (original MO basis),
        ``n_active_electrons``, ``n_active``, ``n_core`` and ``avas`` (the raw
        AVAS info).

    Raises:
        ValueError: when neither a target set nor ``active_space`` is given,
            or when no orbital survives the refinement.
    """
    if active_space is not None:
        basis = mol
        n_elec, n_orb = int(active_space[0]), int(active_space[1])
        n_core = (mol.n_electrons - n_elec) // 2
        if n_core < 0:
            raise ValueError(
                f"active_space={active_space} has more electrons than the "
                f"molecule ({mol.n_electrons})"
            )
        if n_core + n_orb > mol.n_orbitals:
            raise ValueError(
                f"active_space={active_space} needs orbitals "
                f"[{n_core}, {n_core + n_orb}) but the molecule has "
                f"{mol.n_orbitals}"
            )
        window = list(range(n_core, n_core + n_orb))
        perm = list(range(mol.n_orbitals))
        avas_info: Optional[Dict[str, Any]] = None
    else:
        if target_aos is None and target_atoms is None and target_shells is None:
            raise ValueError(
                "give an AVAS target set (target_atoms / target_shells / "
                "target_aos) or an explicit active_space"
            )
        basis = avas_active_space(
            mol,
            target_aos=target_aos,
            target_atoms=target_atoms,
            target_shells=target_shells,
            threshold=avas_threshold,
            max_orbitals=max_orbitals,
        )
        avas_info = dict(basis.avas_info or {})
        n_elec, n_orb = basis.active_orbitals  # type: ignore[misc]
        n_core = int(basis.frozen_core)
        window = [int(i) for i in avas_info["active_indices"]]
        perm = [int(i) for i in avas_info["permutation"]]

    # Hartree-Fock occupation of every orbital in the sliced basis.  ``perm``
    # maps a basis index back to the original MO, and the SCF filling puts
    # alpha before beta, so original orbital p holds two electrons below
    # n_beta, one below n_alpha and none above.
    n_alpha_scf, n_beta_scf = _spin_split(mol.spin, mol.n_electrons)
    orbital_occupation = np.array(
        [
            2.0 if source < n_beta_scf else (1.0 if source < n_alpha_scf else 0.0)
            for source in perm
        ],
        dtype=float,
    )

    n_alpha, n_beta = _spin_split(mol.spin, n_elec)
    if n_alpha > n_orb:
        raise ValueError(
            f"active space holds {n_orb} orbitals but {n_alpha} alpha "
            f"electrons are needed for 2S={mol.spin}"
        )

    h1_as, h2_as, e_core_avas = _fold_core(basis, window, orbital_occupation)
    energy_window, vector, dets = _exact_state(
        h1_as, h2_as, n_orb, n_elec, n_alpha, n_beta
    )
    energy_avas = mol.nuclear_repulsion + e_core_avas + energy_window
    s_i = ent.orbital_entropy(vector, dets, n_orb)
    i_ij = ent.mutual_information(vector, dets, n_orb)

    rd = sub.rdm1(vector, dets, 2 * n_orb)
    occupations = np.array([float((rd[2 * i, 2 * i] + rd[2 * i + 1, 2 * i + 1]).real)
                            for i in range(n_orb)])

    removable = np.zeros(n_orb, dtype=bool)
    for i in range(n_orb):
        near_empty = abs(occupations[i]) <= occupation_tol
        near_full = abs(occupations[i] - 2.0) <= occupation_tol
        if s_i[i] <= entropy_threshold and (near_empty or near_full):
            removable[i] = True

    kept = [i for i in range(n_orb) if not removable[i]]
    pruned = [i for i in range(n_orb) if removable[i]]
    if not kept:
        raise ValueError(
            f"every orbital was pruned at entropy_threshold="
            f"{entropy_threshold}; lower the threshold"
        )

    # Removing a doubly occupied orbital takes its two electrons with it; an
    # empty one takes none. Anything else was excluded above.
    n_elec_pruned = n_elec - 2 * sum(1 for i in pruned if occupations[i] > 1.0)
    e_core_pruned = e_core_avas
    energy_pruned = energy_avas
    if pruned:
        # Fold the pruned orbitals into the core with the occupation the CI
        # gave them, rounded to the 0 / 2 the guard above allows.  Slicing the
        # window integrals a second time would drop their potential and their
        # two-electron energy instead.
        folded = orbital_occupation.copy()
        for i in pruned:
            folded[window[i]] = 2.0 if occupations[i] > 1.0 else 0.0
        h1_red, h2_red, e_core_pruned = _fold_core(
            basis, [window[i] for i in kept], folded
        )
        n_alpha_p, n_beta_p = _spin_split(mol.spin, n_elec_pruned)
        energy_pruned = mol.nuclear_repulsion + e_core_pruned + _exact_state(
            h1_red, h2_red, len(kept), n_elec_pruned, n_alpha_p, n_beta_p
        )[0]

    return {
        "energy_avas": energy_avas,
        "energy_pruned": energy_pruned,
        "energy_window": float(energy_window),
        "e_core_avas": float(e_core_avas),
        "e_core_pruned": float(e_core_pruned),
        "nuclear_repulsion": float(mol.nuclear_repulsion),
        "entropies": s_i,
        "occupations": occupations,
        "mutual_information": i_ij,
        "kept": kept,
        "pruned": pruned,
        "active_indices": tuple(perm[window[i]] for i in kept),
        "n_active_electrons": int(n_elec_pruned),
        "n_active": len(kept),
        "n_core": int(n_core),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "avas": avas_info,
    }
