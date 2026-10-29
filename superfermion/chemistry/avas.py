"""AVAS: atomic-valence active space selection by projection.

Sayfutyarova, Sun, Chan, Knizia and Chan, *JCTC* 13, 4063 (2017).

The idea is to let chemistry pick the space instead of a window.  Choose a
set of atomic orbitals that *should* matter (the metal 3d shell, the O 2p
shell, the pi system), project them onto the molecular orbitals, and keep the
MOs that carry most of that projection.

For a set ``T`` of target AOs the projection is

    M = (C^T S X)(C^T S X)^T,      X[T, :] = S_T^{-1/2}

a Hermitian matrix in the MO basis with eigenvalues in ``[0, 1]``.  Two
properties make it trustworthy rather than just plausible:

* ``trace(M) = |T|``, because ``X`` is ``S``-orthonormal and ``C`` is square.
  So the eigenvalues *partition* the target-space dimension: anything not
  selected leaked somewhere else, which is exactly the information a
  threshold needs.
* ``0 <= M <= 1``, so ``M`` really is a projector.

The ``S_T^{-1/2}`` is not decoration.  The common short form
``M_ij = sum_{mu,nu in T} C_{mu i} S_{mu nu} C_{nu j}`` is only a projector
when the target AOs are already orthonormal; on the O ``2p`` shell of water it
produces eigenvalues up to ``1.34`` and ``trace(M) = 3.51`` for a 3-dimensional
target, which is both unphysical and enough to select the wrong orbitals.  The
orthogonalised form satisfies ``trace(M) = |T|`` to machine precision, and the
tests assert that identity.
* ``M`` is block-diagonal in the occupied/virtual split only if the target
  space is, so the rotation is applied inside each block separately.  That
  keeps the electron count well defined -- rotating an occupied orbital into
  a virtual one would change the determinant without telling anyone.

The result composes with the existing active-space machinery through
``active_orbitals=(n_elec, n_orb)``: the rotated MOs are ordered as
``[core][selected occupied][selected virtual][unselected virtual]``, so the
AVAS space is a contiguous window starting at ``n_core``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from superfermion.chemistry.pyscf_bridge import MolecularData

__all__ = [
    "projector_metric",
    "avas_rotation",
    "rotate_mo_integrals",
    "avas_active_space",
]


def _as_index_list(target_aos: Sequence[int]) -> List[int]:
    idx = sorted({int(i) for i in target_aos})
    if not idx:
        raise ValueError("target_aos is empty; AVAS needs at least one AO")
    if idx[0] < 0:
        raise ValueError(f"target_aos must be non-negative, got {idx[0]}")
    return idx


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Symmetric ``A^{-1/2}``, refusing a singular block.

    A linearly dependent target set (the same shell listed twice, or a
    contracted function duplicated by symmetry) makes the projection
    ill-defined, so it is an error rather than a least-squares fallback.
    """
    values, vectors = np.linalg.eigh(matrix)
    if values.min() <= 1e-10:
        raise ValueError(
            "the target AO block of the overlap matrix is singular "
            f"(smallest eigenvalue {values.min():.3e}); the target set is "
            "linearly dependent"
        )
    return (vectors * values**-0.5) @ vectors.T


def projector_metric(
    mo_coeff: np.ndarray,
    overlap: np.ndarray,
    target_aos: Sequence[int],
) -> np.ndarray:
    """Projector of the (S-orthonormalised) target AOs in the MO basis.

    Args:
        mo_coeff: ``(n_ao, n_mo)`` MO coefficients, ``C[:, i]`` for MO ``i``.
        overlap: ``(n_ao, n_ao)`` AO overlap matrix ``S``.
        target_aos: AO indices defining the chemically interesting set.

    Returns:
        Hermitian ``(n_mo, n_mo)`` matrix ``M`` with eigenvalues in ``[0, 1]``.
        ``trace(M) == len(target_aos)`` when the MOs span the target AOs --
        true whenever there are as many MOs as AOs, which is the case for a
        converged SCF coefficient matrix.  A truncated (``n_mo < n_ao``) set
        reaches only part of the target space and the trace falls short; that
        is a property of the input, not of the projection.
    """
    C = np.asarray(mo_coeff, dtype=float)
    S = np.asarray(overlap, dtype=float)
    idx = _as_index_list(target_aos)

    if C.ndim != 2 or S.shape != (C.shape[0], C.shape[0]):
        raise ValueError(
            f"shape mismatch: mo_coeff {C.shape}, overlap {S.shape}; "
            f"expected ({S.shape[0]}, n_mo)"
        )
    if idx[-1] >= C.shape[0]:
        raise ValueError(
            f"target AO {idx[-1]} is out of range for {C.shape[0]} AOs"
        )

    n_ao = C.shape[0]
    X = np.zeros((n_ao, len(idx)))
    X[idx, :] = _inverse_sqrt(S[np.ix_(idx, idx)])

    projected = C.T @ (S @ X)          # (n_mo, |T|)
    metric = projected @ projected.T
    return 0.5 * (metric + metric.T)


def avas_rotation(
    mo_coeff: np.ndarray,
    overlap: np.ndarray,
    target_aos: Sequence[int],
    *,
    n_occupied: Optional[int] = None,
    threshold: float = 0.2,
    max_orbitals: Optional[int] = None,
) -> Dict[str, object]:
    """Rotate the MOs so the AVAS-selected orbitals come first.

    Args:
        mo_coeff: ``(n_ao, n_mo)`` MO coefficients.
        overlap: ``(n_ao, n_ao)`` AO overlap.
        target_aos: AO indices to project.
        n_occupied: Occupied MO count. When given, the rotation is applied to
            the occupied and virtual blocks separately so the electron count
            is preserved; when ``None`` the whole MO set is rotated as one.
        threshold: Keep orbitals whose projector eigenvalue exceeds this.
            ``0.2`` is the value recommended in the original paper for a
            valence target set; ``0.5`` gives a tighter, more literal space.
        max_orbitals: Optional cap on the number of selected orbitals per
            block, highest eigenvalue first.

    Returns:
        Dict with

        ``mo_coeff``
            ``(n_ao, n_mo)`` rotated coefficients, new column order as
            described in the module docstring.
        ``n_core``, ``n_active``, ``n_active_electrons``
            Sizes of the resulting contiguous active window.
        ``active_indices``
            ``range(n_core, n_core + n_active)`` as a tuple.
        ``eigenvalues``
            Projector eigenvalues, descending, kept-before-dropped per block.
        ``target_dimension``
            ``trace(M)``, i.e. the number of target AOs. ``sum(eigenvalues)``
            must equal it; that identity is the sanity check on ``M``.
        ``selected_occupied``, ``selected_virtual``
            Counts per block before the window is assembled.
    """
    C = np.asarray(mo_coeff, dtype=float)
    n_mo = C.shape[1]
    if n_occupied is None:
        n_occupied = n_mo
    if not 0 <= n_occupied <= n_mo:
        raise ValueError(
            f"n_occupied must be in [0, {n_mo}], got {n_occupied}"
        )

    metric = projector_metric(C, overlap, target_aos)
    blocks = [(0, n_occupied), (n_occupied, n_mo)] if 0 < n_occupied < n_mo else [(0, n_mo)]

    rotation = np.zeros((n_mo, n_mo))
    eigenvalues: List[float] = []
    counts: List[int] = []
    for lo, hi in blocks:
        values, vectors = np.linalg.eigh(metric[lo:hi, lo:hi])
        order = np.argsort(-values)
        values = values[order]
        vectors = vectors[:, order]
        rotation[lo:hi, lo:hi] = vectors
        eigenvalues.extend(float(v) for v in values)
        keep = int(np.count_nonzero(values > threshold))
        if max_orbitals is not None:
            keep = min(keep, int(max_orbitals))
        counts.append(keep)

    selected_occupied = counts[0] if len(counts) > 0 else 0
    selected_virtual = counts[1] if len(counts) > 1 else 0

    # Within each block the eigenvectors are already sorted keep-first, so
    # only the block order has to change: core, selected occupied, selected
    # virtual, the rest.
    core = list(range(selected_occupied, n_occupied))
    sel_occ = list(range(0, selected_occupied))
    sel_vir = list(range(n_occupied, n_occupied + selected_virtual))
    rest_vir = list(range(n_occupied + selected_virtual, n_mo))
    perm = core + sel_occ + sel_vir + rest_vir

    mo_rot = (C @ rotation)[:, perm]
    n_core = len(core)
    n_active = selected_occupied + selected_virtual

    return {
        "mo_coeff": mo_rot,
        "transform": rotation[:, perm],
        "n_core": n_core,
        "n_active": n_active,
        "n_active_electrons": 2 * selected_occupied,
        "active_indices": tuple(range(n_core, n_core + n_active)),
        "eigenvalues": np.array(eigenvalues, dtype=float),
        "target_dimension": float(np.trace(metric)),
        "selected_occupied": selected_occupied,
        "selected_virtual": selected_virtual,
        "permutation": perm,
    }


def rotate_mo_integrals(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    transform: np.ndarray,
) -> tuple:
    """Rotate MO integrals by ``transform``: ``C_new = C_old @ transform``.

    ``h1`` becomes ``T^T h1 T`` and ``h2[i,j,k,l]`` becomes
    ``sum T_pi T_qj T_rk T_sl h2[p,q,r,s]``.  Exactly one convention is used
    here (the one that matches ``_compute_pyscf``), so a rotated
    :class:`MolecularData` stays in the same layout it started in.
    """
    T = np.asarray(transform, dtype=float)
    h1 = np.asarray(h1, dtype=float)
    h1_new = T.T @ h1 @ T
    if h2 is None:
        return h1_new, None
    h2 = np.asarray(h2, dtype=float)
    h2_new = np.einsum("pi,qj,rk,sl,pqrs->ijkl", T, T, T, T, h2, optimize=True)
    return h1_new, h2_new


def _resolve_target_aos(
    mol: MolecularData,
    target_aos: Optional[Sequence[int]],
    target_atoms: Optional[Sequence[object]],
    target_shells: Optional[Sequence[str]],
) -> List[int]:
    """Turn a chemical description ("O 2p", "all d AOs") into AO indices."""
    if target_aos is not None:
        return _as_index_list(target_aos)
    if mol.ao_labels is None:
        raise ValueError(
            "no AO labels on this MolecularData (it was built without PySCF); "
            "pass target_aos= explicitly"
        )

    atoms: Optional[set] = None
    if target_atoms is not None:
        atoms = set()
        for entry in target_atoms:
            if isinstance(entry, (int, np.integer)):
                atoms.add(("index", int(entry)))
            else:
                atoms.add(("symbol", str(entry).capitalize()))

    shells = None if target_shells is None else [s.lower() for s in target_shells]

    selected = []
    for ao, label in enumerate(mol.ao_labels):
        parts = label.split()
        if len(parts) < 3:
            continue
        atom_index, symbol, shell = int(parts[0]), parts[1], parts[2].lower()
        if atoms is not None:
            if ("index", atom_index) not in atoms and ("symbol", symbol) not in atoms:
                continue
        if shells is not None and not any(shell.startswith(s) for s in shells):
            continue
        selected.append(ao)

    if not selected:
        raise ValueError(
            f"no AO matched target_atoms={target_atoms!r} "
            f"target_shells={target_shells!r}"
        )
    return selected


def avas_active_space(
    mol: MolecularData,
    target_aos: Optional[Sequence[int]] = None,
    *,
    target_atoms: Optional[Sequence[object]] = None,
    target_shells: Optional[Sequence[str]] = None,
    threshold: float = 0.2,
    max_orbitals: Optional[int] = None,
) -> MolecularData:
    """Run AVAS and return a new :class:`MolecularData` in the AVAS basis.

    The returned object carries rotated ``h1``/``h2`` and
    ``active_orbitals=(n_active_electrons, n_active)``, so
    ``result.to_hamiltonian()`` builds exactly the AVAS space.  The original
    object is not modified.

    Args:
        mol: A PySCF-built ``MolecularData`` (needs ``_mo_coeffs``,
            ``_ao_overlap`` and integrals).
        target_aos: Explicit AO indices. Overrides the chemical selectors.
        target_atoms: Atom symbols (``["O"]``) or indices (``[0]``).
        target_shells: Shell prefixes, e.g. ``["3d"]`` or ``["2p", "1s"]``.
        threshold: Projector-eigenvalue threshold.
        max_orbitals: Cap per occupied/virtual block.

    Returns:
        New ``MolecularData`` with the AVAS window applied. ``frozen_core``
        records the resulting core size for reporting.

    Raises:
        ValueError: when the molecule has no MO coefficients, no overlap
            matrix, or the selection is empty.
    """
    if mol._mo_coeffs is None or mol._ao_overlap is None:
        raise ValueError(
            "AVAS needs MO coefficients and the AO overlap; build the "
            "molecule with molecule_from_geometry(..., ) on a machine with "
            "PySCF, or supply integrals another way"
        )
    if mol.h1 is None:
        raise ValueError("AVAS needs MO integrals (h1 is None)")

    aos = _resolve_target_aos(mol, target_aos, target_atoms, target_shells)
    n_occupied = mol.n_electrons // 2

    info = avas_rotation(
        mol._mo_coeffs,
        mol._ao_overlap,
        aos,
        n_occupied=n_occupied,
        threshold=threshold,
        max_orbitals=max_orbitals,
    )
    if info["n_active"] == 0:
        raise ValueError(
            f"AVAS selected no orbitals at threshold={threshold}; "
            f"eigenvalues were {np.round(info['eigenvalues'], 4).tolist()}"
        )

    h1_new, h2_new = rotate_mo_integrals(mol.h1, mol.h2, info["transform"])

    return replace(
        mol,
        h1=h1_new,
        h2=h2_new,
        _mo_coeffs=np.asarray(info["mo_coeff"], dtype=float),
        active_orbitals=(int(info["n_active_electrons"]), int(info["n_active"])),
        frozen_core=int(info["n_core"]),
        active_indices=None,
        removed_orbitals=None,
        avas_info={k: v for k, v in info.items() if k != "mo_coeff"},
    )
