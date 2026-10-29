"""One-shot DMET: solve impurity problems instead of the whole molecule.

Wouters, Jiménez-Hoyos, Sun and Chan, *JCTC* 12, 2706 (2016).

Density matrix embedding theory partitions the orbitals into fragments and
replaces everything outside a fragment by a *bath* of at most as many
orbitals as the fragment itself.  The bath is not a guess: given the 1-RDM
``D`` of the reference state, the environment block ``D_FE`` is decomposed as

    D_FE = U s V^T,      bath = columns of V with s_i > 0

so the fragment plus bath carries exactly the fragment's share of the
entanglement, and the impurity problem is at most ``2 * |fragment|`` orbitals
wide.  The remaining environment orbitals are frozen into a mean field folded
into the impurity one-body Hamiltonian.

The total energy uses the standard correction

    E_DMET = E_HF + sum_f (E_imp_f - E_ref_f)

which is exact when a fragment's impurity space covers the whole system, and
in that limit the tests assert equality with full CI.

Honest limits, stated rather than buried:

* one shot, no self-consistency and no chemical-potential fitting, so the
  fragment densities are not required to sum to the exact electron count;
* the bath comes from a determinant density unless a correlated ``density``
  is supplied, which is the standard DMET-HF choice;
* ``E_ref_f`` is the expectation of the impurity Hamiltonian in the
  *reference determinant*, taken from the impurity block of its 1-RDM.  For a
  determinant the two-body contraction is fixed by that block
  (``Gamma = rho ^ rho``), so no SCF is needed and the one-fragment case is
  exact rather than merely close -- an earlier revision diagonalised the
  impurity Fock matrix once and called that the reference, which for
  localised H2 left it 13 mHa above Hartree-Fock.
* the impurity one-body part is ``h + F_core`` and the reference must be taken
  with that *same* one-body part.  An earlier revision evaluated the reference
  with the bare ``h``, which drops ``tr(rho_imp F_core)``; for water with three
  fragments that missing term is about 20 Ha per fragment and turned a
  30 mHa correlation correction into a +59 Ha error.
* the core density comes from the reference 1-RDM, not from an assumed
  ``2 * I``.  The two agree when the fragment partition leaves the core fully
  occupied or empty -- which the SVD bath enforces -- but not for a partition
  whose core still contains virtuals, and there the assumed ``2 * I`` puts
  unoccupied orbitals into the core field.
* when ``rho_imp`` is a ``{0, 2}`` projector the reference is a single
  determinant at the impurity electron count, so the correction is a genuine
  correlation energy and is negative by construction.  For a fragment whose
  impurity block has fractional occupations that no longer holds and the sign
  is not guaranteed.

A correlated ``density`` is accepted, and then the bath is the Schmidt
decomposition of that correlated state -- but note the energy functional is
unchanged, so the result is no longer variational.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from superfermion.chemistry import subspace as sub
from superfermion.chemistry.pyscf_bridge import MolecularData

__all__ = ["embedding_basis", "one_shot_dmet"]


def embedding_basis(
    density: np.ndarray,
    fragment: Sequence[int],
    *,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Fragment + bath + core orbitals from the 1-RDM.

    Args:
        density: ``(n, n)`` symmetric spin-summed 1-RDM in the current basis.
        fragment: Orbital indices that form the fragment.
        tol: Singular values below this contribute no bath orbital.

    Returns:
        Dict with ``transform`` (``(n, n)`` orthogonal, columns ordered
        fragment, bath, core), ``impurity`` and ``core`` (index lists),
        ``n_bath`` and ``bath_occupations`` (the Schmidt singular values).

    Raises:
        ValueError: on a malformed density or an empty/complete fragment.
    """
    D = np.asarray(density, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"density must be square, got {D.shape}")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("density must be symmetric")

    frag = sorted({int(i) for i in fragment})
    if not frag:
        raise ValueError("fragment is empty")
    if frag[0] < 0 or frag[-1] >= n:
        raise ValueError(f"fragment index out of range for {n} orbitals")
    env = [i for i in range(n) if i not in set(frag)]

    transform = np.zeros((n, n))
    transform[frag, : len(frag)] = np.eye(len(frag))

    if not env:
        return {
            "transform": transform,
            "impurity": list(range(len(frag))),
            "core": [],
            "n_bath": 0,
            "bath_occupations": np.zeros(0),
        }

    block = D[np.ix_(frag, env)]
    _, singular, vt = np.linalg.svd(block)
    keep = int(np.sum(singular > tol))
    bath = vt[:keep].T if keep else np.zeros((len(env), 0))

    # Complete the bath to a full environment basis. QR keeps the first
    # `keep` columns spanning exactly the bath, so the fragment+bath block is
    # untouched by the completion.
    q, _ = np.linalg.qr(np.hstack([bath, np.eye(len(env))]))
    transform[env, len(frag):] = q[:, : len(env)]

    return {
        "transform": transform,
        "impurity": list(range(len(frag) + keep)),
        "core": list(range(len(frag) + keep, n)),
        "n_bath": keep,
        "bath_occupations": singular[:keep],
    }


def _rotate(h1: np.ndarray, h2: Optional[np.ndarray], T: np.ndarray):
    h1_new = T.T @ np.asarray(h1, dtype=float) @ T
    if h2 is None:
        return h1_new, None
    h2_new = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl", T, T, T, T, np.asarray(h2, dtype=float),
        optimize=True,
    )
    return h1_new, h2_new


def _carrier(h1: np.ndarray, h2: Optional[np.ndarray]) -> MolecularData:
    """Throwaway MolecularData so the shared slicing code can be reused."""
    n = np.asarray(h1).shape[0]
    return MolecularData(
        name="impurity", geometry="", h1=np.asarray(h1, dtype=float),
        h2=None if h2 is None else np.asarray(h2, dtype=float),
        n_orbitals=n, n_electrons=0,
    )


def _determinant_energy(
    h1: np.ndarray, h2: Optional[np.ndarray], occupied: Sequence[int]
) -> float:
    """Energy of the determinant that doubly occupies ``occupied``."""
    occupied = [int(i) for i in occupied]
    n = np.asarray(h1).shape[0]
    virtual = [i for i in range(n) if i not in set(occupied)]
    _, energy = _carrier(h1, h2).reference_inactive_fock(occupied, virtual)
    return float(energy)


def one_shot_dmet(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    n_electrons: int,
    fragments: Sequence[Sequence[int]],
    *,
    nuclear_repulsion: float = 0.0,
    orbitals: Optional[np.ndarray] = None,
    density: Optional[np.ndarray] = None,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Run one-shot DMET over a partition into fragments.

    Args:
        h1, h2: Spin-free MO integrals in chemist order.
        n_electrons: Total electron count (closed shell assumed).
        fragments: One orbital-index list per fragment. The impurity problem
            is solved exactly for each.
        nuclear_repulsion: Added once to the final energy.
        orbitals: Optional ``(n, n)`` orthogonal matrix that re-expresses the
            integrals in a fragment-local basis before partitioning. Without
            it, fragments are canonical MO indices, where the density is
            diagonal and no bath orbital is generated -- genuine fragments
            normally need this.
        density: Optional ``(n, n)`` 1-RDM in the fragment basis. Defaults to
            the reference determinant's, i.e. ``2 U^T P_occ U``.
        tol: Bath singular-value threshold.

    Returns:
        Dict with ``energy``, ``energy_electronic``, ``hf_energy``,
        ``fragments`` (per-fragment dicts: ``n_bath``, ``n_impurity``,
        ``n_impurity_electrons``, ``impurity_occupation``,
        ``energy_impurity``, ``energy_reference``, ``correlation``) and
        ``correlation_total``.

    Raises:
        ValueError: when a fragment impurity has an odd electron count, which
            a closed-shell reference cannot represent.
    """
    h1 = np.asarray(h1, dtype=float)
    n = h1.shape[0]
    n_occ = int(n_electrons) // 2
    if 2 * n_occ != int(n_electrons):
        raise ValueError(
            f"one_shot_dmet assumes a closed-shell reference; got "
            f"{n_electrons} electrons"
        )
    if n_occ > n:
        raise ValueError(f"{n_occ} occupied orbitals do not fit in {n}")

    U = np.eye(n) if orbitals is None else np.asarray(orbitals, dtype=float)
    if U.shape != (n, n):
        raise ValueError(f"orbitals must be ({n}, {n}), got {U.shape}")
    if not np.allclose(U.T @ U, np.eye(n), atol=1e-8):
        raise ValueError("orbitals must be orthogonal")

    h1_loc, h2_loc = _rotate(h1, h2, U)
    if density is None:
        # The reference determinant expressed in the local basis: its occupied
        # orbitals are ``U^T |i>`` for i < n_occ, which are the first n_occ
        # *columns of U^T*, not the first n_occ columns of U.  The latter is a
        # different state, and every fragment then describes the wrong
        # electron distribution.
        occupied = U.T[:, :n_occ]
        D = 2.0 * occupied @ occupied.T
    else:
        D = np.asarray(density, dtype=float)
        if D.shape != (n, n):
            raise ValueError(f"density must be ({n}, {n}), got {D.shape}")

    hf_elec = _determinant_energy(h1, h2, range(n_occ))

    details: List[Dict[str, Any]] = []
    correction = 0.0
    for fragment in fragments:
        basis = embedding_basis(D, fragment, tol=tol)
        T = basis["transform"]
        h1_emb, h2_emb = _rotate(h1_loc, h2_loc, T)

        impurity = list(basis["impurity"])
        core = list(basis["core"])
        D_emb = T.T @ D @ T
        n_imp_elec = int(round(float(np.trace(D_emb[np.ix_(impurity, impurity)]))))
        if n_imp_elec % 2:
            raise ValueError(
                f"fragment {list(fragment)} gives an odd impurity electron "
                f"count ({n_imp_elec}); a closed-shell impurity cannot "
                f"represent it"
            )

        carrier = _carrier(h1_emb, h2_emb)
        # Core potential and energy from the reference density of the core
        # block.  ``reference_inactive_fock`` defaults to ``2 * I``, which is
        # only right when every core orbital is doubly occupied.
        D_core = D_emb[np.ix_(core, core)]
        fock, e_core = carrier.reference_inactive_fock(
            core, impurity, density=D_core if core else None
        )

        h1_imp = h1_emb[np.ix_(impurity, impurity)] + fock
        h2_imp = (
            None
            if h2_emb is None
            else h2_emb[np.ix_(impurity, impurity, impurity, impurity)]
        )
        n_imp_orb = len(impurity)

        energy_impurity = e_core + sub.exact_energy(
            h1_imp, h2_imp, n_imp_orb, n_imp_elec
        )

        # Reference: the expectation of the impurity Hamiltonian in the
        # reference state.  A determinant's two-body energy is fixed by its
        # 1-RDM (Gamma = rho ^ rho) and the impurity block of rho contains
        # every index that contraction needs, so this is exact and needs no
        # SCF.  Diagonalising the impurity Fock matrix once instead -- as an
        # earlier revision did -- is one SCF step from an arbitrary basis, not
        # the reference.
        #
        # The impurity Hamiltonian carries the core field, so the reference
        # must be evaluated with it too.  ``reference_inactive_fock`` called
        # with an empty active set returns only ``tr(rho h)`` plus the
        # two-body part from the carrier's bare ``h1``; the missing
        # ``tr(rho_imp F_core)`` is the single largest term in the correction.
        D_imp = D_emb[np.ix_(impurity, impurity)]
        _, energy_reference = carrier.reference_inactive_fock(
            impurity, [], density=D_imp
        )
        energy_reference += float(np.einsum("pq,pq->", D_imp, fock, optimize=True))
        energy_reference += e_core

        correction += energy_impurity - energy_reference
        details.append({
            "fragment": [int(i) for i in fragment],
            "n_bath": int(basis["n_bath"]),
            "n_impurity": n_imp_orb,
            "n_impurity_electrons": n_imp_elec,
            "impurity_occupation": float(np.trace(D_imp)),
            "energy_impurity": float(energy_impurity),
            "energy_reference": float(energy_reference),
            "correlation": float(energy_impurity - energy_reference),
        })

    energy_electronic = hf_elec + correction
    return {
        "energy": float(energy_electronic + nuclear_repulsion),
        "energy_electronic": float(energy_electronic),
        "hf_energy": float(hf_elec + nuclear_repulsion),
        "fragments": details,
        "correlation_total": float(correction),
    }
