"""Second-order Møller-Plesset: energy, 1-RDM, natural orbitals, MP2-NO spaces.

Everything here works from the integrals :class:`MolecularData` already stores
(``h1``, ``h2``, ``mo_energies``), not from a live PySCF mean field.  That
matters because the same code then runs on density-fitted integrals, on an
interacting-orbital Hamiltonian or on a dump read back from ``fcidump`` -- the
places where a driver object does not exist.

The closed-shell (restricted) case only.  Open-shell MP2 is a different method
(ROHF-MP2 has spin-orbital denominators that are not the semicanonical ones),
and silently running the closed-shell formula on a doublet would produce a
plausible number for the wrong quantity -- the failure mode this package has
already paid for twice.  :func:`mp2_correlation_energy` raises instead.

Three definitions matter and are easy to get wrong:

* **the amplification tensor** ``t_ij^ab = <ab||ij> / (eps_i + eps_j - eps_a -
  eps_b)`` with ``<ab||ij> = (ai|bj) - (aj|bi)`` in spin orbitals.  In the
  spatial closed-shell reduction the numerator is
  ``2(ia|jb) - (ib|ja)`` for the singlet coupling;
* **the correlation energy** ``E2 = 1/4 sum_ijab |t_ij^ab|^2 D_ij^ab`` over all
  (unordered) spin-orbital indices, which is why the prefactor is 1/4 and not
  1/2: the sum runs over both orderings of each pair;
* **the unrelaxed MP2 1-RDM** ``D = D_ref + dD`` with
  ``dD_ij = -1/2 sum_kab t_ik^ab t_jk^ab`` and
  ``dD_ab = +1/2 sum_ijc t_ij^ac t_ij^bc``.  ``tr(dD) = 0`` exactly, so the
  density still integrates to the electron count -- the cheapest check that the
  factors are right.

``mp2_natural_orbitals`` returns the *unrelaxed* natural orbitals: the MP2
density diagonalised without orbital relaxation (no CPHF / Z-vector step).  That
is the standard object used for active-space selection ("MP2-NO" CAS), and it is
stated here so nobody mistakes it for a relaxed density.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "mp2_amplitudes",
    "mp2_correlation_energy",
    "mp2_density",
    "mp2_natural_orbitals",
    "mp2_no_molecule",
    "active_space_from_mp2_no",
    "casci_energy",
    "mp2_no_activity",
    "mp2_no_rank",
]


def _spin_orbital_chemist(
    h1: np.ndarray, h2: Optional[np.ndarray], n: int
) -> tuple:
    """``h1``/``h2`` (chemist, spin-free) -> chemist spin-orbital tensors.

    Unlike :func:`superfermion.chemistry.pyscf_bridge._spin_orbital_integrals`,
    which returns the layout ``FermionicOperator.from_coeffs`` consumes, this
    keeps the plain chemist slot order ``(P Q | R S)``: MP2's amplitudes are
    written in that order and translating them would be one more convention to
    keep straight.
    """
    h1 = np.asarray(h1, dtype=float)
    h1_so = np.zeros((2 * n, 2 * n), dtype=float)
    for s in (0, 1):
        h1_so[s::2, s::2] = h1
    if h2 is None:
        return h1_so, None
    h2 = np.asarray(h2, dtype=float)
    g = np.zeros((2 * n,) * 4, dtype=float)
    # Chemist (P Q | R S) puts P,Q on electron 1 and R,S on electron 2, so the
    # spin constraint is spin(P) == spin(Q) and spin(R) == spin(S).  Filling
    # ``[a::2, b::2, a::2, b::2]`` instead -- the layout
    # ``FermionicOperator.from_coeffs`` needs -- pairs P with R, which is a
    # different tensor: every mixed-spin integral (a i | b j) with alpha on the
    # first electron and beta on the second comes out zero, and water/STO-3G
    # MP2 came out at -0.0106 Ha against PySCF's -0.0354 Ha before this.
    for a in (0, 1):
        for b in (0, 1):
            g[a::2, a::2, b::2, b::2] = h2
    return h1_so, g


def _occupied_lists(n: int, n_occupied: int, n_frozen: int) -> tuple:
    if n_occupied < 0 or n_occupied > n:
        raise ValueError(f"n_occupied={n_occupied} does not fit in {n} orbitals")
    if not (0 <= n_frozen <= n_occupied):
        raise ValueError(
            f"n_frozen={n_frozen} must lie in 0..n_occupied={n_occupied}"
        )
    core = list(range(n_frozen))
    occ = list(range(n_frozen, n_occupied))
    virt = list(range(n_occupied, n))
    return core, occ, virt


def mp2_amplitudes(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    mo_energies: np.ndarray,
    n_occupied: int,
    *,
    n_frozen: int = 0,
) -> Dict[str, Any]:
    """Singles-free double amplitudes ``t[i, j, a, b]``.

    Args:
        h1: ``(n, n)`` one-body integrals in the (canonical) MO basis.
        h2: ``(n, n, n, n)`` chemist ERIs, or ``None`` for a one-body-only model.
        mo_energies: ``(n,)`` canonical orbital energies, i.e. the Fock diagonal.
        n_occupied: Doubly occupied spatial orbitals.
        n_frozen: Lowest orbitals excluded from correlation (frozen core).

    Returns:
        Dict with ``amplitude`` ``(n_occ, n_occ, n_virt, n_virt)``, ``denominator``
        the same shape, ``occ``/``virt``/``core`` index lists and ``e2`` the
        correlation energy.
    """
    h1 = np.asarray(h1, dtype=float)
    n = h1.shape[0]
    energies = np.asarray(mo_energies, dtype=float).ravel()
    if energies.shape[0] != n:
        raise ValueError(
            f"mo_energies has {energies.shape[0]} entries, expected {n}"
        )
    core, occ, virt = _occupied_lists(n, n_occupied, n_frozen)
    if h2 is None:
        raise ValueError("MP2 needs two-electron integrals (h2=None given)")
    _, g = _spin_orbital_chemist(h1, h2, n)

    # chemist spin-orbital block (a i | b j), i.e. the physicist-ordered
    # <ab|ij>; the antisymmetrised element is (a i|b j) - (a j|b i).
    #
    # It is tempting to write this as (ab|ij) - (ab|ji) and call the difference
    # the same object.  It is not: in the chemist convention the first two
    # slots sit on electron 1, so antisymmetrising the *virtual* pair is a
    # different contraction from antisymmetrising the occupied one.  Measured on
    # the opposite-spin double of H2/STO-3G, (ab|ij) - (ab|ji) returns exactly
    # zero while the true element is the exchange integral -- and on water the
    # energy came out 33 times too large before this was fixed.
    o_so = [2 * p + s for p in occ for s in (0, 1)]
    v_so = [2 * p + s for p in virt for s in (0, 1)]
    block = g[np.ix_(v_so, o_so, v_so, o_so)]  # [a, i, b, j] = (a i | b j)
    direct = block.transpose(1, 3, 0, 2)       # [i, j, a, b] = (a i | b j)
    exch = block.transpose(3, 1, 0, 2)         # [i, j, a, b] = (a j | b i)

    e_o = np.array([energies[p] for p in occ for _ in (0, 1)])
    e_v = np.array([energies[p] for p in virt for _ in (0, 1)])
    denominator = (
        e_o[:, None, None, None]
        + e_o[None, :, None, None]
        - e_v[None, None, :, None]
        - e_v[None, None, None, :]
    )
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError(
            "a zero MP2 denominator: the occupied and virtual spaces are "
            "degenerate. Use semicanonical orbitals for an open-shell or "
            "symmetry-broken reference."
        )
    amplitude = (direct - exch) / denominator
    e2 = 0.25 * float(np.sum(amplitude * amplitude * denominator))
    return {
        "amplitude": amplitude,
        "denominator": denominator,
        "occ": occ,
        "virt": virt,
        "core": core,
        "e2": e2,
        "n_occ_spin": len(o_so),
        "n_virt_spin": len(v_so),
    }


def mp2_correlation_energy(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    mo_energies: np.ndarray,
    n_occupied: int,
    *,
    n_frozen: int = 0,
    spin: int = 0,
) -> float:
    """MP2 correlation energy of a closed-shell reference.

    Raises:
        ValueError: For ``spin != 0``. ``spin`` is accepted so that a caller
            can pass ``mol.spin`` and get a refusal that names the reason
            instead of a wrong number for a doublet.
    """
    if int(spin) != 0:
        raise ValueError(
            f"MP2 here is closed-shell only, but spin={spin} (2S). "
            "ROHF/UHF-MP2 needs semicanonical spin orbitals and a spin-resolved "
            "denominator; running the closed-shell formula would return a "
            "number for the wrong quantity."
        )
    return float(
        mp2_amplitudes(h1, h2, mo_energies, n_occupied, n_frozen=n_frozen)["e2"]
    )


def mp2_density(
    h1: np.ndarray,
    h2: Optional[np.ndarray],
    mo_energies: np.ndarray,
    n_occupied: int,
    *,
    n_frozen: int = 0,
) -> Dict[str, Any]:
    """Unrelaxed MP2 1-RDM, spin-summed, in the MO basis.

    Returns:
        Dict with ``density`` ``(n, n)`` (trace = electron count),
        ``occupations`` its diagonal in the input basis, ``nat_occupations``
        (eigenvalues, descending), ``rotation`` (columns are the natural
        orbitals in the input MO basis), ``e2``, ``gap``
        (``max |density[occ, virt]|``, which must be zero: the unrelaxed MP2
        density is block diagonal, and a non-zero gap means a bug), and
        ``physical``.

    ``nat_occupations`` lands in ``[0, 2]`` when MP2 is perturbative here, but
    that is a property of the *reference*, not of this function.  The unrelaxed
    MP2 1-RDM is not guaranteed to be a physical density: a small denominator
    drives the amplitudes past the point where the second-order correction can
    be trusted, and the diagonal correction can then exceed what positivity
    allows (a random 3-orbital model produced occupations of +121 and -119).
    ``physical`` reports that check instead of hiding it, because an active
    space selected on unphysical occupations is a sorted list of noise.
    """
    data = mp2_amplitudes(h1, h2, mo_energies, n_occupied, n_frozen=n_frozen)
    amp = data["amplitude"]
    occ, virt, core = data["occ"], data["virt"], data["core"]
    n = np.asarray(h1, dtype=float).shape[0]

    # dD_ij = -1/2 sum_kab t_ik^ab t_jk^ab ; dD_ab = +1/2 sum_ijc t_ij^ac t_ij^bc
    d_oo = -0.5 * np.einsum("ikab,jkab->ij", amp, amp, optimize=True)
    d_vv = +0.5 * np.einsum("ijac,ijbc->ab", amp, amp, optimize=True)

    so = np.zeros((2 * n, 2 * n), dtype=float)
    o_so = [2 * p + s for p in occ for s in (0, 1)]
    v_so = [2 * p + s for p in virt for s in (0, 1)]
    so[np.ix_(o_so, o_so)] = d_oo
    so[np.ix_(v_so, v_so)] = d_vv
    for p in core:
        so[2 * p, 2 * p] = 1.0
        so[2 * p + 1, 2 * p + 1] = 1.0
    for p in occ:
        so[2 * p, 2 * p] += 1.0
        so[2 * p + 1, 2 * p + 1] += 1.0

    # spin-summed spatial density: gamma_pq = sum_s D[2p+s, 2q+s]
    density = so[0::2, 0::2] + so[1::2, 1::2]

    # Natural orbitals diagonalise the two correlated blocks; the core block is
    # the identity in its own basis and the occ/virt blocks do not mix.
    rotation = np.eye(n)
    occupations = np.zeros(n)
    for indices in (core, occ, virt):
        if not indices:
            continue
        sub = density[np.ix_(indices, indices)]
        sub = 0.5 * (sub + sub.T)
        values, vectors = np.linalg.eigh(sub)
        order = np.argsort(-values)
        values, vectors = values[order], vectors[:, order]
        occupations[indices] = values
        for col, orbital in enumerate(indices):
            rotation[indices, orbital] = vectors[:, col]

    gap = 0.0
    if occ and virt:
        gap = float(np.max(np.abs(density[np.ix_(occ, virt)])))
    physical = bool(
        np.all(occupations >= -1e-8) and np.all(occupations <= 2.0 + 1e-8)
    )
    order = np.argsort(-occupations, kind="stable")
    return {
        "density": density,
        "spin_density": so,
        "occupations": occupations,
        "nat_occupations": occupations[order],
        "nat_order": order,
        "rotation": rotation[:, order],
        "e2": data["e2"],
        "gap": gap,
        "physical": physical,
        "tr_density": float(np.trace(density)),
    }


def mp2_no_activity(occupations: np.ndarray) -> np.ndarray:
    """Two-level correlation entropy of each MP2 natural occupation.

    ``s_p = -x ln x - (1 - x) ln(1 - x)`` with ``x = n_p / 2``: exactly zero for
    a perfectly full (``n = 2``) or perfectly empty (``n = 0``) orbital and
    maximal (``ln 2``) at ``n = 1``.  It answers "how far is this orbital from
    being decided", which is what an MP2-NO active space is selecting on, and
    it is the same pair entropy :mod:`superfermion.chemistry.apc` sums when it
    ranks orbitals.
    """
    x = np.clip(np.asarray(occupations, dtype=float).ravel() / 2.0, 0.0, 1.0)
    out = np.zeros_like(x)
    inner = (x > 0.0) & (x < 1.0)
    xi = x[inner]
    out[inner] = -(xi * np.log(xi) + (1.0 - xi) * np.log(1.0 - xi))
    return out


def mp2_no_rank(occupations: np.ndarray) -> np.ndarray:
    """Orbitals ordered most-to-least active by MP2 natural occupation.

    Descending :func:`mp2_no_activity`, so the frontier orbitals come first.

    Ranking on ``|n - 1|`` -- what this used to do -- is degenerate: a core
    orbital at ``n = 2.0000`` and an empty one at ``n = 0.0000`` score 1.0
    identically, and as soon as the empty occupations fall below ``1e-4`` they
    are *closer* to 1 than a core orbital is, so they come out on top.  Asking
    for the two most active orbitals then returned two empty ones.
    """
    return np.argsort(-mp2_no_activity(occupations), kind="stable")


def mp2_natural_orbitals(mol: Any, *, n_frozen: int = 0) -> Dict[str, Any]:
    """MP2 natural orbitals of a closed-shell :class:`MolecularData`.

    Args:
        mol: Must carry ``h1``, ``h2``, ``mo_energies`` and ``mo_occupations``
            (everything :func:`molecule_from_geometry` sets).
        n_frozen: Lowest orbitals excluded from correlation.

    Returns:
        Dict with ``occupations`` (descending), ``rotation`` (columns are the
        natural orbitals in the original MO basis), ``h1``/``h2`` in that
        basis, ``mo_energies``/``mo_coeffs`` if the object has them, ``e2``,
        ``e_hf``, ``e_total``, ``gap``, ``physical`` and ``rank``.
    """
    if mol.h1 is None or mol.h2 is None:
        raise ValueError(
            "MP2 natural orbitals need the MO integrals: build the molecule "
            "with PySCF available (h1/h2 are None for a library record)."
        )
    if getattr(mol, "mo_energies", None) is None:
        raise ValueError(
            "MP2 needs orbital energies; set them (set_mean_field(mo_energies="
            "...)) or build the molecule with PySCF."
        )
    if getattr(mol, "mo_occupations", None) is None:
        # Without this the sum below is ``float(None)``, i.e. a bare TypeError
        # from inside numpy rather than a statement of what is missing.
        raise ValueError(
            "MP2 needs the mean-field occupations; call "
            "set_mean_field(occupations=...) or build the molecule with PySCF."
        )
    if int(getattr(mol, "spin", 0)) != 0:
        raise ValueError(
            f"closed-shell MP2 only, but {mol.name} has spin={mol.spin} (2S)."
        )
    n_occupied = int(round(float(np.sum(mol.mo_occupations)) / 2.0))
    data = mp2_density(
        mol.h1, mol.h2, mol.mo_energies, n_occupied, n_frozen=n_frozen
    )
    rotation = data["rotation"]

    if not data["physical"]:
        worst = float(np.min(data["occupations"]))
        best = float(np.max(data["occupations"]))
        warnings.warn(
            f"unrelaxed MP2 natural occupations span [{worst:.4g}, {best:.4g}], "
            "outside the physical [0, 2]: MP2 is not perturbative on this "
            "reference (typically a near-degenerate denominator), so the "
            "natural-orbital ordering and any active space selected from it "
            "are not meaningful. Use a canonical or semicanonical mean field "
            "with a real gap.",
            RuntimeWarning,
            stacklevel=2,
        )

    from superfermion.chemistry.orbital_reduction import rotate_by_unitary

    h1_no, h2_no = rotate_by_unitary(mol.h1, mol.h2, rotation)
    result: Dict[str, Any] = {
        "occupations": data["nat_occupations"],
        "rotation": rotation,
        "h1": h1_no,
        "h2": h2_no,
        "e2": data["e2"],
        "e_hf": float(mol.hf_energy),
        "e_total": float(mol.hf_energy) + data["e2"],
        "gap": data["gap"],
        "n_occupied": n_occupied,
        "n_frozen": n_frozen,
        "physical": data["physical"],
        "rank": mp2_no_rank(data["nat_occupations"]),
    }
    if getattr(mol, "mo_energies", None) is not None:
        # Diagonal of the transformed Fock matrix, i.e. sum_p R_pq^2 eps_p.
        # ``eps @ R`` would be wrong twice over: the NO basis is not canonical,
        # so the transformed matrix is not diagonal and only its diagonal is an
        # orbital energy, and a rotation mixes eigenvalues rather than carrying
        # each one to a new label.
        energies = np.asarray(mol.mo_energies, dtype=float)
        result["mo_energies"] = (rotation ** 2).T @ energies
    if getattr(mol, "_mo_coeffs", None) is not None:
        result["mo_coeffs"] = np.asarray(mol._mo_coeffs, dtype=float) @ rotation
    return result


def mp2_no_molecule(mol: Any, *, n_frozen: int = 0) -> tuple:
    """A copy of ``mol`` re-expressed in its MP2 natural-orbital basis.

    The natural orbitals are a rotation of the stored MOs, so every other
    attribute (geometry, electron count, nuclear repulsion) is unchanged -- but
    ``h1``, ``h2``, ``mo_energies``, ``mo_occupations`` and ``_mo_coeffs`` must
    all be transformed together, and splitting that across a caller is how the
    integrals and the occupations end up in different bases.

    In the returned object the orbitals are ordered by descending MP2 natural
    occupation, so the core is the first ``n_frozen`` orbitals and
    ``active_space_from_mp2_no``'s indices index into it directly.

    Returns:
        ``(molecule, natural_orbitals_dict)``.
    """
    import dataclasses

    no = mp2_natural_orbitals(mol, n_frozen=n_frozen)
    n = int(mol.n_orbitals)
    # The occupations are sorted descending, so the density's diagonal is the
    # occupation vector itself.
    occupations = np.zeros(n, dtype=float)
    occupations[: len(no["occupations"])] = no["occupations"]
    changes: Dict[str, Any] = {
        "h1": no["h1"],
        "h2": no["h2"],
        "mo_occupations": occupations,
        "active_indices": None,
        "active_orbitals": None,
        "removed_orbitals": None,
    }
    if no.get("mo_energies") is not None:
        # Diagonal of the transformed Fock matrix: sum_p R_pq^2 eps_p, not
        # eps @ R -- the NO basis is not canonical, so the transformed matrix
        # has off-diagonal elements and only its diagonal is an "orbital energy".
        energies = np.asarray(mol.mo_energies, dtype=float)
        changes["mo_energies"] = (no["rotation"] ** 2).T @ energies
    if no.get("mo_coeffs") is not None:
        changes["_mo_coeffs"] = no["mo_coeffs"]
    if getattr(mol, "mo_irreps", None) is not None:
        # A rotated basis does not carry pure irreps any more; keeping the old
        # labels would attach symmetry labels to orbitals that do not have them.
        changes["mo_irreps"] = None
        changes["irrep_ids"] = None
        changes["irrep_labels"] = None
        changes["point_group"] = None
    return dataclasses.replace(mol, **changes), no


def active_space_from_mp2_no(
    mol: Any,
    *,
    n_cas: Optional[int] = None,
    threshold: float = 0.02,
    n_frozen: int = 0,
) -> Dict[str, Any]:
    """Choose a CAS from MP2 natural occupations.

    Two selection modes:

    * ``n_cas`` given -- a contiguous window of ``n_cas`` orbitals centred on
      the HOMO.  MP2 natural orbitals come out ordered by *descending*
      occupation, so the window
      ``n_occ - n_cas//2 .. n_occ - n_cas//2 + n_cas`` is the frontier block:
      the partially occupied valence orbitals plus the near-degenerate
      virtuals they correlate with.  The window slides down when it would run
      past the last orbital, so ``n_cas = n_orbitals`` gives the full space and
      ``n_cas = 2`` gives HOMO/LUMO.
    * ``threshold`` (default) -- every orbital with
      ``threshold <= n_p <= 2 - threshold``, the standard "fractionally
      occupied" criterion.  This mode is self-consistent by construction:
      anything it excludes is either above ``2 - threshold`` or below
      ``threshold``.

    Either way the frozen core is the *prefix* below the lowest active index.
    That is forced by the consumer: ``MolecularData._apply_active_space`` takes
    the core as ``range(ncore)`` and derives the active electron count as
    ``n_electrons - 2 * ncore``.  The occupation ordering is what makes that
    prefix the genuinely doubly occupied orbitals.  The reported
    ``n_active_electrons`` is measured from the occupations, so when it
    disagrees with ``n_electrons - 2 * n_core`` the space is not neutral and
    ``warning`` says so -- that disagreement is the signal that freezing the
    prefix throws away real charge.

    Returns:
        Dict with ``active_indices``, ``n_active_electrons``,
        ``n_frozen_core``, ``n_cas``, ``occupations`` (all), ``selected``
        (occupations of the active set), ``activity`` (per-orbital pair
        entropy), ``dropped_occupied``/``dropped_empty`` (counts), ``warning``
        and ``no`` (the full :func:`mp2_natural_orbitals` result).
    """
    no = mp2_natural_orbitals(mol, n_frozen=n_frozen)
    occupations = np.asarray(no["occupations"], dtype=float)
    n_orb = int(occupations.size)
    n_occ = int(no["n_occupied"])

    if n_cas is not None:
        n_cas = int(n_cas)
        if not 1 <= n_cas <= n_orb:
            raise ValueError(
                f"n_cas={n_cas} must be in 1..{n_orb}"
            )
        low = n_occ - n_cas // 2
        low = max(0, min(low, n_orb - n_cas))
        active = list(range(low, low + n_cas))
    else:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold={threshold} must lie in (0, 1]")
        active = [
            int(i)
            for i in range(n_orb)
            if threshold <= occupations[i] <= 2.0 - threshold
        ]
        if not active:
            raise ValueError(
                f"no orbital has an MP2 occupation in "
                f"[{threshold}, {2 - threshold}]; lower threshold= or pass "
                f"n_cas="
            )

    n_core = int(min(active))
    active_set = set(active)
    rest = [i for i in range(n_core, n_orb) if i not in active_set]
    n_active_electrons = int(round(float(np.sum(occupations[active]))))
    from_core = int(mol.n_electrons) - 2 * n_core

    notes: List[str] = []
    core_not_full = [
        i for i in range(n_core) if occupations[i] < 2.0 - threshold
    ]
    dropped_not_empty = [i for i in rest if occupations[i] > threshold]
    if core_not_full:
        notes.append(
            f"frozen-core orbitals {core_not_full} are not full "
            f"(occupations "
            f"{[round(float(occupations[i]), 4) for i in core_not_full]})"
        )
    if dropped_not_empty:
        notes.append(
            f"dropped orbitals {dropped_not_empty} are not empty "
            f"(occupations "
            f"{[round(float(occupations[i]), 4) for i in dropped_not_empty]})"
        )
    if n_active_electrons != from_core:
        notes.append(
            f"the selected occupations sum to {n_active_electrons} electrons "
            f"but freezing {n_core} doubly occupied orbitals leaves "
            f"{from_core}; the space is not charge-neutral"
        )
    warning = None
    if notes:
        warning = (
            "; ".join(notes)
            + f" (threshold={threshold}); the CAS truncates a partially "
            f"occupied orbital"
        )
    return {
        "active_indices": tuple(active),
        "n_active_electrons": n_active_electrons,
        "n_frozen_core": n_core,
        "n_cas": len(active),
        "occupations": occupations,
        "selected": occupations[active],
        "activity": mp2_no_activity(occupations),
        "dropped_occupied": n_core,
        "dropped_empty": len(rest),
        "warning": warning,
        "no": no,
    }


def casci_energy(
    mol: Any,
    active_indices: Sequence[int],
    *,
    n_frozen_core: int = 0,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    max_determinants: int = 20000,
) -> float:
    """CASCI energy in an explicit active set, with the core frozen.

    The active-space contraction is the same one ``to_hamiltonian`` uses, so a
    CASCI number reported here and a Hamiltonian built from the same selection
    cannot disagree.  The matrix is built sparse and diagonalised by ARPACK, but
    the determinant list itself is enumerated explicitly, so a space larger than
    ``max_determinants`` is refused rather than started: a 12-orbital CAS with 8
    electrons is 245 000 determinants, which does not fit in this process and
    used to hang instead of failing.

    Raises:
        ValueError: if the space has more than ``max_determinants``
            determinants. Use :func:`superfermion.chemistry.sqd.qsci` or a
            seniority-truncated pool for those.
    """
    from math import comb

    from superfermion.chemistry.pyscf_bridge import MolecularData
    from superfermion.chemistry.subspace import fermionic_terms, ground_state
    from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals

    if not isinstance(mol, MolecularData):
        raise TypeError("casci_energy expects a MolecularData")
    active = [int(i) for i in active_indices]
    h1_as, h2_as, n_elec, n_orbs, e_fc = mol._apply_active_space(
        None, active_indices=active, frozen_core=n_frozen_core
    )
    if n_alpha is None and n_beta is None:
        n_alpha = (n_elec + mol.spin) // 2
        n_beta = n_elec - n_alpha
    size = comb(n_orbs, n_alpha) * comb(n_orbs, n_beta)
    if size > max_determinants:
        raise ValueError(
            f"a CAS({n_elec},{n_orbs}) in this spin sector has {size} "
            f"determinants, above max_determinants={max_determinants}; "
            f"use qsci()/sqd_energy() with a truncated pool"
        )
    h1_so, h2_so = _spin_orbital_integrals(h1_as, h2_as)
    terms = fermionic_terms(h1_so, h2_so)
    from superfermion.chemistry.subspace import all_determinants

    dets = all_determinants(
        n_orbs, n_elec, n_alpha=n_alpha, n_beta=n_beta
    )
    energies, _ = ground_state(terms, dets, sparse=True)
    return float(energies[0] + mol.nuclear_repulsion + e_fc)
