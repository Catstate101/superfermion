"""Point-group sectors in the determinant basis: sorting, not guessing.

``MolecularData.mo_irreps`` labels each MO with its point-group irrep when the
molecule was built with ``symmetry=True``.  A labelling is only worth having if
it buys a *reduction*, so this module turns the labels into the one thing a
determinant-basis solver can use directly: the irrep of a determinant.  Sort
the determinants by it and the Hamiltonian is block diagonal, because the
Hamiltonian commutes with every point-group operation.

The rule
--------
A determinant's irrep is the direct product of the irreps of the spatial
orbitals it occupies, each taken with the parity of its occupation: doubly
occupied contributes ``irrep * irrep = 1``, singly occupied contributes
``irrep``, empty contributes 1.  Every irrep of the point groups below is
one-dimensional, so the product of labels is an abelian group operation and
``pyscf``'s ``irrep_id`` is an integer bit pattern for that operation -- the
product is the XOR of the ids.  That is the whole calculation:

    irrep(det) = XOR over p with odd occupation of irrep_id(p)

Why not simply trust it
-----------------------
The XOR shortcut is a property of how ``pyscf`` orders irreps for the eight
abelian groups in :data:`ABELIAN_POINT_GROUPS`, not a theorem about point
groups.  For a group with degenerate irreps (``C3v``, ``D3h``, ``Td``, ``Oh``)
``irrep_id`` is just an index into a list of representations and XOR means
nothing, so :func:`require_abelian` refuses those outright rather than return a
plausible label.

For the abelian groups the rule is checked, not assumed: the measurement in
``_verify_qn_parity.py`` builds the full Hamiltonian matrix and reports the
largest coupling between two different sectors, which must be zero up to
round-off.  If that number were not zero this module would be wrong, and the
number is what a reader should look at.

What this is not
----------------
No group table, no character vectors, no symmetry-adapted basis: the sector
label only ever decides *which determinants to keep*.  Anything that needs the
symmetry of the many-body state itself (spatial degeneracy, non-abelian
selection rules) is out of scope and says so instead of approximating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ABELIAN_POINT_GROUPS",
    "require_abelian",
    "orbit_irreps",
    "determinant_irrep",
    "sector_from_occupation",
    "sector_determinants",
]

#: Point groups whose irreps are all one-dimensional, so that the group
#: operation on ``pyscf``'s ``irrep_id`` integers is bitwise XOR.
ABELIAN_POINT_GROUPS = frozenset(
    {"C1", "Ci", "Cs", "C2", "C2v", "C2h", "D2", "D2h"}
)


def require_abelian(point_group: Optional[str]) -> str:
    """Return ``point_group`` if the XOR rule applies to it, else raise.

    Args:
        point_group: A ``pyscf`` ``groupname`` such as ``"C2v"``.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: if the group is missing or has a degenerate irrep.  A
            degenerate irrep makes ``irrep_id`` an index into a representation
            list rather than a bit pattern, so the direct-product rule below
            would silently return a wrong label.
    """
    if not point_group:
        raise ValueError(
            "no point group recorded: build the molecule with symmetry=True "
            "before asking for a symmetry sector"
        )
    name = str(point_group)
    if name not in ABELIAN_POINT_GROUPS:
        raise ValueError(
            f"point group {name!r} is not one of the supported abelian groups "
            f"{sorted(ABELIAN_POINT_GROUPS)}; its irreps are degenerate, so "
            "an irrep id is not a bit pattern and the direct-product rule "
            "would return a wrong sector instead of failing"
        )
    return name


def orbit_irreps(mol: Any, orbitals: Optional[Sequence[int]] = None) -> List[int]:
    """``pyscf`` irrep id of each spatial orbital, in the given order.

    Args:
        mol: A :class:`~superfermion.chemistry.pyscf_bridge.MolecularData`
            built with ``symmetry=True``.
        orbitals: Spatial orbital indices to report.  Default: all of them, in
            MO order.  Pass the active window when the determinant basis is an
            active space, because the returned list must line up
            position-for-position with the bits of a determinant.

    Returns:
        One ``irrep_id`` per requested orbital.

    Raises:
        ValueError: if the molecule carries no irrep data, or an MO label is
            absent from the recorded irreps.
    """
    labels = getattr(mol, "mo_irreps", None)
    irrep_labels = getattr(mol, "irrep_labels", None)
    irrep_ids = getattr(mol, "irrep_ids", None)
    if labels is None or irrep_labels is None or irrep_ids is None:
        raise ValueError(
            "this molecule has no irrep data: build it with symmetry=True"
        )
    require_abelian(getattr(mol, "point_group", None))

    label_to_id: Dict[str, int] = {
        str(label): int(index)
        for label, index in zip(irrep_labels, irrep_ids)
    }
    if orbitals is None:
        indices = range(len(labels))
    else:
        indices = [int(p) for p in orbitals]
    out: List[int] = []
    for p in indices:
        if not 0 <= p < len(labels):
            raise ValueError(
                f"orbital {p} is outside the {len(labels)} MOs of this molecule"
            )
        label = str(labels[p])
        try:
            out.append(label_to_id[label])
        except KeyError:
            raise ValueError(
                f"MO {p} has irrep {label!r}, which is not among the recorded "
                f"irreps {list(label_to_id)}"
            ) from None
    return out


def determinant_irrep(irreps: Sequence[int], det: int) -> int:
    """Point-group irrep of determinant ``det``, as an XOR of ``irrep_id``.

    Args:
        irreps: ``irrep_id`` of each spatial orbital, position-for-position
            with the determinant's bit layout ``P = 2p + s``.
        det: Determinant bitmask; bit ``P`` set means spin orbital ``P`` is
            occupied.

    Returns:
        The irrep id of the direct product of the singly occupied orbitals.

    Raises:
        ValueError: if ``det`` is negative or sets a bit beyond `2 * len(irreps)`.
    """
    orbitals = [int(x) for x in irreps]
    n_so = 2 * len(orbitals)
    value = int(det)
    if value < 0:
        raise ValueError(f"a determinant is a non-negative bitmask, got {det}")
    if value >> n_so:
        raise ValueError(
            f"determinant {value} sets a bit at or above {n_so}, but only "
            f"{len(orbitals)} spatial orbitals were given"
        )
    sector = 0
    for p, irrep in enumerate(orbitals):
        # An odd occupation contributes the irrep once; a doubly occupied
        # orbital contributes it twice, which is the identity in a 1-D irrep.
        occupied = ((value >> (2 * p)) & 1) + ((value >> (2 * p + 1)) & 1)
        if occupied & 1:
            sector ^= irrep
    return sector


def sector_from_occupation(irreps: Sequence[int], occupation: Sequence[int]) -> int:
    """Irrep of the determinant described by a per-spin-orbital occupation list.

    Exists because :meth:`MolecularData.reference_occupation` returns a list of
    0/1 entries in Pauli-string order, while :func:`determinant_irrep` wants a
    bitmask; this converts one to the other so no caller hand-rolls the shift.
    """
    entries = [int(x) for x in occupation]
    if len(entries) != 2 * len([int(x) for x in irreps]):
        raise ValueError(
            f"occupation has {len(entries)} entries but {len(irreps)} spatial "
            "orbitals imply 2x that many spin orbitals"
        )
    det = 0
    for bit, occupied in enumerate(entries):
        if occupied:
            det |= 1 << bit
    return determinant_irrep(irreps, det)


def sector_determinants(
    irreps: Sequence[int],
    determinants: Sequence[int],
    *,
    sector: int,
) -> List[int]:
    """The determinants whose irrep equals ``sector``, order preserved.

    ``sector`` is required rather than defaulted to ``0``: the totally
    symmetric sector happens to be the right answer for a closed-shell ground
    state, but assuming it would silently return an empty list for a molecule
    whose reference is not totally symmetric, and an empty basis is a much
    worse failure than a missing keyword.

    Args:
        irreps: ``irrep_id`` per spatial orbital, as for
            :func:`determinant_irrep`.
        determinants: Determinant bitmasks to filter.
        sector: The irrep id to keep; get it from
            :func:`sector_from_occupation` for the reference determinant.

    Returns:
        The kept determinants.  Empty if no determinant has that irrep, which
        means the basis and the orbital irreps disagree -- the caller should
        check rather than diagonalise an empty matrix.
    """
    wanted = int(sector)
    return [
        int(det)
        for det in determinants
        if determinant_irrep(irreps, det) == wanted
    ]
