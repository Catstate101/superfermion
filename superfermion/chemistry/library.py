"""Molecules Library — a small set of *real* pre-computed molecular integrals.

Provenance
----------
Every record here was computed with PySCF 2.14.0 (RHF, STO-3G) and stored with
the geometry it was computed for, so a lookup can compare the request against
the record instead of guessing by name.  The H2 record was produced by::

    mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
    mf  = scf.RHF(mol).run()
    h1  = C.T @ mf.get_hcore() @ C                      # MO basis
    h2  = ao2mo.full(mol, C, compact=False)             # chemist (pq|rs)

and its FCI energy, also from PySCF, is ``-1.137306035753`` Ha.

Why this file is this small
---------------------------
It used to carry "mock coefficients for demonstration" for H2, a hand-written
six-term LiH sum, and an empty term list for water, all selected by substring
against the molecule name.  ``_compute_fallback`` then reported 2 electrons and
2 orbitals for any name containing ``"h2"`` -- including ``"h2o"`` -- and
``hf_energy`` was ``0.0`` for every record, because no record had the key.  A
fallback that returns fabricated integrals is worse than no fallback: the
numbers look like physics and are compared against physics.  Only records with
real integrals and a stored geometry survive, and a request for anything else
raises instead of inventing an answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["MoleculeLibrary", "normalise_geometry"]


def normalise_geometry(geometry: str) -> Tuple[Tuple[str, float, float, float], ...]:
    """Canonical form of an XYZ string, for comparing a request to a record.

    Atom order is discarded (a set of nuclei has no order) and coordinates are
    rounded to 1e-4 Å so that ``"0.735"`` and ``"0.7350001"`` compare equal.
    The rounding is a comparison tolerance, not a change to the integrals: the
    record was computed at the exact coordinates in its ``geometry`` field.
    """
    atoms: List[Tuple[str, float, float, float]] = []
    for part in geometry.replace("\n", ";").split(";"):
        fields = part.split()
        if not fields:
            continue
        if len(fields) < 4:
            raise ValueError(
                f"cannot parse {part!r} as 'symbol x y z'; got {len(fields)} "
                f"field(s)"
            )
        symbol = fields[0].capitalize()
        x, y, z = (round(float(v), 4) for v in fields[1:4])
        atoms.append((symbol, x + 0.0, y + 0.0, z + 0.0))  # +0.0 folds -0.0
    return tuple(sorted(atoms))


#: Real STO-3G H2, d = 0.735 Å, RHF.  See the module docstring for provenance.
_H2_STO3G_0735 = {
    "name": "H2",
    "geometry": "H 0 0 0; H 0 0 0.735",
    "basis": "sto-3g",
    "charge": 0,
    "spin": 0,
    "n_electrons": 2,
    "n_orbitals": 2,
    "nuclear_repulsion": 0.719968994449,
    "hf_energy": -1.116998996754,
    "fci_energy": -1.137306035753,
    "h1": np.array(
        [
            [-1.256339073003, -0.000000000000],
            [-0.000000000000, -0.471896007281],
        ]
    ),
    # Chemist notation (pq|rs), the layout MolecularData.h2 documents.
    "h2": np.array(
        [
            [
                [[0.675710154804, 0.000000000000],
                 [0.000000000000, 0.664581730255]],
                [[0.000000000000, 0.180931199784],
                 [0.180931199784, 0.000000000000]],
            ],
            [
                [[0.000000000000, 0.180931199784],
                 [0.180931199784, -0.000000000000]],
                [[0.664581730255, 0.000000000000],
                 [0.000000000000, 0.698573722732]],
            ],
        ]
    ),
}


class MoleculeLibrary:
    """Real pre-computed integrals, keyed by (geometry, basis).

    ``get`` is the only lookup path: it returns a record or ``None``.  There is
    deliberately no name-based matching, because name matching is what made
    ``"h2o"`` resolve to hydrogen.
    """

    _RECORDS: Tuple[Dict[str, Any], ...] = (_H2_STO3G_0735,)

    @classmethod
    def _key(cls, geometry: str, basis: str) -> Tuple[Any, str]:
        return normalise_geometry(geometry), str(basis).strip().lower()

    @classmethod
    def available(cls) -> List[str]:
        """One line per record: name, basis and the geometry it is valid for."""
        return [
            f"{rec['name']}/{rec['basis']}: {rec['geometry']}"
            for rec in cls._RECORDS
        ]

    @classmethod
    def get(
        cls,
        geometry: str,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """The record for this molecule, or ``None``.

        Charge and spin are part of the key: the stored RHF integrals describe
        a neutral singlet and are not a starting point for an ion.
        """
        try:
            wanted = cls._key(geometry, basis)
        except (ValueError, TypeError):
            return None
        for rec in cls._RECORDS:
            if cls._key(rec["geometry"], rec["basis"]) != wanted:
                continue
            if int(rec["charge"]) != int(charge) or int(rec["spin"]) != int(spin):
                return None
            return rec
        return None

    @classmethod
    def get_by_name(cls, name: str, basis: str = "sto-3g") -> Optional[Dict[str, Any]]:
        """Exact, case-insensitive name lookup (``"H2"``), never a substring."""
        target = str(name).strip().lower()
        for rec in cls._RECORDS:
            if rec["name"].strip().lower() == target and (
                str(rec["basis"]).lower() == str(basis).strip().lower()
            ):
                return rec
        return None

    # ---- convenience accessors kept for the public API -------------------
    @classmethod
    def hydrogen(cls, distance: float = 0.735) -> Dict[str, Any]:
        """The STO-3G H2 record.  Only ``distance == 0.735`` is tabulated."""
        rec = cls.get(f"H 0 0 0; H 0 0 {float(distance):.4f}", "sto-3g")
        if rec is None:
            raise KeyError(
                f"no tabulated H2 record at d = {distance} A; available: "
                + "; ".join(cls.available())
            )
        return rec
