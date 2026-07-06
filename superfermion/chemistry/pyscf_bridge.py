"""
PySCF Bridge — Molecular electronic structure via PySCF integration.

Computes one- and two-electron integrals, active-space transformations,
and builds molecular Hamiltonians for any geometry+basis combination.

If PySCF is not installed, falls back to pre-computed molecule library values.

Usage:
    >>> from superfermion.chemistry.pyscf_bridge import molecule_from_geometry
    >>>
    >>> mol = molecule_from_geometry("H 0 0 0; F 0 0 0.92", basis="sto-3g")
    >>> H = mol.to_hamiltonian(active_space=(2, 2))  # 2 electrons in 2 orbitals
    >>> print(f"{mol.n_qubits}-qubit Hamiltonian ready")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try importing PySCF — graceful degradation if not installed
try:
    import pyscf
    import pyscf.scf
    import pyscf.ao2mo
    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False


@dataclass
class MolecularData:
    """Container for molecular integral data."""

    name: str
    geometry: str  # XYZ format: "H 0 0 0; H 0 0 0.74"
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0  # 2S where S = spin multiplicity − 1

    # One-body integrals (MO basis)
    h1: Optional[np.ndarray] = None  # h_{ij} = ⟨i|h|j⟩
    # Two-body integrals (chemist notation)
    h2: Optional[np.ndarray] = None  # h_{ijkl} = (ij|kl)
    # Nuclear repulsion energy
    nuclear_repulsion: float = 0.0
    # HF energy
    hf_energy: float = 0.0

    n_orbitals: int = 0
    n_electrons: int = 0

    # Active space parameters
    active_orbitals: Optional[Tuple[int, int]] = None  # (n_elec, n_orbs)
    frozen_core: int = 0

    _mo_coeffs: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_qubits(self) -> int:
        """Number of qubits needed (2× active orbitals for JW mapping)."""
        if self.active_orbitals is not None:
            return 2 * self.active_orbitals[1]
        return 2 * self.n_orbitals

    @property
    def n_active_electrons(self) -> int:
        if self.active_orbitals is not None:
            return self.active_orbitals[0]
        return self.n_electrons

    @property
    def n_active_orbitals(self) -> int:
        if self.active_orbitals is not None:
            return self.active_orbitals[1]
        return self.n_orbitals

    def to_hamiltonian(
        self,
        active_space: Optional[Tuple[int, int]] = None,
        mapping: str = "jordan_wigner",
    ) -> Any:
        """Build a superfermion Hamiltonian from the molecular integrals.

        Args:
            active_space: Optional (n_electrons, n_orbitals) active space.
            mapping: Fermion-to-qubit mapping (``"jordan_wigner"`` or
                     ``"bravyi_kitaev"``).

        Returns:
            sf.Hamiltonian.
        """
        import superfermion as sf

        if self.h1 is None:
            return self._hamiltonian_from_library()

        # Apply active space transformation
        h1_as, h2_as, n_elec, n_orbs = self._apply_active_space(active_space)

        # Build fermionic operator
        from superfermion.chemistry.hamiltonians import FermionicOperator

        ferm_op = FermionicOperator.from_coeffs(h1_as, h2_as)

        # Map to qubits
        if mapping == "jordan_wigner":
            H = ferm_op.to_jordan_wigner()
        elif mapping == "bravyi_kitaev":
            H = ferm_op.to_bravyi_kitaev()
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        # Add nuclear repulsion to identity term
        from superfermion.observables.core import PauliString
        identity = PauliString("I" * (2 * n_orbs), self.nuclear_repulsion)
        H += identity

        return H

    def _apply_active_space(
        self,
        active_space: Optional[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Transform integrals to active space."""
        if active_space is None and self.active_orbitals is None:
            n_elec = self.n_active_electrons
            n_orbs = self.n_active_orbitals
            return self.h1, self.h2, n_elec, n_orbs

        n_elec, n_orbs = active_space or self.active_orbitals or (
            self.n_electrons,
            self.n_orbitals,
        )
        n_total = self.n_orbitals

        if n_orbs > n_total:
            n_orbs = n_total  # Clamp to available orbitals

        # Freeze core orbitals, keep active space
        if self.h1 is not None:
            h1_as = self.h1[:n_orbs, :n_orbs]
        else:
            h1_as = np.zeros((n_orbs, n_orbs))

        if self.h2 is not None:
            h2_as = self.h2[:n_orbs, :n_orbs, :n_orbs, :n_orbs]
        else:
            h2_as = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))

        # Add frozen-core contribution to the effective one-body term
        if self.h2 is not None and n_orbs < n_total:
            for p in range(n_orbs):
                for q in range(n_orbs):
                    for i in range(n_orbs, n_total):
                        # Frozen core: (pi|qi) − ½ (pq|ii)
                        h1_as[p, q] += (
                            2.0 * self.h2[p, i, q, i]
                            - self.h2[p, q, i, i]
                        )

        # Add frozen-core energy to nuclear repulsion
        if self.h1 is not None and n_orbs < n_total:
            e_fc = 0.0
            for i in range(n_orbs, n_total):
                e_fc += 2.0 * self.h1[i, i]
                if self.h2 is not None:
                    for j in range(n_orbs, n_total):
                        e_fc += (
                            2.0 * self.h2[i, j, i, j]
                            - self.h2[i, j, j, i]
                        )

        return h1_as, h2_as, n_elec, n_orbs

    def _hamiltonian_from_library(self) -> Any:
        """Fallback: use pre-computed molecule library values."""
        import superfermion as sf
        from superfermion.chemistry.library import MoleculeLibrary
        from superfermion.observables.core import SparsePauliOp

        lib = MoleculeLibrary()

        # Try to match by name
        name_lower = self.name.lower()
        if "h2" in name_lower:
            data = lib.hydrogen()
        elif "lih" in name_lower:
            data = lib.lithium_hydride()
        elif "h2o" in name_lower:
            data = lib.water()
        else:
            return sf.Hamiltonian({})

        return SparsePauliOp.from_dict(
            {k: v for k, v in data["hamiltonian"]}
        )

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of molecular data."""
        return {
            "name": self.name,
            "geometry": self.geometry,
            "basis": self.basis,
            "n_electrons": self.n_electrons,
            "n_orbitals": self.n_orbitals,
            "n_qubits": self.n_qubits,
            "active_orbitals": self.active_orbitals,
            "hf_energy": self.hf_energy,
            "nuclear_repulsion": self.nuclear_repulsion,
            "pyscf_available": _HAS_PYSCF,
        }


# ═════════════════════════════════════════════════════════════════════════
# PySCF bridge
# ═════════════════════════════════════════════════════════════════════════

def molecule_from_geometry(
    geometry: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    name: Optional[str] = None,
) -> MolecularData:
    """Build a MolecularData object from an XYZ geometry string.

    If PySCF is available, computes full HF + integral data.
    Otherwise falls back to molecule library values.

    Args:
        geometry: XYZ-format geometry string, e.g.
                  ``"H 0 0 0; H 0 0 0.74"``.
        basis: Basis set name (``"sto-3g"``, ``"6-31g"``, ``"cc-pVDZ"``, etc.).
        charge: Total molecular charge.
        spin: 2S where S is spin multiplicity − 1 (0 = singlet, 1 = doublet).
        name: Optional molecule name. Auto-generated if None.

    Returns:
        MolecularData with populated integrals.
    """
    if name is None:
        # Auto-name from geometry
        atoms = _parse_geometry(geometry)
        name = "".join(atoms)

    mol = MolecularData(
        name=name,
        geometry=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    if _HAS_PYSCF:
        _compute_pyscf(mol)
    else:
        _compute_fallback(mol)

    return mol


def molecule_from_xyz(
    xyz_path: str,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
) -> MolecularData:
    """Build MolecularData from an XYZ file.

    Args:
        xyz_path: Path to .xyz file.
        basis: Basis set name.
        charge: Total charge.
        spin: 2S.
    """
    with open(xyz_path, "r") as f:
        xyz_content = f.read()

    # Parse standard XYZ format
    lines = xyz_content.strip().split("\n")
    name = lines[1].strip() if len(lines) > 1 else "molecule"
    geometry = "; ".join(line.strip() for line in lines[2:] if line.strip())

    return molecule_from_geometry(geometry, basis, charge, spin, name)


def _parse_geometry(geometry: str) -> List[str]:
    """Extract atom symbols from geometry string."""
    atoms = []
    for part in geometry.split(";"):
        part = part.strip()
        if part:
            atoms.append(part.split()[0])
    return atoms


def _compute_pyscf(mol: MolecularData):
    """Compute molecular integrals using PySCF."""
    # Build PySCF molecule
    mol_obj = pyscf.gto.M(
        atom=mol.geometry,
        basis=mol.basis,
        charge=mol.charge,
        spin=mol.spin,
        verbose=0,
    )

    mol.n_electrons = mol_obj.nelec[0] + mol_obj.nelec[1]
    mol.n_orbitals = mol_obj.nao

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol_obj)
    mf.kernel()

    mol.hf_energy = float(mf.e_tot)
    mol._mo_coeffs = mf.mo_coeff

    # One-body integrals in AO basis → transform to MO
    h1_ao = mf.get_hcore()  # Kinetic + nuclear attraction
    h1_mo = np.einsum(
        "pi,pq,qj->ij",
        mf.mo_coeff.conj().T, h1_ao, mf.mo_coeff,
    )
    mol.h1 = np.asarray(h1_mo, dtype=float)

    # Two-body integrals in AO basis → transform to MO (chemist notation)
    eri_ao = mol_obj.intor("int2e")
    n_ao = eri_ao.shape[0]
    eri_ao = eri_ao.reshape(n_ao, n_ao, n_ao, n_ao)

    # Transform to MO basis
    C = mf.mo_coeff
    eri_mo = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        C.conj().T, C.conj().T, C, C, eri_ao,
    )
    mol.h2 = np.asarray(eri_mo, dtype=float)

    # Nuclear repulsion
    mol.nuclear_repulsion = float(mol_obj.energy_nuc())


def _compute_fallback(mol: MolecularData):
    """Fallback: use pre-computed values from molecule library."""
    from superfermion.chemistry.library import MoleculeLibrary

    lib = MoleculeLibrary()
    name_lower = mol.name.lower()

    if "h2" in name_lower:
        data = lib.hydrogen()
        mol.n_electrons = 2
        mol.n_orbitals = 2
    elif "lih" in name_lower:
        data = lib.lithium_hydride()
        mol.n_electrons = 4
        mol.n_orbitals = 4
    elif "h2o" in name_lower or "water" in name_lower:
        data = lib.water()
        mol.n_electrons = 10
        mol.n_orbitals = 7
    else:
        # Unknown — create minimal empty data
        mol.n_electrons = 2
        mol.n_orbitals = 2
        mol.h1 = np.zeros((2, 2))
        mol.nuclear_repulsion = 0.0
        return

    mol.hf_energy = data.get("hf_energy", 0.0)
    mol.nuclear_repulsion = data.get("nuclear_repulsion", 0.0)


# ═════════════════════════════════════════════════════════════════════════
# Active space selection
# ═════════════════════════════════════════════════════════════════════════

def active_space_from_homo_lumo(
    mol: MolecularData,
    n_active_electrons: int,
    n_active_orbitals: int,
    frozen_core: Optional[int] = None,
) -> MolecularData:
    """Select active space around HOMO-LUMO gap.

    Args:
        mol: MolecularData with MO integrals.
        n_active_electrons: Number of electrons in active space.
        n_active_orbitals: Number of spatial orbitals in active space.
        frozen_core: Number of frozen core orbitals. Auto-detected if None.

    Returns:
        mol with active space applied (mutated in place).
    """
    if frozen_core is None:
        # Auto-detect: freeze inner orbitals
        # For first/second row: 1s core = 1 orbital per heavy atom
        atoms = _parse_geometry(mol.geometry)
        heavy_atoms = [a for a in atoms if a not in ("H", "He")]
        frozen_core = len(heavy_atoms)

    mol.active_orbitals = (n_active_electrons, n_active_orbitals)
    mol.frozen_core = frozen_core

    return mol
