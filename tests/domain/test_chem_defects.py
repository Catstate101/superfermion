"""Regressions for two defects that shipped silently in v0.1.21.

Both produced plausible numbers, which is why each one is pinned by a test that
fails against the pre-fix code rather than by a comment:

1. ``MolecularData.to_hamiltonian`` appended the nuclear-repulsion constant as a
   *second* all-identity ``PauliString`` although the mapper had already emitted
   one.  H2 therefore came out as 16 terms for 15 distinct strings and water as
   2111 for 2110.  Every energy stayed correct (the coefficients sum to the same
   number), so only a label-keyed consumer or a term count noticed.
2. The no-PySCF fallback matched ``"h2"`` before ``"h2o"``, so ``name="h2o"``
   received hydrogen's 2 electrons / 2 orbitals; the unmatched arm returned
   zeros; and ``hf_energy`` was ``0.0`` for every record because
   ``chemistry/library.py`` was a placeholder quoting "mock coefficients for
   demonstration".
"""

import numpy as np
import pytest

from superfermion.chemistry import subspace as sub
from superfermion.chemistry.pyscf_bridge import (
    MolecularData,
    _HAS_PYSCF,
    _fold_constant,
    _spin_orbital_integrals,
    molecule_from_geometry,
)

pytestmark = pytest.mark.domain

_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _synthetic(n_orbs: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_orbs, n_orbs))
    h1 = a + a.T
    raw = rng.normal(size=(n_orbs,) * 4) / 4.0
    h2 = sum(raw.transpose(p) for p in _PERMUTATIONS) / 8.0
    return h1, h2


def _mol(n_orbs=3, n_elec=4, enuc=0.0):
    h1, h2 = _synthetic(n_orbs)
    return MolecularData(
        name="model",
        geometry="H 0 0 0",
        h1=h1,
        h2=h2,
        nuclear_repulsion=enuc,
        n_orbitals=n_orbs,
        n_electrons=n_elec,
    )


class TestSingleIdentityTerm:
    @pytest.mark.parametrize("mapping", ["jordan_wigner", "bravyi_kitaev"])
    def test_no_label_appears_twice(self, mapping):
        """A term list is a basis expansion, so labels must be unique."""
        mol = _mol(enuc=0.7137)
        ham = mol.to_hamiltonian(mapping=mapping)
        labels = [t.pauli_str for t in ham.terms]
        assert len(set(labels)) == len(labels)

    def test_the_constant_lands_in_the_identity_coefficient(self):
        """The identity coefficient must be the sum, not one of two entries."""
        mol = _mol(enuc=0.7137)
        ham = mol.to_hamiltonian(term_tolerance=0.0)

        identity = "I" * (2 * mol.n_orbitals)
        matches = [t for t in ham.terms if t.pauli_str == identity]
        assert len(matches) == 1

        # Reproduce the same operator with the constant supplied separately.
        raw = _fold_constant(ham, -mol.nuclear_repulsion, 2 * mol.n_orbitals)
        from superfermion.chemistry.hamiltonians import FermionicOperator

        h1_so, h2_so = _spin_orbital_integrals(mol.h1, mol.h2)
        bare = FermionicOperator.from_coeffs(h1_so, h2_so).jordan_wigner(
            2 * mol.n_orbitals, tol=0.0
        )
        bare_id = [t for t in bare.terms if t.pauli_str == identity][0]
        assert float(np.real(matches[0].coeffs)) == pytest.approx(
            float(np.real(bare_id.coeffs)) + mol.nuclear_repulsion, abs=1e-12
        )
        assert float(np.real(raw.terms[0].coeffs)) == pytest.approx(
            float(np.real(bare_id.coeffs)), abs=1e-12
        )

    def test_energy_is_unchanged_by_the_merge(self):
        """Folding must not move the spectrum, only the term bookkeeping."""
        mol = _mol(n_orbs=2, n_elec=2, enuc=0.7199)
        ham = mol.to_hamiltonian(term_tolerance=0.0)

        exact = sub.exact_energy(
            mol.h1, mol.h2, 2, 2, nuclear_repulsion=mol.nuclear_repulsion
        )
        matrix = sub.pauli_matrix(ham)
        lowest = float(np.linalg.eigvalsh(matrix).min())
        assert lowest == pytest.approx(exact, abs=1e-10)

    def test_a_hamiltonian_without_an_identity_gains_one(self):
        from superfermion.observables.core import Hamiltonian, PauliString

        ham = Hamiltonian([PauliString("ZI", 2.0), PauliString("IZ", -1.0)])
        out = _fold_constant(ham, 5.0, 2)
        assert len(out.terms) == 3
        assert [t.coeffs for t in out.terms if t.pauli_str == "II"] == [5.0]

    def test_a_non_finite_constant_is_rejected(self):
        from superfermion.observables.core import Hamiltonian, PauliString

        ham = Hamiltonian([PauliString("II", 1.0)])
        with pytest.raises(ValueError, match="finite"):
            _fold_constant(ham, float("nan"), 2)

    def test_folding_zero_is_a_no_op(self):
        from superfermion.observables.core import Hamiltonian, PauliString

        ham = Hamiltonian([PauliString("II", 1.0), PauliString("ZZ", 2.0)])
        assert _fold_constant(ham, 0.0, 2) is ham


no_pyscf = pytest.mark.skipif(
    _HAS_PYSCF, reason="PySCF is installed, so the tabulated fallback never runs"
)


class TestTabulatedFallback:
    @no_pyscf
    def test_h2_matches_the_record(self):
        mol = molecule_from_geometry("H 0 0 0; H 0 0 0.735", basis="sto-3g")
        assert (mol.n_electrons, mol.n_orbitals) == (2, 2)
        assert mol.hf_energy == pytest.approx(-1.116998996754, abs=1e-12)

        exact = sub.exact_energy(
            mol.h1, mol.h2, 2, 2, nuclear_repulsion=mol.nuclear_repulsion
        )
        assert exact == pytest.approx(-1.137306035753, abs=1e-9)

    @no_pyscf
    def test_water_never_resolves_to_hydrogen(self):
        """The pre-fix matcher returned 2 electrons / 2 orbitals for 'h2o'."""
        with pytest.raises(LookupError, match="not in it|no integrals"):
            molecule_from_geometry(
                "O 0 0 0; H 0 0 0.9584; H 0.9311 0 -0.2410",
                basis="sto-3g",
                name="h2o",
            )

    @no_pyscf
    def test_an_unmatched_geometry_raises_instead_of_zero_filling(self):
        with pytest.raises(LookupError):
            molecule_from_geometry("H 0 0 0; H 0 0 0.9", basis="sto-3g")

    @no_pyscf
    def test_a_different_basis_is_not_silently_served(self):
        with pytest.raises(LookupError):
            molecule_from_geometry("H 0 0 0; H 0 0 0.735", basis="6-31g")

    @no_pyscf
    def test_an_ion_is_not_silently_served(self):
        with pytest.raises(LookupError):
            molecule_from_geometry("H 0 0 0; H 0 0 0.735", charge=1, spin=1)

    @no_pyscf
    def test_atom_order_and_trailing_zeros_do_not_matter(self):
        mol = molecule_from_geometry(
            "H 0.0000 0 0.735; H 0 0 -0.0000", basis="STO-3G"
        )
        assert mol.hf_energy == pytest.approx(-1.116998996754, abs=1e-12)

    def test_a_hand_built_mol_without_integrals_refuses_to_invent_them(self):
        mol = MolecularData(name="mystery", geometry="Xe 0 0 0")
        with pytest.raises(LookupError, match="no integrals"):
            mol.to_hamiltonian()
