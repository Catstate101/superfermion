"""Bravyi-Kitaev, SCBK, and the fermion-to-qubit mapping contract.

The mapping tests compare the qubit spectrum against the *fermionic* operator
assembled independently in :mod:`superfermion.chemistry.subspace`.  Agreement
with Jordan-Wigner alone would not catch a shared mistake in the integral
layout; agreement with the Fock-space matrix does.

These tests exist because ``bravyi_kitaev`` once put ``X``/``Y`` on the mode's
own site and swapped the update and parity sets.  It produced a Hermitian
operator with plausible-looking terms, a spectrum 1.078 Ha away from the
Fock-space one on the two-orbital model built below, and no conserved parities
at all, so SCBK could not run.  The figure comes from running the committed
blob of ``hamiltonians.py`` against these integrals.
"""

import numpy as np
import pytest

from superfermion.chemistry import subspace as sub
from superfermion.chemistry.hamiltonians import FermionicOperator
from superfermion.chemistry.pyscf_bridge import _spin_orbital_integrals
from superfermion.chemistry.scbk import scbk_from_integrals
from superfermion.chemistry.tapering import z2_symmetries

pytestmark = pytest.mark.domain


# The eight index permutations a chemist-ordered two-electron integral obeys.
_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _integrals(n_orbs: int = 2, seed: int = 0):
    """Synthetic integrals with all eight symmetries and real correlation."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_orbs, n_orbs))
    h1 = a + a.T
    raw = rng.normal(size=(n_orbs,) * 4) / 4.0
    h2 = sum(raw.transpose(p) for p in _PERMUTATIONS) / 8.0
    return h1, h2


@pytest.fixture(scope="module")
def model():
    h1, h2 = _integrals()
    h1_so, h2_so = _spin_orbital_integrals(h1, h2)
    return h1, h2, h1_so, h2_so


class TestFermionToQubitMapping:
    """Both encoders must be the same operator as the fermionic one."""

    @staticmethod
    def _fock_spectrum(h1_so, h2_so):
        terms = sub.fermionic_terms(h1_so, h2_so)
        dets = list(range(1 << h1_so.shape[0]))
        matrix = sub.hamiltonian_matrix(terms, dets, sparse=False)
        return np.sort(np.linalg.eigvalsh(np.asarray(matrix, dtype=float)))

    def test_bravyi_kitaev_reproduces_the_fock_space_spectrum(self, model):
        _, _, h1_so, h2_so = model
        n_modes = h1_so.shape[0]
        op = FermionicOperator.from_coeffs(h1_so, h2_so)
        expected = self._fock_spectrum(h1_so, h2_so)
        got = np.sort(np.linalg.eigvalsh(sub.pauli_matrix(op.bravyi_kitaev(n_modes))))
        assert got.shape == expected.shape
        assert np.allclose(got, expected, atol=1e-10)

    def test_jordan_wigner_reproduces_the_fock_space_spectrum(self, model):
        _, _, h1_so, h2_so = model
        n_modes = h1_so.shape[0]
        op = FermionicOperator.from_coeffs(h1_so, h2_so)
        expected = self._fock_spectrum(h1_so, h2_so)
        got = np.sort(np.linalg.eigvalsh(sub.pauli_matrix(op.jordan_wigner(n_modes))))
        assert np.allclose(got, expected, atol=1e-10)

    def test_the_two_encoders_agree_with_each_other(self, model):
        _, _, h1_so, h2_so = model
        n_modes = h1_so.shape[0]
        op = FermionicOperator.from_coeffs(h1_so, h2_so)
        jw = np.sort(np.linalg.eigvalsh(sub.pauli_matrix(op.jordan_wigner(n_modes))))
        bk = np.sort(np.linalg.eigvalsh(sub.pauli_matrix(op.bravyi_kitaev(n_modes))))
        assert np.allclose(jw, bk, atol=1e-10)

    def test_bravyi_kitaev_exposes_the_two_parity_symmetries(self, model):
        """SCBK needs two {I,Z} generators; a wrong update set leaves none."""
        _, _, h1_so, h2_so = model
        n_modes = h1_so.shape[0]
        op = FermionicOperator.from_coeffs(h1_so, h2_so)
        found = z2_symmetries(op.bravyi_kitaev(n_modes))
        assert len(found) >= 2
        assert all(set(s) <= {"I", "Z"} for s in found)
        assert all(len(s) == n_modes for s in found)

    def test_a_number_operator_stays_diagonal_under_bravyi_kitaev(self):
        """``n_3`` in a four-mode register must be a product of Z's alone.

        The Fenwick tree fixes this without any choice: qubit 3 carries the
        parity of modes 1, 2 and 3, so ``n_3 = (1 - Z_1 Z_2 Z_3) / 2``.
        """
        h1_so = np.zeros((4, 4))
        h1_so[3, 3] = 1.0
        op = FermionicOperator.from_coeffs(h1_so, None)
        mapped = op.bravyi_kitaev(4)
        assert len(list(mapped.terms)) > 0
        assert all(set(term.pauli_str) <= {"I", "Z"} for term in mapped.terms)

        diagonal = np.real(np.diag(sub.pauli_matrix(mapped)))
        assert np.allclose(diagonal, np.round(diagonal), atol=1e-10)
        assert np.allclose(diagonal, 0.0) is False
        # half of the sixteen basis states have mode 3 occupied
        assert int(np.sum(np.round(diagonal))) == 8

    def test_a_mode_outside_the_register_is_rejected(self, model):
        _, _, h1_so, h2_so = model
        op = FermionicOperator.from_coeffs(h1_so, h2_so)
        with pytest.raises(ValueError):
            op.bravyi_kitaev(h1_so.shape[0] - 1)


class TestSCBK:
    """Two qubits are removed and the four sectors must tile the spectrum."""

    def test_exactly_two_qubits_are_removed(self, model):
        h1, h2 = model[0], model[1]
        result = scbk_from_integrals(h1, h2)
        assert result["n_qubits_before"] == 4
        assert result["n_qubits_after"] == 2
        assert len(result["symmetries"]) == 2
        assert result["n_qubits_after"] == result["n_qubits_before"] - 2

    def test_the_four_sectors_are_all_present(self, model):
        result = scbk_from_integrals(model[0], model[1])
        assert set(result["hamiltonians"]) == {
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        }
        for hamiltonian in result["hamiltonians"].values():
            widths = {len(term.pauli_str) for term in hamiltonian.terms}
            assert widths == {2}

    def test_the_sectors_tile_the_bravyi_kitaev_spectrum(self, model):
        h1, h2, h1_so, h2_so = model
        result = scbk_from_integrals(h1, h2)
        tiled = np.sort(np.concatenate([
            np.linalg.eigvalsh(sub.pauli_matrix(h))
            for h in result["hamiltonians"].values()
        ]))
        op = FermionicOperator.from_coeffs(h1_so, h2_so)
        full = np.sort(np.linalg.eigvalsh(sub.pauli_matrix(op.bravyi_kitaev(4))))
        assert tiled.shape == full.shape
        assert np.allclose(tiled, full, atol=1e-9)

    def test_the_reported_sector_energies_match_that_tiling(self, model):
        result = scbk_from_integrals(model[0], model[1])
        reported = np.sort([result["energies"][sector] for sector in result["energies"]])
        recomputed = np.sort([
            np.linalg.eigvalsh(sub.pauli_matrix(h))[0]
            for h in result["hamiltonians"].values()
        ])
        assert np.allclose(reported, recomputed, atol=1e-9)

    def test_the_energy_offset_is_added_to_every_sector(self, model):
        """The operator is electronic only, so E_nuc has to enter as an offset."""
        plain = scbk_from_integrals(model[0], model[1])
        shifted = scbk_from_integrals(model[0], model[1], energy_offset=0.7151)
        for sector, value in plain["energies"].items():
            assert shifted["energies"][sector] == pytest.approx(value + 0.7151)

    def test_skipping_the_spectrum_reports_no_energies(self, model):
        """``compute_spectrum=False`` is the documented way to pay nothing."""
        result = scbk_from_integrals(
            model[0], model[1], compute_spectrum=False, energy_offset=0.7151
        )
        assert set(result["energies"].values()) == {None}
        assert set(result["hamiltonians"]) == {(-1, -1), (-1, 1), (1, -1), (1, 1)}

    def test_asking_for_more_qubits_than_the_encoding_offers_fails(self, model):
        with pytest.raises(ValueError, match="SCBK needs"):
            scbk_from_integrals(model[0], model[1], n_symmetries=10)
