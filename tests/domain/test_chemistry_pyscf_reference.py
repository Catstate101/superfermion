"""Chemistry checked against frozen PySCF references.

PySCF publishes no Windows wheel, so the PySCF-backed chemistry path could only
ever be exercised on Linux, and the tests that do touch it are skipped on
Windows.  ``pyscf_sto3g_reference.json`` holds the MO integrals and scalar
energies that PySCF 2.14.0 produced for six STO-3G molecules inside WSL --
``_pyscf_ref.py`` imports PySCF only, so no superfermion arithmetic is involved
in producing them.  Replaying those integrals here makes the correlated-method
path testable on every platform, and pins it to an external implementation
instead of to superfermion's own earlier output.

What this covers that nothing else did:

* FCI, MP2 and Hartree-Fock on real (not tabulated) MO integrals;
* the MO-basis convention itself -- chemists' notation, spatial orbitals,
  nuclear repulsion added once;
* the tabulated fallback that is served when PySCF is absent, checked against
  what PySCF actually computes;
* :func:`superfermion.chemistry.entropies_from_integrals`, which had no test
  coverage at all.
"""

import functools
import itertools
import json
import pathlib

import numpy as np
import pytest

import superfermion.chemistry as chem
from superfermion.chemistry import mp2
from superfermion.chemistry.pyscf_bridge import MolecularData

pytestmark = pytest.mark.domain

FIXTURE = pathlib.Path(__file__).with_name("pyscf_sto3g_reference.json")
_RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
MOLECULES = _RAW["molecules"]

# Molecules with a correlated reference (H2+ is a one-electron doublet: no FCI
# or MP2 number was produced for it, so it only feeds the integral checks).
CORRELATED = sorted(n for n, rec in MOLECULES.items() if "fci_energy" in rec)
CLOSED_SHELL = sorted(
    n for n, rec in MOLECULES.items() if rec["spin"] == 0 and rec["nelectron"] % 2 == 0
)
# Molecules small enough to diagonalise repeatedly without dominating the suite.
CHEAP = ("H2", "HeH+", "LiH")


def _record(name):
    return MOLECULES[name]


def _integrals(rec):
    return np.asarray(rec["h1"], dtype=float), np.asarray(rec["h2"], dtype=float)


def _n_occupied(rec):
    """Occupied *spatial* orbitals.

    PySCF's ``mo_occ`` counts electrons per orbital (``[2.0, 0.0, ...]``), not
    orbitals, so the naive ``sum(1 for o in mo_occ if o > 0)`` overcounts by a
    factor of two on a closed shell.
    """
    return int(round(sum(rec["mo_occ"]) / 2))


def _rhf_energy(rec):
    """Closed-shell RHF energy from MO-basis integrals in chemists' notation.

    Written out here rather than called from superfermion on purpose: the point
    is to check that ``h1``/``h2`` are the quantities PySCF says they are, so
    the formula has to be independent of superfermion's own integral handling.
    """
    h1, h2 = _integrals(rec)
    n_occ = _n_occupied(rec)
    two_electron = 0.0
    for i, j in itertools.product(range(n_occ), repeat=2):
        two_electron += 2.0 * h2[i, i, j, j] - h2[i, j, j, i]
    return (
        2.0 * float(np.trace(h1[:n_occ, :n_occ])) + two_electron + float(rec["nuclear_repulsion"])
    )


@functools.cache
def _fci(name):
    """``exact_energy`` including nuclear repulsion, computed once per molecule.

    H2O takes several seconds, so every assertion that needs it shares this.
    """
    rec = _record(name)
    h1, h2 = _integrals(rec)
    return chem.exact_energy(
        h1,
        h2,
        int(rec["nao"]),
        int(rec["nelectron"]),
        nuclear_repulsion=float(rec["nuclear_repulsion"]),
    )


@functools.cache
def _entropies(name):
    """``(s_i, mutual_information, energy)`` including nuclear repulsion."""
    rec = _record(name)
    h1, h2 = _integrals(rec)
    return chem.entropies_from_integrals(
        h1,
        h2,
        int(rec["nao"]),
        int(rec["nelectron"]),
        nuclear_repulsion=float(rec["nuclear_repulsion"]),
    )


def test_the_fixture_is_a_consistent_fock_representation():
    """Guard the fixture itself before any assertion leans on it."""
    assert len(MOLECULES) == 6
    assert _RAW["_pyscf_version"].startswith("2.")
    for name, rec in MOLECULES.items():
        h1, h2 = _integrals(rec)
        n = int(rec["nao"])
        assert h1.shape == (n, n), name
        assert h2.shape == (n, n, n, n), name
        assert len(rec["mo_energies"]) == n, name
        assert np.max(np.abs(h1 - h1.T)) < 1e-10, name
        # Chemists' notation (pq|rs): symmetric under p<->q, r<->s and (pq)<->(rs).
        assert np.max(np.abs(h2 - h2.transpose(1, 0, 2, 3))) < 1e-10, name
        assert np.max(np.abs(h2 - h2.transpose(0, 1, 3, 2))) < 1e-10, name
        assert np.max(np.abs(h2 - h2.transpose(2, 3, 0, 1))) < 1e-10, name


class TestFciAgainstPySCF:
    @pytest.mark.parametrize("name", CORRELATED)
    def test_total_energy_matches_pyscf(self, name):
        assert float(_fci(name)) == pytest.approx(_record(name)["fci_energy"], abs=1e-9)

    def test_omitting_nuclear_repulsion_returns_the_electronic_part(self):
        """The default must stay electronic-only, as it always was."""
        rec = _record("H2")
        h1, h2 = _integrals(rec)
        got = chem.exact_energy(h1, h2, int(rec["nao"]), int(rec["nelectron"]))
        assert float(got) == pytest.approx(rec["fci_energy"] - rec["nuclear_repulsion"], abs=1e-9)


class TestMp2AgainstPySCF:
    @pytest.mark.parametrize("name", CORRELATED)
    def test_correlation_energy_matches_pyscf(self, name):
        rec = _record(name)
        h1, h2 = _integrals(rec)
        got = mp2.mp2_correlation_energy(h1, h2, np.asarray(rec["mo_energies"]), _n_occupied(rec))
        assert float(got) == pytest.approx(rec["mp2_correlation"], abs=1e-9)

    @pytest.mark.parametrize("name", CORRELATED)
    def test_correlation_is_negative_and_lies_between_hf_and_fci(self, name):
        """MP2 must recover part of the correlation energy, never all of it.

        Energies are negative, so the ordering is FCI < MP2 < HF.
        """
        rec = _record(name)
        h1, h2 = _integrals(rec)
        corr = float(
            mp2.mp2_correlation_energy(h1, h2, np.asarray(rec["mo_energies"]), _n_occupied(rec))
        )
        assert corr < 0.0
        assert rec["hf_energy"] + corr > rec["fci_energy"]
        assert rec["hf_energy"] + corr < rec["hf_energy"]


class TestHartreeFockFromIntegrals:
    """The MO-basis convention: chemists' notation, spatial orbitals, one E_nuc."""

    @pytest.mark.parametrize("name", CLOSED_SHELL)
    def test_closed_shell_energy_matches_pyscf(self, name):
        assert _rhf_energy(_record(name)) == pytest.approx(_record(name)["hf_energy"], abs=1e-12)


class TestTabulatedFallback:
    def test_the_served_h2_record_matches_pyscf(self):
        """The no-PySCF path serves a table; the table must be PySCF's numbers.

        With PySCF installed this compares a live run against the frozen one,
        which is the same claim; without it, the table is being checked.
        """
        rec = _record("H2")
        mol = chem.molecule_from_geometry("H 0 0 0; H 0 0 0.735", basis="sto-3g")

        assert mol.n_orbitals == rec["nao"]
        assert mol.n_electrons == rec["nelectron"]
        assert float(mol.hf_energy) == pytest.approx(rec["hf_energy"], abs=1e-9)
        assert float(mol.nuclear_repulsion) == pytest.approx(rec["nuclear_repulsion"], abs=1e-9)
        assert np.max(np.abs(np.asarray(mol.h1) - np.asarray(rec["h1"]))) < 1e-9
        assert np.max(np.abs(np.asarray(mol.h2) - np.asarray(rec["h2"]))) < 1e-9


class TestMolecularDataFromRealIntegrals:
    def test_the_pauli_hamiltonian_is_the_expected_four_qubit_h2(self):
        """Two spatial orbitals become four qubits with the canonical terms."""
        rec = _record("H2")
        h1, h2 = _integrals(rec)
        mol = MolecularData(
            name="H2_pyscf",
            geometry=rec["geometry"],
            basis="sto-3g",
            h1=h1,
            h2=h2,
            nuclear_repulsion=float(rec["nuclear_repulsion"]),
            hf_energy=float(rec["hf_energy"]),
            n_orbitals=int(rec["nao"]),
            n_electrons=int(rec["nelectron"]),
            mo_energies=np.asarray(rec["mo_energies"]),
        )

        assert (mol.n_qubits, mol.n_electrons) == (4, 2)
        ham = mol.to_hamiltonian()
        strings = {term.pauli_str for term in ham.terms}
        assert len(ham.terms) == 15
        # The number-conserving spin-flip block is what the interleaved
        # (2p + spin) orbital ordering produces; a different ordering shows up
        # here as a different set of four-qubit strings.
        assert {"XXYY", "XYYX", "YXXY", "YYXX"} <= strings
        assert all(set(s) <= {"I", "X", "Y", "Z"} for s in strings)


class TestEntropiesFromIntegrals:
    """``entropies_from_integrals`` had no test coverage before this file."""

    @pytest.mark.parametrize("name", CORRELATED)
    def test_entropies_are_bounded_by_the_orbital_dimension(self, name):
        rec = _record(name)
        s_i = np.asarray(_entropies(name)[0])
        assert s_i.shape == (int(rec["nao"]),)
        # s_i is the entropy of a 4x4 reduced density matrix: in [0, ln 4].
        assert np.all(s_i >= -1e-12)
        assert np.all(s_i <= np.log(4) + 1e-12)

    @pytest.mark.parametrize("name", CORRELATED)
    def test_mutual_information_is_symmetric_with_a_zero_diagonal(self, name):
        n = int(_record(name)["nao"])
        mi = np.asarray(_entropies(name)[1])
        assert mi.shape == (n, n)
        assert np.max(np.abs(mi - mi.T)) < 1e-12
        assert np.max(np.abs(np.diag(mi))) < 1e-12
        # Subadditivity makes I_ij = (s_i + s_j - s_ij) / 2 non-negative.
        assert np.all(mi >= -1e-12)

    def test_h2_mutual_information_equals_its_orbital_entropy(self):
        """For two orbitals the state is pure, so s_ij = 0 and I_01 = s_0.

        This is what the factor of 1/2 in the documented definition buys: the
        pair correlation reproduces the single-orbital value instead of twice
        it.  With a plain ``s_i + s_j - s_ij`` this assertion fails by 2x.
        """
        s_i, mi, _ = _entropies("H2")
        s_i, mi = np.asarray(s_i), np.asarray(mi)
        assert s_i[0] == pytest.approx(s_i[1], abs=1e-12)
        assert mi[0, 1] == pytest.approx(s_i[0], abs=1e-12)

    def test_h2_orbital_entropy_is_the_known_fci_value(self):
        """Regression pin: H2/STO-3G ground-state entanglement between orbitals."""
        s_i, _, _ = _entropies("H2")
        assert float(np.asarray(s_i)[0]) == pytest.approx(0.066935729, abs=1e-8)

    def test_degenerate_orbitals_carry_equal_entropy(self):
        """LiH's two pi orbitals are exactly degenerate, so s_3 must equal s_4."""
        s_i = np.asarray(_entropies("LiH")[0])
        assert s_i[3] == pytest.approx(s_i[4], abs=1e-12)

    @pytest.mark.parametrize("name", CORRELATED)
    def test_the_returned_energy_agrees_with_exact_energy(self, name):
        """Two entry points, one Hamiltonian -- they must not drift apart."""
        _, _, energy = _entropies(name)
        assert float(energy) == pytest.approx(float(_fci(name)), abs=1e-10)
        assert float(energy) == pytest.approx(_record(name)["fci_energy"], abs=1e-9)

    @pytest.mark.parametrize("name", CHEAP)
    def test_nuclear_repulsion_shifts_the_energy_and_nothing_else(self, name):
        """Before the keyword existed the energy was electronic-only; the
        entropies themselves never depended on the nuclear term."""
        rec = _record(name)
        h1, h2 = _integrals(rec)
        n_orbitals, n_electrons = int(rec["nao"]), int(rec["nelectron"])
        bare = chem.entropies_from_integrals(h1, h2, n_orbitals, n_electrons)
        shifted = chem.entropies_from_integrals(
            h1,
            h2,
            n_orbitals,
            n_electrons,
            nuclear_repulsion=float(rec["nuclear_repulsion"]),
        )

        assert np.array_equal(np.asarray(bare[0]), np.asarray(shifted[0]))
        assert np.array_equal(np.asarray(bare[1]), np.asarray(shifted[1]))
        assert float(shifted[2]) - float(bare[2]) == pytest.approx(
            float(rec["nuclear_repulsion"]), abs=1e-12
        )
        assert float(bare[2]) == pytest.approx(
            rec["fci_energy"] - rec["nuclear_repulsion"], abs=1e-9
        )
