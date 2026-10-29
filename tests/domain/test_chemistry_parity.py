"""Tests for the chemistry surface added for Qiskit-Nature parity.

Everything here runs **without PySCF**: the molecules are synthetic integr/tensor
pairs with the eight-fold permutational symmetry a real two-electron integral
has, so the Jordan-Wigner operator still carries the particle-number parities
that the symmetry machinery keys on.  That keeps these tests runnable in the
plain Windows venv, which is where the rest of the suite runs.

The one exception is :class:`TestQiskitInterop`, which is skipped when qiskit is
not importable rather than silently passing.
"""

import numpy as np
import pytest

from superfermion.chemistry import apc
from superfermion.chemistry import bksf
from superfermion.chemistry import excited as exc
from superfermion.chemistry import interop
from superfermion.chemistry import mp2
from superfermion.chemistry import seniority as sen
from superfermion.chemistry import subspace as sub
from superfermion.chemistry import symmetry as sym
from superfermion.chemistry import tapering
from superfermion.chemistry.pyscf_bridge import MolecularData
from superfermion.observables.core import Hamiltonian, PauliString

pytestmark = pytest.mark.domain

_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _synthetic(n_orbs=3, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_orbs, n_orbs))
    h1 = a + a.T
    raw = rng.normal(size=(n_orbs,) * 4) / 4.0
    h2 = sum(raw.transpose(p) for p in _PERMUTATIONS) / 8.0
    return h1, h2


def _mol(n_orbs=3, n_elec=4, enuc=0.0, seed=0):
    h1, h2 = _synthetic(n_orbs, seed)
    rng = np.random.default_rng(seed + 1)
    energies = np.sort(rng.normal(size=n_orbs)) - 0.5
    # Aufbau occupations, because the MP2 routes read the mean field rather
    # than re-deriving it: leaving this unset is a caller error, not a molecule.
    filled = n_elec // 2
    occupations = np.array([2.0] * filled + [0.0] * (n_orbs - filled))
    return MolecularData(
        name="model",
        geometry="H 0 0 0",
        h1=h1,
        h2=h2,
        nuclear_repulsion=enuc,
        n_orbitals=n_orbs,
        n_electrons=n_elec,
        mo_energies=energies,
        mo_occupations=occupations,
    )


def _gapped_mol(n_orbs=3, n_elec=2, seed=2):
    """A reference MP2 can actually be trusted on.

    Well-separated orbital energies and a coupling scaled well below the gap, so
    the second-order correction really is a correction.  ``_mol`` draws random
    integrals and near-degenerate energies, which is fine for testing that a
    reduction is exact but is *not* a reference MP2 is perturbative about.
    """
    h1, h2 = _synthetic(n_orbs, seed)
    energies = np.linspace(-2.0, 3.5, n_orbs)
    filled = n_elec // 2
    occupations = np.array([2.0] * filled + [0.0] * (n_orbs - filled))
    return MolecularData(
        name="gapped",
        geometry="H 0 0 0",
        h1=h1,
        h2=0.05 * h2,
        nuclear_repulsion=0.0,
        n_orbitals=n_orbs,
        n_electrons=n_elec,
        mo_energies=energies,
        mo_occupations=occupations,
    )


def _degenerate_mol():
    """Nearly degenerate orbital energies: MP2 has no small parameter here."""
    h1, h2 = _synthetic(3, seed=0)
    return MolecularData(
        name="degenerate",
        geometry="H 0 0 0",
        h1=h1,
        h2=h2,
        nuclear_repulsion=0.0,
        n_orbitals=3,
        n_electrons=2,
        mo_energies=np.array([-1.0, -1.0 + 1e-4, -1.0 + 2e-4]),
        mo_occupations=np.array([2.0, 0.0, 0.0]),
    )


def _bits(index, n):
    """Basis bits of ``index``, character 0 = qubit 0 = most significant."""
    return [(index >> (n - 1 - q)) & 1 for q in range(n)]


def _index(bits):
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _rotate(bits, circuit):
    out = list(bits)
    for control, target in circuit:
        out[target] ^= out[control]
    return out


def _z_expect(bits, label):
    return -1.0 if sum(b for b, ch in zip(bits, label) if ch == "Z") % 2 else 1.0


def _aufbau(n_elec, n_orb):
    filled = n_elec // 2
    return [1] * (2 * filled) + [0] * (2 * (n_orb - filled))


def _sector_states(info):
    n_before = info["n_qubits_before"]
    return [
        (idx, _bits(idx, n_before))
        for idx in range(2 ** n_before)
        if all(
            _z_expect(_bits(idx, n_before), lab) == s
            for lab, s in zip(info["symmetries"], info["sector"])
        )
    ]


class TestTaperedInfoInverse:
    """``TaperedInfo`` must invert the reduction, not merely describe it."""

    @pytest.fixture()
    def tapered(self):
        mol = _mol(n_orbs=2, n_elec=2)
        ham = mol.to_hamiltonian()
        n = len(ham.terms[0].pauli_str)
        reference = _aufbau(2, 2)
        ham_t, info = tapering.taper_z2(ham, reference=reference)
        return ham, ham_t, info, n, reference

    def test_the_reduction_is_not_trivial(self, tapered):
        """A test on a system with no symmetry would pass vacuously."""
        _, _, info, n, _ = tapered
        assert info["symmetries"], "expected at least the particle-number parities"
        assert info["n_qubits_after"] < n

    def test_it_is_still_a_plain_dict(self, tapered):
        _, _, info, n, _ = tapered
        assert isinstance(info, dict)
        assert len(dict(info)) == len(info)
        assert set(info) == {
            "n_qubits_before", "n_qubits_after", "tapered_qubits",
            "symmetries", "sector", "clifford_gates", "terms_before",
            "terms_after",
        }
        assert info["n_qubits_before"] == n

    def test_keep_qubits_are_the_complement(self, tapered):
        _, _, info, n, _ = tapered
        assert info.keep_qubits == [
            q for q in range(n) if q not in set(info["tapered_qubits"])
        ]

    def test_unrotate_round_trips_every_sector_state(self, tapered):
        _, _, info, _, _ = tapered
        states = _sector_states(info)
        assert len(states) == 2 ** info["n_qubits_after"]

        seen = set()
        for _, bits in states:
            rotated = _rotate(bits, info["clifford_gates"])
            reduced = "".join(str(rotated[q]) for q in info.keep_qubits)
            seen.add(reduced)
            assert info.unrotate(reduced) == "".join(str(b) for b in bits)
        # A bijection, not just a map that lands somewhere valid.
        assert len(seen) == len(states)

    def test_unrotate_rejects_a_wrong_length(self, tapered):
        _, _, info, _, _ = tapered
        with pytest.raises(ValueError):
            info.unrotate("0" * (info["n_qubits_after"] + 1))

    def test_expand_observable_reproduces_the_sector_block(self, tapered):
        ham, ham_t, info, _, _ = tapered
        states = _sector_states(info)
        lifted = info.expand_observable(ham_t)
        m_full = sub.pauli_matrix(ham)
        m_lift = sub.pauli_matrix(lifted)
        m_tap = sub.pauli_matrix(ham_t)

        worst = 0.0
        for idx1, b1 in states:
            r1 = _rotate(b1, info["clifford_gates"])
            t1 = _index([r1[q] for q in info.keep_qubits])
            for idx2, b2 in states:
                r2 = _rotate(b2, info["clifford_gates"])
                t2 = _index([r2[q] for q in info.keep_qubits])
                worst = max(worst, abs(m_lift[idx1, idx2] - m_tap[t1, t2]))
                worst = max(worst, abs(m_lift[idx1, idx2] - m_full[idx1, idx2]))
        assert worst == pytest.approx(0.0, abs=1e-12)

    def test_a_single_qubit_observable_lifts_exactly(self, tapered):
        _, _, info, _, _ = tapered
        states = _sector_states(info)
        n_after = info["n_qubits_after"]
        for j in range(n_after):
            for pauli in ("Z", "X", "Y"):
                label = "".join(pauli if k == j else "I" for k in range(n_after))
                obs = Hamiltonian([PauliString(label, coeffs=1.0)])
                m_tap = sub.pauli_matrix(obs)
                m_lift = sub.pauli_matrix(info.expand_observable(obs))
                for idx1, b1 in states:
                    r1 = _rotate(b1, info["clifford_gates"])
                    t1 = _index([r1[q] for q in info.keep_qubits])
                    for idx2, b2 in states:
                        r2 = _rotate(b2, info["clifford_gates"])
                        t2 = _index([r2[q] for q in info.keep_qubits])
                        assert m_lift[idx1, idx2] == pytest.approx(
                            m_tap[t1, t2], abs=1e-12
                        )

    def test_no_symmetries_is_the_identity_map(self):
        ham = _mol(n_orbs=2, n_elec=2).to_hamiltonian()
        n = len(ham.terms[0].pauli_str)
        _, info = tapering.taper_z2(ham, reference=_aufbau(2, 2), symmetries=[])
        assert info["n_qubits_after"] == n
        assert info.keep_qubits == list(range(n))
        reference = "".join(str(b) for b in _aufbau(2, 2))
        assert info.unrotate(reference) == reference


class TestCoreDescription:
    """``core_description`` must name the convention that will actually run."""

    def test_full_space(self):
        mol = _mol(n_orbs=4, n_elec=4)
        info = mol.core_description()
        assert info["mode"] == "full_space"
        assert info["n_core_orbitals"] == 0
        assert info["n_active_orbitals"] == 4

    def test_frozen_core(self):
        mol = _mol(n_orbs=4, n_elec=4)
        mol.frozen_core = 1
        info = mol.core_description()
        assert info["mode"] == "frozen_core"
        assert info["n_core_orbitals"] == 1
        assert info["n_active_orbitals"] == 3

    def test_active_space_window_counts_pairs(self):
        mol = _mol(n_orbs=4, n_elec=4)
        mol.active_orbitals = (2, 2)
        info = mol.core_description()
        assert info["mode"] == "active_space"
        assert info["n_core_orbitals"] == 1

    def test_a_window_overrides_frozen_core_and_says_so(self):
        mol = _mol(n_orbs=4, n_elec=4)
        mol.active_orbitals = (2, 2)
        mol.frozen_core = 2
        info = mol.core_description()
        assert info["mode"] == "active_space"
        assert info["n_core_orbitals"] == 1
        assert "IGNORED" in info["note"]
        assert "frozen_core=2" in info["note"]

    def test_explicit_indices(self):
        mol = _mol(n_orbs=4, n_elec=4)
        mol.active_indices = [1, 2]
        info = mol.core_description()
        assert info["mode"] == "explicit"
        assert info["n_core_orbitals"] == 0
        assert info["n_active_orbitals"] == 2

    def test_summary_reports_the_same_answer(self):
        mol = _mol(n_orbs=4, n_elec=4)
        mol.active_orbitals = (2, 2)
        summary = mol.summary()
        info = mol.core_description()
        assert summary["core_mode"] == info["mode"]
        assert summary["core_orbitals"] == info["n_core_orbitals"]
        assert summary["core_note"] == info["note"]

    def test_it_agrees_with_the_qubit_count_to_hamiltonian_builds(self):
        mol = _mol(n_orbs=4, n_elec=4)
        mol.active_orbitals = (2, 2)
        ham = mol.to_hamiltonian()
        info = mol.core_description()
        assert len(ham.terms[0].pauli_str) == 2 * info["n_active_orbitals"]


class TestMP2Activity:
    def test_entropy_vanishes_for_full_and_empty(self):
        s = mp2.mp2_no_activity(np.array([2.0, 2.0, 0.0, 0.0]))
        assert np.allclose(s, 0.0, atol=1e-15)

    def test_entropy_is_maximal_at_half_filling(self):
        s = mp2.mp2_no_activity(np.array([1.0]))
        assert float(s[0]) == pytest.approx(np.log(2.0), abs=1e-12)

    def test_entropy_is_symmetric_about_half_filling(self):
        s = mp2.mp2_no_activity(np.array([0.3, 1.7]))
        assert float(s[0]) == pytest.approx(float(s[1]), abs=1e-12)

    def test_the_old_abs_deviation_ranking_was_degenerate(self):
        """``|n - 1|`` scores a full and an empty orbital identically."""
        occupations = np.array([2.0, 0.0, 1.2])
        assert np.allclose(np.abs(occupations - 1.0), [1.0, 1.0, 0.2])
        rank = mp2.mp2_no_rank(occupations)
        assert list(rank) == [2, 0, 1]  # only the fractionally occupied one scores

    def test_rank_is_by_descending_activity(self):
        occupations = np.array([2.0, 1.9, 0.6, 0.0])
        assert list(mp2.mp2_no_rank(occupations)) == [2, 1, 0, 3]

    @pytest.mark.parametrize("n_cas", [1, 2, 3, 4])
    def test_the_core_is_always_the_prefix_below_the_window(self, n_cas):
        """``_apply_active_space`` takes ``core = range(ncore)``, so it must be."""
        mol = _gapped_mol(4, 4)
        space = mp2.active_space_from_mp2_no(mol, n_cas=n_cas)
        active = space["active_indices"]
        assert space["n_frozen_core"] == min(active)
        assert max(active) < mol.n_orbitals
        assert len(active) == n_cas

    def test_the_full_window_is_the_whole_space(self):
        mol = _gapped_mol(4, 4)
        space = mp2.active_space_from_mp2_no(mol, n_cas=4)
        assert list(space["active_indices"]) == [0, 1, 2, 3]
        assert space["n_frozen_core"] == 0

    def test_a_bad_n_cas_is_rejected(self):
        mol = _gapped_mol(4, 4)
        with pytest.raises(ValueError, match="n_cas"):
            mp2.active_space_from_mp2_no(mol, n_cas=0)
        with pytest.raises(ValueError, match="n_cas"):
            mp2.active_space_from_mp2_no(mol, n_cas=5)

    def test_the_natural_orbital_basis_preserves_the_energy_sum(self):
        """``(R**2).T @ eps`` is the Fock diagonal; ``eps @ R`` is not.

        Both are orthogonal transforms of the same spectrum, so the *sum* of the
        orbital energies is the invariant that separates them: the Gram-weighted
        form preserves it exactly, the left-multiplied form does not.
        """
        mol = _gapped_mol(3, 2)
        no = mp2.mp2_natural_orbitals(mol)
        assert float(np.sum(no["mo_energies"])) == pytest.approx(
            float(np.sum(mol.mo_energies)), abs=1e-10
        )
        assert not np.allclose(
            float(np.sum(mol.mo_energies @ no["rotation"])),
            float(np.sum(mol.mo_energies)),
            atol=1e-12,
        )

    def test_the_natural_orbital_molecule_is_a_consistent_copy(self):
        mol = _gapped_mol(3, 2)
        rotated, no = mp2.mp2_no_molecule(mol)
        assert rotated is not mol
        # Everything that is not a basis-dependent array has to survive.
        assert rotated.n_electrons == mol.n_electrons
        assert rotated.nuclear_repulsion == mol.nuclear_repulsion
        assert rotated.name == mol.name
        # The arrays that are basis dependent must all have been transformed.
        assert np.shape(rotated.h1) == np.shape(mol.h1)
        assert np.shape(rotated.h2) == np.shape(mol.h2)
        assert np.shape(rotated.mo_occupations) == (3,)
        assert np.all(np.diff(rotated.mo_occupations) <= 1e-12)  # descending
        assert float(np.sum(rotated.mo_occupations)) == pytest.approx(2.0, abs=1e-8)
        assert float(np.sum(rotated.mo_energies)) == pytest.approx(
            float(np.sum(mol.mo_energies)), abs=1e-10
        )
        # A stale active space would index the new ordering with old positions.
        assert rotated.active_indices is None
        assert rotated.active_orbitals is None
        assert no["rotation"].shape == (3, 3)

    def test_the_natural_orbital_basis_does_not_move_the_fci_energy(self):
        """A unitary rotation of the MO basis cannot change the spectrum."""
        mol = _gapped_mol(3, 2)
        rotated, _ = mp2.mp2_no_molecule(mol)
        before = sub.exact_energy(mol.h1, mol.h2, 3, 2)
        after = sub.exact_energy(rotated.h1, rotated.h2, 3, 2)
        assert after == pytest.approx(before, abs=1e-9)

    def test_amplitudes_and_the_correlation_energy_agree(self):
        mol = _gapped_mol(3, 2)
        data = mp2.mp2_amplitudes(mol.h1, mol.h2, mol.mo_energies, 1)
        assert data["occ"] == [0]
        assert data["virt"] == [1, 2]
        assert data["core"] == []
        assert data["n_occ_spin"] == 2 and data["n_virt_spin"] == 4
        # Every term is ``-|V|^2 / gap``, so a gapped closed shell is negative.
        assert data["e2"] < 0.0
        assert mp2.mp2_correlation_energy(
            mol.h1, mol.h2, mol.mo_energies, 1
        ) == pytest.approx(data["e2"], abs=1e-14)

    def test_an_open_shell_reference_is_refused_by_name(self):
        mol = _gapped_mol(3, 2)
        with pytest.raises(ValueError, match="closed-shell"):
            mp2.mp2_correlation_energy(
                mol.h1, mol.h2, mol.mo_energies, 1, spin=1
            )

    def test_a_zero_denominator_is_refused(self):
        mol = _gapped_mol(3, 2)
        with pytest.raises(ValueError, match="zero MP2 denominator"):
            mp2.mp2_amplitudes(mol.h1, mol.h2, np.zeros(3), 1)

    def test_a_full_space_casci_is_the_fci_energy(self):
        """The documented invariant: the same contraction ``to_hamiltonian`` uses."""
        mol = _gapped_mol(3, 2)
        casci = mp2.casci_energy(mol, [0, 1, 2])
        fci = sub.exact_energy(mol.h1, mol.h2, 3, 2)
        assert casci == pytest.approx(fci, abs=1e-10)

    def test_casci_refuses_a_space_above_the_determinant_cap(self):
        mol = _gapped_mol(3, 2)
        with pytest.raises(ValueError, match="max_determinants"):
            mp2.casci_energy(mol, [0, 1], max_determinants=1)
        with pytest.raises(TypeError, match="MolecularData"):
            mp2.casci_energy(object(), [0])

    def test_natural_occupations_are_descending_and_bounded(self):
        mol = _gapped_mol(3, 2)
        data = mp2.mp2_natural_orbitals(mol)
        occupations = data["occupations"]
        assert data["physical"] is True
        assert np.all(np.diff(occupations) <= 1e-12)
        assert np.all(occupations >= -1e-10)
        assert np.all(occupations <= 2.0 + 1e-10)
        assert float(np.sum(occupations)) == pytest.approx(2.0, abs=1e-8)

    def test_the_unrelaxed_density_is_block_diagonal(self):
        """``tr(dD) = 0`` and no occ/virt coupling -- the two cheap checks."""
        mol = _gapped_mol(3, 2)
        data = mp2.mp2_density(mol.h1, mol.h2, mol.mo_energies, 1)
        assert data["gap"] == pytest.approx(0.0, abs=1e-12)
        assert data["tr_density"] == pytest.approx(2.0, abs=1e-10)

    def test_a_nonperturbative_reference_is_reported_not_hidden(self):
        """Near-degenerate denominators make the MP2 occupations unphysical.

        The trace survives regardless (``tr(dD) = 0`` exactly), so a trace check
        alone cannot see this -- the occupations are what betray it.  Reporting
        it matters because an active space ranked on them is a sorted list of
        noise, and silently returning them invites exactly that.
        """
        mol = _degenerate_mol()
        with pytest.warns(RuntimeWarning, match="not perturbative"):
            data = mp2.mp2_natural_orbitals(mol)
        assert data["physical"] is False
        assert float(np.sum(data["occupations"])) == pytest.approx(2.0, abs=1e-8)


class TestSpinSquared:
    def test_a_closed_shell_singlet_is_zero(self):
        det = (1 << 0) | (1 << 1)  # orbital 0 doubly occupied
        assert exc.spin_squared(np.array([1.0]), [det]) == pytest.approx(0.0, abs=1e-12)

    def test_one_unpaired_electron_gives_three_quarters(self):
        det = 1 << 0  # a single alpha electron
        assert exc.spin_squared(np.array([1.0]), [det]) == pytest.approx(0.75, abs=1e-12)

    def test_two_parallel_electrons_give_two(self):
        det = (1 << 0) | (1 << 2)  # alpha in orbital 0 and alpha in orbital 1
        assert exc.spin_squared(np.array([1.0]), [det]) == pytest.approx(2.0, abs=1e-12)

    def test_the_operator_route_agrees_with_the_rdm_route(self):
        mol = _mol(n_orbs=2, n_elec=2)
        result = exc.excited_states(mol, n_roots=4)
        dets = result["determinants"]
        n_qubits = 2 * result["n_active_orbitals"]
        vectors = np.asarray(result["vectors"])
        # ``vectors`` are COLUMNS (one per root), indexed like ``determinants``.
        # Iterating the array yields rows, which would silently feed rdm1/rdm2 a
        # vector of the wrong length.
        assert vectors.shape == (len(dets), len(result["energies"]))
        for k, expected in enumerate(result["spin_squared"]):
            vector = vectors[:, k]
            rho1 = sub.rdm1(vector, dets, n_qubits)
            rho2 = sub.rdm2(vector, dets, n_qubits)
            assert exc.spin_squared_from_rdms(rho1, rho2) == pytest.approx(
                expected, abs=1e-10
            )

    def test_the_term_builder_and_the_matrix_helper_agree(self):
        """``spin_squared_matrix`` is a wrapper; the two must not diverge."""
        dets = sub.all_determinants(2, 2)
        direct = np.asarray(
            sub.hamiltonian_matrix(exc.spin_squared_terms(2), dets)
        )
        helper = exc.spin_squared_matrix(dets)
        assert np.allclose(direct, helper, atol=1e-12)
        # ``S^2`` is Hermitian, so the matrix has to come back symmetric.
        assert np.allclose(helper, helper.T, atol=1e-12)
        with pytest.raises(ValueError, match="n_spatial"):
            exc.spin_squared_terms(0)

    def test_the_matrix_helper_refuses_an_empty_space(self):
        with pytest.raises(ValueError, match="at least one determinant"):
            exc.spin_squared_matrix([])

    def test_spin_and_multiplicity_are_consistent(self):
        assert exc.spin_of(0.0) == (0.0, 1)
        assert exc.spin_of(0.75) == (0.5, 2)
        assert exc.spin_of(2.0) == (1.0, 3)

    def test_the_two_by_two_space_has_one_triplet_and_singlets(self):
        mol = _mol(n_orbs=2, n_elec=2)
        result = exc.excited_states(mol, n_roots=4)
        s2 = np.round(np.asarray(result["spin_squared"]), 10)
        assert sorted(set(s2.tolist())) == [0.0, 2.0]
        assert int(np.sum(s2 == 2.0)) == 1  # the M_S = 0 component appears once
        # ``multiplicity`` comes from S, not from <S^2>: S(S+1) = 2 means S = 1,
        # i.e. a triplet.  Feeding <S^2> straight into 2S+1 would say quintet.
        assert list(result["multiplicity"]) == [exc.spin_of(s)[1] for s in s2]
        assert list(result["multiplicity"]) == [1, 1, 3, 1]

    def test_qse_from_hartree_fock_is_exact_on_a_two_orbital_space(self):
        """SD out of one determinant spans this whole sector, so QSE is FCI."""
        mol = _mol(n_orbs=2, n_elec=2)
        result = exc.excited_states(mol, n_roots=4)
        hf = (1 << 0) | (1 << 1)
        pool = exc.excitation_pool(2, hf)
        qse = exc.qse_excited(
            result["terms"],
            result["determinants"],
            hf,
            pool=pool,
            n_spatial=2,
            n_roots=4,
        )
        # ``leaked`` is one number per pool operator, not a scalar.
        assert np.max(np.abs(qse["leaked"])) == pytest.approx(0.0, abs=1e-12)
        assert qse["subspace_dimension"] == len(result["determinants"])
        assert np.allclose(
            np.sort(np.asarray(qse["energies"])),
            np.sort(np.asarray(result["energies"])),
            atol=1e-10,
        )

    def test_the_exact_alias_matches(self):
        mol = _mol(n_orbs=2, n_elec=2)
        assert np.allclose(
            exc.exact_excited_states(mol, n_roots=3)["energies"],
            exc.excited_states(mol, n_roots=3)["energies"],
            atol=1e-12,
        )


class TestExcitationRank:
    """``excitation_pool(rank=...)`` above 2, and what a richer pool buys.

    The parity harness measured this on water, where a singles+doubles pool is
    141 of 441 determinants and the excited roots cannot be exact.  These tests
    pin the same structure on a synthetic model so they run without PySCF.
    """

    n_orbs = 5
    n_elec = 6
    reference = (1 << n_elec) - 1  # three lowest spatial orbitals, doubly filled

    def _pool(self, rank):
        return exc.excitation_pool(self.n_orbs, self.reference, rank=rank)

    def test_each_rank_extends_the_previous_one(self):
        """Higher ranks are appended, so a lower-rank pool is a prefix."""
        two, three, four = self._pool(2), self._pool(3), self._pool(4)
        assert three[: len(two)] == two
        assert four[: len(three)] == three
        assert len(two) < len(three) < len(four)

    def test_a_triple_moves_three_electrons_without_flipping_spin(self):
        two = self._pool(2)
        added = self._pool(3)[len(two):]
        assert added, "rank 3 must add operators on a five-orbital model"
        for term in added:
            creations = [site for site, create in term if create]
            annihilations = [site for site, create in term if not create]
            assert len(creations) == len(annihilations) == 3
            # alpha sites are even, so a spin-preserving move creates and
            # annihilates the same number of alpha electrons
            assert sum(1 for s in creations if s % 2 == 0) == sum(
                1 for s in annihilations if s % 2 == 0
            )

    def test_the_rank_is_a_ceiling_not_a_request(self):
        """Three occupied and two virtual spatial orbitals cap the rank at 4."""
        assert self._pool(9) == self._pool(4)

    def test_a_richer_pool_moves_the_roots_closer(self):
        mol = _mol(n_orbs=self.n_orbs, n_elec=self.n_elec)
        result = exc.excited_states(mol, n_roots=3)
        target = np.asarray(result["energies"], dtype=float)
        ground = np.asarray(result["vectors"])[:, 0]
        worst = []
        for rank in (1, 2, 3, 4):
            qse = exc.qse_excited(
                result["terms"],
                result["determinants"],
                ground,
                pool=self._pool(rank),
                n_spatial=self.n_orbs,
                n_roots=3,
            )
            got = np.asarray(qse["energies"], dtype=float)
            worst.append(float(np.max(np.abs(got - target[: got.size]))))
        # Measured on this model: 3.06e-01, 1.93e-02, 2.81e-04, 6.22e-14.
        assert worst[0] > worst[1] > worst[2] > worst[3]
        assert worst[3] < 1e-10


class TestSeniority:
    def test_closed_shell_has_zero_seniority(self):
        assert sen.seniority((1 << 0) | (1 << 1)) == 0
        assert sen.seniority(0) == 0

    def test_a_singly_occupied_orbital_counts_once(self):
        assert sen.seniority(1 << 0) == 1
        assert sen.seniority(1 << 1) == 1
        assert sen.seniority((1 << 0) | (1 << 2)) == 2

    def test_a_negative_determinant_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            sen.seniority(-1)

    def test_width_validation(self):
        with pytest.raises(ValueError, match="at or above"):
            sen.seniority_of(1 << 4, n_orbitals=2)
        assert sen.seniority_of(1 << 3, n_orbitals=2) == 1

    def test_distribution_counts_every_determinant_once(self):
        dets = sub.all_determinants(2, 2)
        counts = sen.seniority_distribution(dets)
        assert sum(counts.values()) == len(dets)
        assert list(counts) == sorted(counts)
        assert counts[0] == 2  # |a2>, |b2>

    def test_the_state_report_weights_the_seniority_sectors(self):
        """``all_determinants(2, 2)`` is ordered, so the sectors are known."""
        dets = sub.all_determinants(2, 2)  # [3, 5, 6, 9, 10, 12]
        assert [sen.seniority(d) for d in dets] == [0, 2, 2, 2, 2, 0]
        report = sen.seniority_of_state(
            [np.sqrt(0.5), 0.0, 0.0, 0.0, 0.0, np.sqrt(0.5)], dets
        )
        assert list(report["weights"]) == [0]
        assert report["weights"][0] == pytest.approx(1.0, abs=1e-12)
        assert report["mean_seniority"] == pytest.approx(0.0, abs=1e-12)

    def test_the_mean_seniority_is_the_weighted_average(self):
        dets = sub.all_determinants(2, 2)
        # Half the weight on Omega = 0 (det 3), half on Omega = 2 (det 5).
        report = sen.seniority_of_state(
            [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0, 0.0, 0.0], dets
        )
        assert report["weights"] == pytest.approx({0: 0.5, 2: 0.5})
        assert report["mean_seniority"] == pytest.approx(1.0, abs=1e-12)

    def test_the_weights_are_normalised_by_the_retained_weight(self):
        """A dropped amplitude removes weight from ``norm``, not from the sum."""
        dets = sub.all_determinants(2, 2)
        report = sen.seniority_of_state([1.0, 0.0, 0.0, 0.0, 0.0, 1e-12], dets)
        assert report["norm"] == pytest.approx(1.0, abs=1e-20 + 1e-12)
        assert float(sum(report["weights"].values())) == pytest.approx(1.0, abs=1e-12)

    def test_the_state_report_rejects_bad_input(self):
        with pytest.raises(ValueError, match="amplitudes for"):
            sen.seniority_of_state([1.0], [0, 1])
        with pytest.raises(ValueError, match="below the threshold"):
            sen.seniority_of_state([0.0, 0.0], [0, 1])

    def test_the_pool_agrees_with_filtering_the_full_set(self):
        for cap in (0, 2, 4):
            pool = sen.seniority_pool(3, 4, max_seniority=cap)
            direct = sen.filter_by_seniority(sub.all_determinants(3, 4), cap)
            assert pool == direct

    def test_the_pool_respects_the_spin_sector(self):
        pool = sen.seniority_pool(3, 4, max_seniority=None, n_alpha=1, n_beta=1)
        assert len(pool) == 9  # C(3,1) * C(3,1)
        assert all(bin(d).count("1") == 2 for d in pool)

    def test_a_negative_cap_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            sen.filter_by_seniority([1], -1)


class TestAPC:
    def test_it_refuses_without_a_mean_field(self):
        mol = _mol(n_orbs=3, n_elec=2)
        with pytest.raises(ValueError, match="mean field|fock_mo"):
            apc.apc_active_space(mol)

    def test_entropies_are_non_negative_and_shaped_like_the_orbitals(self):
        rng = np.random.default_rng(3)
        n = 4
        fock = np.diag(np.linspace(-1.0, 1.0, n))
        a = rng.normal(size=(n, n))
        exchange = (a + a.T) / 8.0
        occupations = np.array([2.0, 2.0, 0.0, 0.0])
        entropies = apc.apc_entropies(fock, exchange, occupations, rounds=2)
        assert np.shape(entropies) == (n,)
        assert np.all(np.asarray(entropies) >= -1e-12)

    @pytest.mark.parametrize(
        "nactel,norbs,expected",
        [(2, 2, 3), (4, 4, 20), (6, 6, 175), (0, 3, 1)],
    )
    def test_csf_counts_match_the_closed_forms(self, nactel, norbs, expected):
        assert apc.n_csfs(nactel, norbs) == expected

    def test_the_chooser_returns_the_dataclass_the_rest_of_the_api_takes(self):
        occupations = np.array([2.0, 1.0, 1.0, 0.0, 0.0])
        entropies = np.array([0.1, 0.9, 0.8, 0.5, 0.2])
        choice = apc.chooser(occupations, entropies, max_size=(3, 3))
        assert isinstance(choice, apc.ActiveSpaceChoice)
        # The open shells carry the two electrons that survive here.
        assert choice.active_indices == (1, 2, 3)
        assert choice.n_active_electrons == 2
        assert choice.n_active_orbitals == 3
        assert choice.active_orbitals == (2, 3)
        assert choice.n_csfs == apc.n_csfs(2, 3) == 6
        # The properties must agree with the stored tuples, not restate them.
        assert choice.n_active_orbitals == len(choice.active_indices)
        assert choice.n_active_electrons == choice.n_alpha + choice.n_beta
        assert "ActiveSpaceChoice(active=" in repr(choice)

    def test_a_fixed_window_keeps_every_open_shell(self):
        """``fixed=True`` takes exactly ``norbs``, filled by the ranking."""
        occupations = np.array([2.0, 1.0, 1.0, 0.0, 0.0])
        entropies = np.array([0.1, 0.5, 0.4, 0.9, 0.0])
        choice = apc.chooser(occupations, entropies, max_size=(4, 4), fixed=True)
        # (4 electrons, 4 orbitals) is one core pair, both open shells, one virtual:
        # the virtual kept is the higher-scored one, orbital 3.
        assert choice.active_indices == (0, 1, 2, 3)
        assert choice.inactive_indices == ()
        assert choice.secondary_indices == (4,)

    def test_a_fixed_window_with_an_orbital_count_is_refused(self):
        occupations = np.array([2.0, 0.0])
        with pytest.raises(ValueError, match="fixed=True"):
            apc.chooser(occupations, [1.0, 0.0], max_size=1, fixed=True)

    def test_the_pyscf_variant_is_the_two_outcome_entropy(self):
        """``-p log p - q log q`` over the *collapsed* weight, not per partner.

        This is the one place where the default deliberately disagrees with
        ``pyscf.mcscf.apc``: that implementation sums the pair amplitudes into a
        single outcome before the logarithm.  Pinning the reference's arithmetic
        here keeps ``variant="pyscf"`` honest for anyone porting a selection.
        """
        fock = np.diag([-1.0, 0.5, 1.5])
        exchange = np.diag([0.0, 0.4, 0.2])
        amplitudes = []
        for v in (1, 2):
            k12 = 0.5 * exchange[v, v]
            delta = fock[v, v] - fock[0, 0]
            amplitudes.append(-k12 / (delta + np.hypot(k12, delta)))
        norm = np.sqrt(sum(a * a for a in amplitudes) + 1.0)
        pair = sum(a * a for a in amplitudes) / norm ** 2
        vacuum = 1.0 / norm ** 2
        expected = -(pair * np.log(pair) + vacuum * np.log(vacuum))

        occupation = np.array([2, 0, 0])
        aggregate = apc._apc(fock, exchange, occupation, 1e-3, "pyscf")
        paper = apc._apc(fock, exchange, occupation, 1e-3, "paper")
        assert aggregate[0] == pytest.approx(expected, abs=1e-14)
        # Merging outcomes cannot add entropy, so the paper form is larger.
        assert paper[0] > aggregate[0] + 1e-6

    def test_the_two_variants_agree_when_there_is_one_partner(self):
        fock = np.diag([-1.0, 0.5])
        exchange = np.diag([0.0, 0.4])
        occupation = np.array([2, 0])
        assert apc._apc(fock, exchange, occupation, 1e-3, "pyscf")[0] == (
            pytest.approx(apc._apc(fock, exchange, occupation, 1e-3, "paper")[0],
                          abs=1e-15)
        )

    def test_an_unknown_variant_is_refused(self):
        fock = np.diag([-1.0, 0.5])
        with pytest.raises(ValueError, match="variant"):
            apc.apc_entropies(fock, fock, [2.0, 0.0], variant="textbook")

    def test_the_choice_is_a_partition_with_consistent_electron_counts(self):
        """What APC actually promises: a partition, and counts that add up.

        It does *not* promise to exclude doubly occupied orbitals -- a frozen
        core is exactly that, and is variational -- so asserting that would be
        asserting a property the algorithm never had.
        """
        mol = _mol(n_orbs=4, n_elec=4)
        rng = np.random.default_rng(5)
        a = rng.normal(size=(4, 4))
        occupations = np.array([2.0, 2.0, 0.0, 0.0])
        mol.set_mean_field(
            fock=np.diag(np.linspace(-1.5, 0.5, 4)),
            exchange=(a + a.T) / 8.0,
            occupations=occupations,
        )
        choice = apc.apc_active_space(mol, max_size=(2, 2))
        active = set(choice.active_indices)
        inactive = set(choice.inactive_indices)
        secondary = set(choice.secondary_indices)

        assert active | inactive | secondary == set(range(4))
        assert not (active & inactive or active & secondary or inactive & secondary)
        # Core orbitals are occupied (or singly occupied), secondary are empty.
        assert all(occupations[i] > 0 for i in inactive)
        assert all(occupations[i] == 0 for i in secondary)
        # The reported CSF count is the closed form for the chosen window.
        nactel = sum(occupations[i] for i in active)
        assert choice.n_csfs == apc.n_csfs(int(nactel), len(active))
        assert choice.n_alpha + choice.n_beta == int(nactel)
        assert choice.n_csfs <= apc.n_csfs(2, 2)


class TestDeterminantHelpers:
    """The two bit helpers, pinned because every other module indexes with them."""

    def test_popcount_counts_set_bits(self):
        assert sub.popcount(0) == 0
        assert sub.popcount(0b1011) == 3

    def test_the_occupation_list_is_ordered_like_a_pauli_string(self):
        # Character 0 of a Pauli string is qubit 0, so entry 0 is bit 0.
        assert sub.det_occupation(0b101, 4) == [1, 0, 1, 0]
        assert sub.det_occupation(0, 3) == [0, 0, 0]

    def test_the_two_helpers_agree(self):
        for det in range(16):
            assert sub.popcount(det) == sum(sub.det_occupation(det, 4))


class TestSuperfastCost:
    """The superfast encoding is a qubit-count loss everywhere; say so."""

    def test_a_chain_is_the_sparsest_coupled_graph(self):
        assert bksf.chain_edges(4) == [(0, 1), (1, 2), (2, 3)]
        assert bksf.chain_edges(1) == []
        with pytest.raises(ValueError, match="n_modes"):
            bksf.chain_edges(0)

    def test_the_register_has_one_qubit_per_direction(self):
        edges = bksf.chain_edges(5)
        register = bksf.directed_edge_register(5, edges)
        assert len(register) == 2 * len(edges)
        assert sorted(register.values()) == list(range(2 * len(edges)))
        # Deterministic across calls: the qubit order is part of the interface.
        assert register == bksf.directed_edge_register(5, list(reversed(edges)))

    def test_the_register_rejects_a_bad_edge(self):
        with pytest.raises(ValueError, match="self-loop"):
            bksf.directed_edge_register(3, [(1, 1)])
        with pytest.raises(ValueError, match="outside"):
            bksf.directed_edge_register(3, [(0, 3)])

    def test_the_vertex_operator_is_a_pure_z_parity(self):
        """``B_i = 1 - 2 n_i``: Z only, one factor per incident edge."""
        edges = bksf.chain_edges(4)
        for i in range(4):
            operator = bksf.vertex_operator(4, edges, i)
            assert set(operator.pauli_str) <= {"I", "Z"}
            expected = 1 if i in (0, 3) else 2
            assert sum(1 for ch in operator.pauli_str if ch != "I") == expected
        # An isolated mode is unrepresentable, not an identity: returning B = I
        # would make (1 - B)/2 claim the mode is always empty.
        with pytest.raises(ValueError, match="no incident edge"):
            bksf.vertex_operator(2, [], 0)
        report = bksf.cost(2, [])
        assert report["n_isolated"] == 2
        assert report["n_edges"] == 0

    def test_the_edge_operator_is_one_x_plus_a_bounded_z_string(self):
        n, edges = 6, bksf.chain_edges(6)
        for p, q in edges:
            for a, b in ((p, q), (q, p)):
                operator = bksf.edge_operator(n, edges, a, b)
                label = operator.pauli_str
                assert label.count("X") == 1
                assert set(label) <= {"I", "X", "Z"}
                # Weight <= 1 + deg(p) + deg(q); on a chain the degree is <= 2.
                assert sum(1 for ch in label if ch != "I") <= 5
                # eps_pq: the reverse direction carries the opposite sign.
                reverse = bksf.edge_operator(n, edges, b, a)
                assert operator.coeffs == -reverse.coeffs

    def test_qubit_count_is_two_per_undirected_edge(self):
        """``2 |E|``, with an edge counted once whichever way it is written."""
        edges = bksf.chain_edges(4)
        assert bksf.qubit_count(4, edges) == 6
        assert bksf.qubit_count(4, [(0, 1), (1, 0), (2, 1)]) == 4
        # The standalone count and the report must not diverge.
        assert bksf.qubit_count(4, edges) == bksf.cost(4, edges)["ose_qubits"]

    def test_an_uncoupled_pair_has_no_edge_operator(self):
        with pytest.raises(ValueError, match="not an edge"):
            bksf.edge_operator(4, bksf.chain_edges(4), 0, 2)

    def test_the_qubit_count_loses_to_jordan_wigner_even_on_a_chain(self):
        """``2 (n - 1) > n`` for every ``n >= 3`` -- the decisive number."""
        for n in (3, 4, 8, 20):
            report = bksf.cost(n, bksf.chain_edges(n))
            assert report["ose_qubits"] == 2 * (n - 1)
            assert report["ose_qubits"] > report["jw_qubits"]
            assert report["ose_cheaper_on_qubits"] is False
        # It only ties at the smallest non-trivial size.
        assert bksf.cost(2, bksf.chain_edges(2))["ose_qubits"] == 2

    def test_a_dense_graph_is_quadratic_in_the_mode_count(self):
        """Which is the molecular case: a near-complete interaction graph."""
        mol = _mol(n_orbs=4, n_elec=4)
        edges = bksf.interaction_graph(mol.h1, mol.h2, threshold=1e-8)
        assert len(edges) == 6  # every pair of 4 orbitals is coupled
        report = bksf.cost(mol.n_orbitals, edges)
        assert report["ose_qubits"] == 12  # 2 * 6, against 4 for Jordan-Wigner
        assert report["qubit_ratio"] == pytest.approx(3.0)

    def test_the_threshold_is_what_decides_the_graph(self):
        mol = _mol(n_orbs=3, n_elec=2)
        loose = bksf.interaction_graph(mol.h1, mol.h2, threshold=0.0)
        tight = bksf.interaction_graph(mol.h1, mol.h2, threshold=1e6)
        assert len(loose) == 3
        assert tight == []
        with pytest.raises(ValueError, match="square"):
            bksf.interaction_graph(np.zeros((2, 3)))


class TestPointGroupSectors:
    """The irrep of a determinant, and the sector it selects.

    The rule is a direct product over singly occupied orbitals, which for a
    one-dimensional irrep is the XOR of ``pyscf``'s ``irrep_id`` values.  These
    tests pin the arithmetic; the validation that the rule really does block
    diagonalise the Hamiltonian lives in the PySCF test at the end, where a
    real point group is available.
    """

    @staticmethod
    def _c2v():
        mol = _mol(n_orbs=3, n_elec=2)
        mol.point_group = "C2v"
        # A2 carries no basis function in water/STO-3G, so its id is skipped:
        # the ids are the group's, not positions in this tuple.
        mol.irrep_labels = ("A1", "B1", "B2")
        mol.irrep_ids = (0, 2, 3)
        mol.mo_irreps = ("A1", "B1", "B2")
        return mol

    def test_a_singly_occupied_orbital_carries_its_irrep(self):
        assert sym.determinant_irrep([0, 2, 3], 1 << 2) == 2

    def test_a_doubly_occupied_orbital_is_the_identity(self):
        # irrep * irrep = 1 in a one-dimensional irrep, so the pair cancels.
        assert sym.determinant_irrep([0, 2, 3], (1 << 2) | (1 << 3)) == 0

    def test_two_singly_occupied_orbitals_multiply(self):
        det = (1 << 2) | (1 << 4)
        assert sym.determinant_irrep([0, 2, 3], det) == 2 ^ 3

    def test_a_bit_beyond_the_orbital_count_is_refused(self):
        with pytest.raises(ValueError, match="above 4"):
            sym.determinant_irrep([0, 1], 1 << 4)

    def test_a_negative_determinant_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            sym.determinant_irrep([0, 1], -1)

    def test_every_abelian_group_is_accepted(self):
        for name in sorted(sym.ABELIAN_POINT_GROUPS):
            assert sym.require_abelian(name) == name

    def test_a_degenerate_irrep_group_is_refused(self):
        # In C3v irrep_id indexes a representation list, so XOR is meaningless.
        with pytest.raises(ValueError, match="degenerate"):
            sym.require_abelian("C3v")

    def test_a_missing_point_group_is_refused(self):
        with pytest.raises(ValueError, match="no point group"):
            sym.require_abelian(None)

    def test_the_occupation_list_agrees_with_the_bitmask(self):
        irreps = [0, 2, 3]
        occupation = [1, 1, 0, 0, 0, 0]        # orbital 0 doubly occupied
        assert sym.sector_from_occupation(irreps, occupation) == 0
        occupation = [0, 0, 1, 0, 1, 0]        # orbitals 1 and 2, alpha each
        assert sym.sector_from_occupation(irreps, occupation) == 2 ^ 3

    def test_a_mismatched_occupation_length_is_refused(self):
        with pytest.raises(ValueError, match="3 entries"):
            sym.sector_from_occupation([0, 1, 2], [1, 0, 0])

    def test_the_sector_filter_keeps_order_and_drops_the_rest(self):
        irreps = [0, 2, 3]
        a1 = (1 << 2) | (1 << 3)               # doubly occupied B1: identity
        b1 = 1 << 2                            # singly occupied B1
        b2 = 1 << 4                            # singly occupied B2
        assert sym.sector_determinants(irreps, [b1, a1, b2], sector=0) == [a1]
        assert sym.sector_determinants(irreps, [b1, a1, b2], sector=2) == [b1]

    def test_an_unreachable_sector_returns_nothing(self):
        # An empty basis is a caller problem, not something to paper over.
        assert sym.sector_determinants([0, 0], [1 << 1], sector=7) == []

    def test_orbit_irreps_maps_labels_to_the_recorded_ids(self):
        mol = self._c2v()
        assert sym.orbit_irreps(mol) == [0, 2, 3]
        assert sym.orbit_irreps(mol, [2, 1]) == [3, 2]

    def test_orbit_irreps_needs_symmetry_data(self):
        with pytest.raises(ValueError, match="no irrep data"):
            sym.orbit_irreps(_mol(n_orbs=3, n_elec=2))

    def test_orbit_irreps_rejects_an_out_of_range_orbital(self):
        with pytest.raises(ValueError, match="outside"):
            sym.orbit_irreps(self._c2v(), [5])

    def test_the_reference_sector_reproduces_the_full_fci(self):
        """The real check: the reduction must not move the energy."""
        pytest.importorskip("pyscf")
        from superfermion.chemistry.pyscf_bridge import (
            _spin_orbital_integrals,
            molecule_from_geometry,
        )

        mol = molecule_from_geometry(
            "O 0 0 0; H 0 0 0.9697; H 0.9262 0 -0.2426",
            basis="sto-3g",
            symmetry=True,
        )
        assert mol.point_group in sym.ABELIAN_POINT_GROUPS
        irreps = sym.orbit_irreps(mol)
        h1_so, h2_so = _spin_orbital_integrals(mol.h1, mol.h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        n_elec, n_orb = mol.n_electrons, mol.n_orbitals
        dets = sub.all_determinants(n_orb, n_elec)
        occupation = mol.reference_occupation()
        sector = sym.sector_from_occupation(irreps, occupation)
        kept = sym.sector_determinants(irreps, dets, sector=sector)

        assert 0 < len(kept) < len(dets)
        full, _ = sub.ground_state(terms, dets, k=1)
        reduced, _ = sub.ground_state(terms, kept, k=1)
        assert float(full[0]) == pytest.approx(float(reduced[0]), abs=1e-10)

        # And the block structure that makes the reduction legitimate.
        matrix = np.asarray(sub.hamiltonian_matrix(terms, dets))
        labels = np.array([sym.determinant_irrep(irreps, d) for d in dets])
        off = labels[:, None] != labels[None, :]
        assert float(np.max(np.abs(matrix[off]))) < 1e-10


needs_qiskit = pytest.mark.skipif(
    not interop.QISKIT_AVAILABLE, reason="qiskit is not installed"
)


class TestQiskitInterop:
    @needs_qiskit
    def test_the_matrices_are_identical(self):
        mol = _mol(n_orbs=2, n_elec=2)
        ham = mol.to_hamiltonian()
        mine = sub.pauli_matrix(ham)
        theirs = np.asarray(interop.operator_matrix(ham))
        assert np.max(np.abs(mine - theirs)) == pytest.approx(0.0, abs=1e-12)

    @needs_qiskit
    def test_labels_round_trip_exactly(self):
        mol = _mol(n_orbs=2, n_elec=2)
        ham = mol.to_hamiltonian()
        op = interop.to_qiskit_op(ham)
        back = interop.from_qiskit_op(op)
        before = {t.pauli_str: complex(t.coeffs) for t in ham.terms}
        after = {t.pauli_str: complex(t.coeffs) for t in back.terms}
        assert set(before) == set(after)
        for label, coefficient in before.items():
            assert after[label] == pytest.approx(coefficient, abs=1e-12)

    @needs_qiskit
    def test_a_short_string_is_padded_on_the_right(self):
        """Qubit 0 is the leftmost character, so padding adds high qubits."""
        ham = Hamiltonian([PauliString("ZI", coeffs=1.0)])
        labels, coeffs = interop.to_qiskit_labels(ham, n_qubits=4)
        assert labels == ["ZIII"]
        # The two lists are parallel by position, so this pins the pairing that
        # lets the result go straight into SparsePauliOp(labels, coeffs).
        assert [complex(c) for c in coeffs] == [1.0 + 0j]

    @needs_qiskit
    def test_the_width_defaults_to_the_widest_label(self):
        ham = Hamiltonian(
            [PauliString("Z", coeffs=1.0), PauliString("XYZ", coeffs=2.0)]
        )
        labels, coeffs = interop.to_qiskit_labels(ham)
        assert labels == ["ZII", "XYZ"]
        assert [complex(c) for c in coeffs] == [1.0 + 0j, 2.0 + 0j]

    @needs_qiskit
    def test_a_label_wider_than_n_qubits_is_refused(self):
        """Truncating would silently change which qubits the term acts on."""
        ham = Hamiltonian([PauliString("XYZ", coeffs=1.0)])
        with pytest.raises(ValueError, match="wider than n_qubits"):
            interop.to_qiskit_labels(ham, n_qubits=2)
