"""Contracts for the chemistry methods layered on the integral container.

Every class here pins a defect that shipped silently:

* ``ground_state`` asked ARPACK for one eigenvalue of a one-determinant matrix,
  which raises ``TypeError`` before returning anything;
* ``qsci`` started from a single determinant, so the first diagonalisation was
  1x1 and carried no correlation;
* ``one_shot_dmet`` evaluated the fragment reference with the bare one-body
  operator, dropping ``tr(rho_imp F_core)`` and turning water's 30 mHa
  correlation correction into a +59 Ha error;
* ``_fold_core`` sliced the actives out without the inactive potential, which
  reports water at -23.69 Ha instead of -74.98 Ha; measured, the other two
  omissions give -95.11 Ha (potential kept out of the window one-body but
  core energy added) and -3.55 Ha (potential present, core energy dropped);
* ``optimise_orbital_basis`` returned a rotation chosen by a tie even when the
  surrogate gained nothing, and its objective was documented as even in the
  generator; measured, it is not even and the origin is not stationary.
"""

import numpy as np
import pytest

from superfermion.chemistry import aegiss
from superfermion.chemistry import avas
from superfermion.chemistry import dmet
from superfermion.chemistry import entropy as ent
from superfermion.chemistry import fcidump
from superfermion.chemistry import orbital_reduction as ored
from superfermion.chemistry import sqd
from superfermion.chemistry import subspace as sub
from superfermion.chemistry.pyscf_bridge import MolecularData, _spin_orbital_integrals

pytestmark = pytest.mark.domain


# The eight index permutations a chemist-ordered two-electron integral obeys.
_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _synthetic(n_orbs: int = 3, seed: int = 0):
    """Synthetic integrals with all eight symmetries and real correlation."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_orbs, n_orbs))
    h1 = a + a.T
    raw = rng.normal(size=(n_orbs,) * 4) / 4.0
    h2 = sum(raw.transpose(p) for p in _PERMUTATIONS) / 8.0
    return h1, h2


def _mol(h1, h2, n_elec, nuclear_repulsion=0.0, name="model"):
    return MolecularData(
        name=name, geometry="H 0 0 0", h1=h1, h2=h2,
        nuclear_repulsion=nuclear_repulsion,
        n_orbitals=np.asarray(h1).shape[0], n_electrons=n_elec,
    )


class TestSubspace:
    def test_a_one_determinant_space_is_its_own_energy(self):
        """The sparse path must not ask ARPACK for k eigenvalues of a 1x1."""
        h1, h2 = _synthetic(2, seed=1)
        h1_so, h2_so = _spin_orbital_integrals(h1, h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        dets = [0b0011]

        energies, vectors = sub.ground_state(terms, dets, sparse=True)
        diagonal = np.real(np.asarray(sub.hamiltonian_matrix(terms, dets))[0, 0])
        assert energies[0] == pytest.approx(diagonal, abs=1e-12)
        assert vectors.shape == (1, 1)

    def test_sparse_and_dense_agree_on_a_small_space(self):
        h1, h2 = _synthetic(2, seed=2)
        h1_so, h2_so = _spin_orbital_integrals(h1, h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        dets = sub.all_determinants(2, 2, n_alpha=1, n_beta=1)[:3]

        sparse, _ = sub.ground_state(terms, dets, sparse=True)
        dense, _ = sub.ground_state(terms, dets, sparse=False)
        assert np.allclose(sparse, dense, atol=1e-12)

    def test_the_space_is_closed_under_requests_larger_than_it(self):
        h1, h2 = _synthetic(2, seed=3)
        h1_so, h2_so = _spin_orbital_integrals(h1, h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        dets = [0b0011, 0b1100]
        energies, vectors = sub.ground_state(terms, dets, sparse=True, k=5)
        assert energies.shape == (2,)
        assert vectors.shape == (2, 2)

    def test_an_empty_space_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            sub.ground_state({}, [], sparse=False)

    def test_rdm1_trace_is_the_electron_count(self):
        h1, h2 = _synthetic(2, seed=4)
        h1_so, h2_so = _spin_orbital_integrals(h1, h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        dets = sub.all_determinants(2, 2, n_alpha=1, n_beta=1)
        _, vectors = sub.ground_state(terms, dets, sparse=False)

        rho = sub.rdm1(vectors[:, 0], dets, 2 * h1.shape[0])
        assert np.trace(np.real(rho)) == pytest.approx(2.0, abs=1e-10)

    def test_rdm2_pair_occupation_sums_to_n_times_n_minus_one(self):
        h1, h2 = _synthetic(2, seed=5)
        h1_so, h2_so = _spin_orbital_integrals(h1, h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        dets = sub.all_determinants(2, 2, n_alpha=1, n_beta=1)
        _, vectors = sub.ground_state(terms, dets, sparse=False)

        n_so = 2 * h1.shape[0]
        gamma = np.real(sub.rdm2(vectors[:, 0], dets, n_so))
        total = sum(gamma[p, q, p, q] for p in range(n_so) for q in range(n_so) if p != q)
        assert total == pytest.approx(2.0, abs=1e-10)


class TestSQD:
    @staticmethod
    def _model(seed=6):
        h1, h2 = _synthetic(3, seed=seed)
        return h1, h2

    def test_qsci_with_the_full_pool_equals_exact_diagonalisation(self):
        h1, h2 = self._model()
        pool = sub.all_determinants(3, 2, n_alpha=1, n_beta=1)
        exact = sub.exact_energy(h1, h2, 3, 2, n_alpha=1, n_beta=1)

        result = sqd.qsci(h1, h2, 3, 2, n_alpha=1, n_beta=1, n_select=len(pool))
        assert result["exact"] is True
        assert result["energy"] == pytest.approx(exact, abs=1e-10)
        assert result["pool_size"] == len(pool)

    def test_qsci_can_start_from_a_single_determinant(self):
        """The 1x1 start is the shape that used to raise inside ARPACK."""
        h1, h2 = self._model(seed=7)
        result = sqd.qsci(h1, h2, 3, 2, n_alpha=1, n_beta=1, n_select=1)
        assert len(result["determinants"]) == 1
        assert result["iterations"] >= 1
        assert np.isfinite(result["energy"])

    def test_more_determinants_never_raise_the_energy(self):
        h1, h2 = self._model(seed=8)
        few = sqd.qsci(h1, h2, 3, 2, n_alpha=1, n_beta=1, n_select=2)
        many = sqd.qsci(h1, h2, 3, 2, n_alpha=1, n_beta=1, n_select=4)
        assert many["energy"] <= few["energy"] + 1e-12

    def test_the_energy_history_is_monotone(self):
        h1, h2 = self._model(seed=9)
        result = sqd.qsci(h1, h2, 3, 2, n_alpha=1, n_beta=1, n_select=4)
        history = np.asarray(result["history"])
        assert np.all(np.diff(history) <= 1e-10)

    def test_sampling_keeps_only_the_requested_sector(self):
        amplitudes = np.array([0.5, 0.5])
        determinants = [0b0011, 0b0101]        # (1a,1b) and (2a,0b)
        sampled = sqd.sample_determinants(
            amplitudes, determinants, 200, seed=0,
            n_alpha=1, n_beta=1, n_orbitals=2,
        )
        assert sum(sampled["counts"].values()) + sampled["n_discarded"] == 200
        assert 0b0101 not in sampled["counts"]
        assert sampled["determinants"] == [0b0011]

    def test_sampling_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="amplitudes"):
            sqd.sample_determinants(np.ones(3), [1, 2], 5)

    def test_a_zero_vector_cannot_be_sampled(self):
        with pytest.raises(ValueError, match="zero"):
            sqd.sample_determinants(np.zeros(2), [1, 2], 5)

    def test_one_determinant_energy_is_its_fock_space_diagonal(self):
        h1, h2 = self._model(seed=10)
        h1_so, h2_so = _spin_orbital_integrals(h1, h2)
        terms = sub.fermionic_terms(h1_so, h2_so)
        det = sub.all_determinants(3, 2, n_alpha=1, n_beta=1)[0]

        result = sqd.sqd_energy(h1, h2, [det])
        diagonal = np.real(np.asarray(sub.hamiltonian_matrix(terms, [det]))[0, 0])
        assert result["energy"] == pytest.approx(diagonal, abs=1e-12)


class TestDMET:
    @staticmethod
    def _model(seed=11):
        h1, h2 = _synthetic(4, seed=seed)
        return h1, h2

    def test_a_fragment_covering_everything_is_exact(self):
        h1, h2 = self._model()
        exact = sub.exact_energy(h1, h2, 4, 4)
        result = dmet.one_shot_dmet(h1, h2, 4, [list(range(4))])
        assert result["fragments"][0]["n_bath"] == 0
        assert result["energy_electronic"] == pytest.approx(exact, abs=1e-9)

    def test_canonical_fragments_need_no_bath(self):
        h1, h2 = self._model()
        result = dmet.one_shot_dmet(h1, h2, 4, [[0, 1], [2, 3]])
        assert all(f["n_bath"] == 0 for f in result["fragments"])

    def test_a_localised_partition_gets_a_bath(self):
        h1, h2 = self._model()
        rng = np.random.default_rng(12)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        result = dmet.one_shot_dmet(h1, h2, 4, [[0, 1], [2, 3]], orbitals=q)
        assert any(f["n_bath"] > 0 for f in result["fragments"])

    def test_the_reference_is_the_mean_field_energy(self):
        """The reference carries the core field; dropping it costs tens of Ha."""
        h1, h2 = self._model(seed=13)
        rng = np.random.default_rng(14)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        result = dmet.one_shot_dmet(h1, h2, 4, [[0, 1], [2, 3]], orbitals=q)

        for fragment in result["fragments"]:
            assert fragment["energy_reference"] == pytest.approx(
                result["hf_energy"], abs=1e-8
            )

    def test_every_correction_is_a_correlation_energy(self):
        h1, h2 = self._model(seed=15)
        rng = np.random.default_rng(16)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        result = dmet.one_shot_dmet(h1, h2, 4, [[0, 1], [2, 3]], orbitals=q)

        # One-shot multi-fragment DMET is not variational -- on this model it
        # lands below FCI -- so only the per-fragment sign is asserted here.
        assert all(f["correlation"] <= 1e-10 for f in result["fragments"])
        assert result["correlation_total"] <= 0.0

    def test_the_impurity_occupation_is_an_integer(self):
        h1, h2 = self._model(seed=17)
        rng = np.random.default_rng(18)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        result = dmet.one_shot_dmet(h1, h2, 4, [[0, 1], [2, 3]], orbitals=q)
        for fragment in result["fragments"]:
            occupation = fragment["impurity_occupation"]
            assert occupation == pytest.approx(round(occupation), abs=1e-9)

    def test_an_odd_electron_count_is_rejected(self):
        h1, h2 = self._model()
        with pytest.raises(ValueError, match="closed-shell"):
            dmet.one_shot_dmet(h1, h2, 3, [[0, 1]])

    def test_a_non_orthogonal_basis_is_rejected(self):
        h1, h2 = self._model()
        with pytest.raises(ValueError, match="orthogonal"):
            dmet.one_shot_dmet(h1, h2, 4, [[0, 1], [2, 3]], orbitals=2.0 * np.eye(4))


class TestOrbitalReduction:
    @staticmethod
    def _generator(n=4, seed=19):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n, n))
        return a - a.T

    def test_the_rotation_is_orthogonal(self):
        rotation = ored.rotation_matrix(self._generator())
        assert np.allclose(rotation.T @ rotation, np.eye(4), atol=1e-12)

    def test_rotating_the_integrals_preserves_the_energy(self):
        h1, h2 = _synthetic(4, seed=20)
        before = sub.exact_energy(h1, h2, 4, 4)
        h1_new, h2_new = ored.rotate_integrals(h1, h2, self._generator())
        after = sub.exact_energy(h1_new, h2_new, 4, 4)
        assert after == pytest.approx(before, abs=1e-9)

    def test_a_general_generator_breaks_the_evenness(self):
        """``U(-k) = U(k)^T`` does not leave the entries' moduli alone.

        The objective used to be documented as even in ``kappa``; it is not.
        The difference is first order in the step, so the origin is not a
        stationary point either.
        """
        h1, h2 = _synthetic(4, seed=21)
        kappa = self._generator(seed=22)
        plus = ored.basis_objective(*ored.rotate_integrals(h1, h2, kappa))
        minus = ored.basis_objective(*ored.rotate_integrals(h1, h2, -kappa))
        assert abs(plus - minus) > 1e-3

        step = 1e-4
        direction = kappa / np.linalg.norm(kappa)
        forward = ored.basis_objective(*ored.rotate_integrals(h1, h2, step * direction))
        backward = ored.basis_objective(*ored.rotate_integrals(h1, h2, -step * direction))
        # O(step) - O(step^2) would be 1e-8 here and the measured value is 1e-3.
        assert abs(forward - backward) / step > 1.0

    def test_a_zero_first_start_can_still_improve_the_objective(self):
        """The origin is not a dead end in general, only for symmetric inputs."""
        h1, h2 = _synthetic(4, seed=21)
        result = ored.optimise_orbital_basis(
            h1, h2, n_starts=1, seed=0, maxiter=200, first_start_scale=0.0
        )
        assert result["improvement"] > 0.5
        assert result["objective_after"] < result["objective_before"]

    def test_the_optimiser_never_increases_the_objective(self):
        h1, h2 = _synthetic(4, seed=23)
        result = ored.optimise_orbital_basis(h1, h2, n_starts=2, seed=0, maxiter=40)
        assert result["objective_after"] <= result["objective_before"] + 1e-12
        assert result["improvement"] >= 0.0

    def test_a_flat_objective_returns_the_scf_basis(self):
        """``h1 = I`` is invariant under any rotation, so nothing can be gained."""
        h1 = np.eye(2)
        result = ored.optimise_orbital_basis(h1, None, n_starts=2, seed=0, maxiter=20)
        assert result["improvement"] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(result["kappa"], 0.0)
        assert np.array_equal(result["h1"], h1)

    def test_pauli_statistics_agree_with_the_rotated_operator(self):
        h1, h2 = _synthetic(3, seed=24)
        stats = ored.pauli_term_stats(h1, h2)
        assert stats["n_qubits"] == 6
        assert stats["n_terms"] > 0
        assert 0 <= stats["max_weight"] <= 6
        assert stats["l1_norm"] > 0.0


class TestEntropy:
    def test_a_single_determinant_has_zero_orbital_entropy(self):
        values = ent.orbital_entropy(np.array([1.0]), [0b0011], 3)
        assert np.allclose(values, 0.0, atol=1e-12)

    def test_a_bell_pair_puts_ln_two_on_the_entangled_orbitals(self):
        vector = np.array([1.0, 1.0]) / np.sqrt(2.0)
        determinants = [0b000011, 0b001100]         # 0a0b and 1a1b
        values = ent.orbital_entropy(vector, determinants, 3)
        assert values[0] == pytest.approx(np.log(2.0), abs=1e-10)
        assert values[1] == pytest.approx(np.log(2.0), abs=1e-10)
        assert values[2] == pytest.approx(0.0, abs=1e-10)

    def test_entropy_stays_inside_the_orbital_dimension(self):
        vector = np.array([0.6, 0.8]) / 1.0
        determinants = [0b000011, 0b001100]
        values = ent.orbital_entropy(vector, determinants, 3)
        assert np.all(values >= -1e-12)
        assert np.all(values <= np.log(4.0) + 1e-12)

    def test_mutual_information_is_symmetric_with_a_zero_diagonal(self):
        vector = np.array([1.0, 1.0]) / np.sqrt(2.0)
        determinants = [0b000011, 0b001100]
        matrix = ent.mutual_information(vector, determinants, 3)
        assert np.allclose(matrix, matrix.T, atol=1e-12)
        assert np.allclose(np.diag(matrix), 0.0, atol=1e-12)
        assert np.all(matrix >= 0.0)

    def test_autocas_selects_the_entangled_pair(self):
        vector = np.array([1.0, 1.0]) / np.sqrt(2.0)
        determinants = [0b000011, 0b001100]
        assert ent.autocas_selection(vector, determinants, 3, threshold=0.1) == [0, 1]


class TestAVAS:
    @staticmethod
    def _orbitals(n_ao=5, seed=25):
        """Square orthogonal MO set, so the MOs span the target AOs.

        ``trace(M) == len(target_aos)`` needs that spanning: with fewer MOs
        than AOs the projector only reaches part of the target space and the
        trace falls short, which is a property of the input, not a defect.
        """
        rng = np.random.default_rng(seed)
        coefficients, _ = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))
        return coefficients, np.eye(n_ao)

    def test_the_projector_trace_is_the_target_dimension(self):
        coefficients, overlap = self._orbitals()
        metric = avas.projector_metric(coefficients, overlap, [0, 2])
        assert metric.shape == (5, 5)
        assert np.trace(metric) == pytest.approx(2.0, abs=1e-12)
        assert np.allclose(metric, metric.T, atol=1e-12)

    def test_the_projector_eigenvalues_are_probabilities(self):
        coefficients, overlap = self._orbitals(seed=26)
        metric = avas.projector_metric(coefficients, overlap, [1, 3])
        eigenvalues = np.linalg.eigvalsh(metric)
        assert np.all(eigenvalues > -1e-12)
        assert np.all(eigenvalues < 1.0 + 1e-12)

    def test_the_rotation_keeps_the_selected_dimension(self):
        coefficients, overlap = self._orbitals(seed=27)
        result = avas.avas_rotation(
            coefficients, overlap, [0, 2], n_occupied=1, threshold=0.0
        )
        assert result["target_dimension"] == pytest.approx(2.0, abs=1e-12)
        assert sum(result["eigenvalues"]) == pytest.approx(2.0, abs=1e-10)
        assert result["n_active"] == len(result["active_indices"])

    def test_an_out_of_range_target_orbital_is_rejected(self):
        coefficients, overlap = self._orbitals()
        with pytest.raises(ValueError, match="out of range"):
            avas.projector_metric(coefficients, overlap, [9])


class TestAEGISSCoreFolding:
    def test_folding_the_core_reproduces_the_full_determinant_energy(self):
        """The invariant the missing-core-field defect broke."""
        h1, h2 = _synthetic(4, seed=28)
        mol = _mol(h1, h2, 4)
        occupations = np.array([2.0, 2.0, 0.0, 0.0])
        active = [2, 3]

        h1_eff, h2_eff, e_core = aegiss._fold_core(mol, active, occupations)
        full = dmet._determinant_energy(h1, h2, [0, 1, 2])
        window = dmet._determinant_energy(h1_eff, h2_eff, [0])
        assert e_core + window == pytest.approx(full, abs=1e-10)

    def test_an_empty_core_leaves_the_integrals_untouched(self):
        h1, h2 = _synthetic(3, seed=29)
        mol = _mol(h1, h2, 4)
        occupations = np.array([2.0, 2.0, 0.0])
        h1_eff, h2_eff, e_core = aegiss._fold_core(mol, [0, 1, 2], occupations)
        assert e_core == 0.0
        assert np.allclose(h1_eff, h1)
        assert np.allclose(h2_eff, h2)

    def test_the_core_potential_is_not_dropped_from_the_window(self):
        h1, h2 = _synthetic(3, seed=30)
        mol = _mol(h1, h2, 4)
        occupations = np.array([2.0, 0.0, 0.0])
        h1_eff = aegiss._fold_core(mol, [1, 2], occupations)[0]
        bare = h1[np.ix_([1, 2], [1, 2])]
        assert not np.allclose(h1_eff, bare)


class TestFCIDump:
    def test_the_round_trip_preserves_the_integrals(self):
        h1, h2 = _synthetic(2, seed=31)
        mol = _mol(h1, h2, 2, nuclear_repulsion=0.7)
        text = fcidump.write_fcidump_text(mol)
        back = fcidump.read_fcidump_text(text)

        assert back.n_orbitals == 2
        assert back.n_electrons == 2
        assert back.nuclear_repulsion == pytest.approx(0.7, abs=1e-10)
        assert np.allclose(back.h1, h1, atol=1e-10)
        assert np.allclose(back.h2, h2, atol=1e-10)

    def test_the_round_trip_preserves_the_energy(self):
        h1, h2 = _synthetic(3, seed=32)
        mol = _mol(h1, h2, 2)
        back = fcidump.read_fcidump_text(fcidump.write_fcidump_text(mol))
        assert sub.exact_energy(back.h1, back.h2, 3, 2) == pytest.approx(
            sub.exact_energy(h1, h2, 3, 2), abs=1e-10
        )

    def test_a_file_without_a_header_is_rejected(self):
        with pytest.raises(ValueError):
            fcidump.read_fcidump_text("1.0 1 1 1 1")
