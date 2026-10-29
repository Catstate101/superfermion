"""Chemistry module domain tests."""

import itertools
import warnings

import numpy as np
import pytest

import superfermion as sf
from superfermion.chemistry.ansatz import uccsd_ansatz
from superfermion.chemistry.hamiltonians import FermionicOperator, get_molecular_hamiltonian
from superfermion.chemistry.pyscf_bridge import (
    MolecularData,
    _spin_orbital_integrals,
)
from superfermion.chemistry.tapering import taper_z2, z2_symmetries
from superfermion.observables.core import Hamiltonian, PauliString


pytestmark = pytest.mark.domain


class TestMolecularHamiltonian:
    def test_h2_hamiltonian_term_count(self):
        H = get_molecular_hamiltonian("H2")
        assert len(H.terms) == 5
        pauli_strings = {term.pauli_str for term in H.terms}
        assert pauli_strings == {"II", "ZI", "IZ", "ZZ", "XX"}


class TestUCCSDAnsatz:
    def test_h2_ansatz_gate_count(self):
        ansatz = uccsd_ansatz(2, 2)
        assert isinstance(ansatz, sf.Circuit)
        assert ansatz.n_qubits == 2
        assert ansatz.gate_count == 3
        assert ansatz.n_parameters == 1


class TestJordanWigner:
    def test_number_operator_maps_to_paulis(self):
        # n_0 = a†_0 a_0  →  0.5·I − 0.5·Z_0  (single qubit, JW convention)
        ferm_op = FermionicOperator({((0, 1), (0, 0)): 1.0})
        H = ferm_op.jordan_wigner(1)

        terms = {term.pauli_str: float(term.coeffs.real) for term in H.terms}
        assert len(terms) == 2
        assert abs(terms["I"] - 0.5) < 1e-10
        assert abs(terms["Z"] + 0.5) < 1e-10

    def test_fermionic_from_coeffs_jordan_wigner(self):
        h1 = np.array([[1.0, 0.0], [0.0, 0.5]])
        ferm_op = FermionicOperator.from_coeffs(h1)
        H = ferm_op.jordan_wigner(2)

        terms = {term.pauli_str: float(term.coeffs.real) for term in H.terms}
        assert len(terms) == 3
        assert abs(terms["II"] - 0.75) < 1e-10
        assert abs(terms["ZI"] + 0.5) < 1e-10
        assert abs(terms["IZ"] + 0.25) < 1e-10


class TestSpinOrbitalIntegrals:
    """The molecular Hamiltonian must be built from spin-orbital integrals.

    The stored ``h1``/``h2`` are spin-free (restricted), but the operator spans
    ``2*nao`` modes.  These tests pin the expansion and the tensor layout that
    ``FermionicOperator.from_coeffs`` actually consumes (``T[i,j,k,l] =
    (i k|j l)``), on a 2-orbital model whose Slater-Condon answer is exact.
    """

    @staticmethod
    def _two_orbital_model():
        # u0=(00|00), u1=(11|11), v=(00|11) Coulomb, t=(01|01) pair scattering
        h0, h1 = -1.25, -0.40
        u0, u1, v, t = 0.70, 0.55, 0.31, 0.17
        h1_mo = np.diag([h0, h1])
        h2_mo = np.zeros((2, 2, 2, 2))
        h2_mo[0, 0, 0, 0] = u0
        h2_mo[1, 1, 1, 1] = u1
        h2_mo[0, 0, 1, 1] = h2_mo[1, 1, 0, 0] = v
        h2_mo[0, 1, 0, 1] = h2_mo[1, 0, 1, 0] = t
        h2_mo[0, 1, 1, 0] = h2_mo[1, 0, 0, 1] = t
        return h1_mo, h2_mo, (h0, h1, u0, u1, v, t)

    @staticmethod
    def _det(modes, nq):
        sv = np.zeros(2 ** nq, dtype=complex)
        sv[sum(1 << m for m in modes)] = 1.0
        return sv

    @staticmethod
    def _block(H, nelec, nq):
        """Matrix of H on the determinants with exactly ``nelec`` occupied modes."""
        dim = 2 ** nq
        basis = np.eye(dim, dtype=complex)
        idx = [i for i in range(dim) if bin(i).count("1") == nelec]
        block = np.zeros((len(idx), len(idx)))
        for a, i in enumerate(idx):
            block[a, a] = float(np.real(H.expectation(basis[i])))
            for b in range(a + 1, len(idx)):
                j = idx[b]
                plus = (basis[i] + basis[j]) / np.sqrt(2.0)
                minus = (basis[i] - basis[j]) / np.sqrt(2.0)
                block[a, b] = block[b, a] = 0.5 * (
                    float(np.real(H.expectation(plus)))
                    - float(np.real(H.expectation(minus)))
                )
        return idx, block

    def test_shapes_and_spin_blocks(self):
        h1_mo, h2_mo, _ = self._two_orbital_model()
        h1_so, h2_so = _spin_orbital_integrals(h1_mo, h2_mo)

        assert h1_so.shape == (4, 4)
        assert h2_so.shape == (4, 4, 4, 4)
        # interleaved: spatial p -> modes 2p (alpha) and 2p+1 (beta)
        for p in range(2):
            for q in range(2):
                for s in (0, 1):
                    assert h1_so[2 * p + s, 2 * q + s] == pytest.approx(h1_mo[p, q])
        # spin is conserved on the one-body term
        assert np.allclose(h1_so[0::2, 1::2], 0.0)
        assert np.allclose(h1_so[1::2, 0::2], 0.0)

    def test_operator_reproduces_slater_condon_elements(self):
        h1_mo, h2_mo, (h0, h1, u0, u1, v, t) = self._two_orbital_model()
        H = FermionicOperator.from_coeffs(*_spin_orbital_integrals(h1_mo, h2_mo))
        H = H.jordan_wigner(4)

        hf = self._det((0, 1), 4)      # spatial orbital 0 doubly occupied
        de = self._det((2, 3), 4)      # spatial orbital 1 doubly occupied
        assert float(np.real(H.expectation(hf))) == pytest.approx(2 * h0 + u0, abs=1e-10)
        assert float(np.real(H.expectation(de))) == pytest.approx(2 * h1 + u1, abs=1e-10)

        # one electron per orbital: Coulomb only for anti-parallel spins,
        # Coulomb minus exchange for parallel spins
        assert float(np.real(H.expectation(self._det((0, 3), 4)))) == pytest.approx(
            h0 + h1 + v, abs=1e-10
        )
        assert float(np.real(H.expectation(self._det((0, 2), 4)))) == pytest.approx(
            h0 + h1 + v - t, abs=1e-10
        )

        # The pair 0->1 scattering integral is (01|01), not (00|11): recover it
        # from the symmetric combination.  A bra-index transpose flips this
        # element to (00|11) and moves the open-shell energies above.
        mix = (hf + de) / np.sqrt(2.0)
        e_mix = float(np.real(H.expectation(mix)))
        coupling = e_mix - 0.5 * (
            float(np.real(H.expectation(hf))) + float(np.real(H.expectation(de)))
        )
        assert coupling == pytest.approx(t, abs=1e-10)

    def test_static_correlation_ground_state(self):
        h1_mo, h2_mo, (h0, h1, u0, u1, v, t) = self._two_orbital_model()
        H = FermionicOperator.from_coeffs(
            *_spin_orbital_integrals(h1_mo, h2_mo)
        ).jordan_wigner(4)

        idx, block = self._block(H, 2, 4)
        assert np.allclose(block, block.T)

        # closed-shell 2x2: they couple through (01|01) only
        two_by_two = np.array([[2 * h0 + u0, t], [t, 2 * h1 + u1]])
        e_closed = float(np.linalg.eigvalsh(two_by_two)[0])
        pos = {state: k for k, state in enumerate(idx)}
        hf_de = [pos[0b0011], pos[0b1100]]
        assert np.allclose(block[np.ix_(hf_de, hf_de)], two_by_two)

        # ... and they are the lowest states, so the full {N=2} problem agrees
        assert e_closed < min(h0 + h1 + v, h0 + h1 + v - t) - 1e-6
        assert float(np.linalg.eigvalsh(block)[0]) == pytest.approx(e_closed, abs=1e-9)


class TestActiveSpace:
    """``_apply_active_space`` must follow the standard CAS layout."""

    @staticmethod
    def _model(n_orbs=3, n_elec=4, seed=0):
        """Synthetic 8-fold-symmetric integrals (not physical, but exact)."""
        rng = np.random.default_rng(seed)
        h1 = np.diag(np.arange(1.0, n_orbs + 1.0))
        w = rng.normal(size=(n_orbs, n_orbs))
        w = 0.5 * (w + w.T)
        h2 = np.einsum("ab,cd->abcd", w, w)
        return MolecularData(name="model", geometry="H 0 0 0", h1=h1, h2=h2,
                             nuclear_repulsion=0.5, n_orbitals=n_orbs,
                             n_electrons=n_elec)

    def test_active_window_starts_after_the_frozen_core(self):
        mol = self._model()
        h1_as, h2_as, n_elec, n_orbs, _ = mol._apply_active_space((2, 2))
        assert (n_elec, n_orbs) == (2, 2)
        # ncore = (4 - 2) / 2 = 1, so the window is orbitals 1..2, not 0..1
        assert h1_as.shape == (2, 2) and h2_as.shape == (2, 2, 2, 2)
        expected = mol.h1[1:3, 1:3].copy()
        for p in range(2):
            for q in range(2):
                expected[p, q] += 2 * mol.h2[1 + p, 1 + q, 0, 0] - mol.h2[1 + p, 0, 1 + q, 0]
        assert np.allclose(h1_as, expected)

    def test_frozen_core_energy_uses_ii_jj(self):
        mol = self._model(n_orbs=4, n_elec=6)     # ncore = 2, window = {2, 3}
        _, _, _, _, e_fc = mol._apply_active_space((2, 2))
        core = range(2)
        expected = sum(
            2.0 * mol.h1[i, i]
            + sum(2.0 * mol.h2[i, i, j, j] - mol.h2[i, j, j, i] for j in core)
            for i in core
        )
        assert e_fc == pytest.approx(expected, abs=1e-12)
        # the pre-fix index order (ij|ij) is a different number for ncore > 1
        wrong = sum(
            2.0 * mol.h1[i, i]
            + sum(2.0 * mol.h2[i, j, i, j] - mol.h2[i, j, j, i] for j in core)
            for i in core
        )
        assert abs(wrong - e_fc) > 1e-3

        # Freezing the core must not move the energy of a determinant that only
        # occupies active orbitals: full space det(0,1,2) vs CAS det(0).
        e_full = float(np.real(mol.to_hamiltonian().expectation(
            TestSpinOrbitalIntegrals._det((0, 1, 2, 3, 4, 5), 8))))
        e_cas = float(np.real(mol.to_hamiltonian(active_space=(2, 2)).expectation(
            TestSpinOrbitalIntegrals._det((0, 1), 4))))
        assert e_cas == pytest.approx(e_full, abs=1e-10)

    def test_repeated_calls_do_not_compound(self):
        mol = self._model()
        h1_before = mol.h1.copy()
        first = [float(np.real(t.coeffs)) for t in mol.to_hamiltonian(active_space=(2, 2)).terms]
        second = [float(np.real(t.coeffs)) for t in mol.to_hamiltonian(active_space=(2, 2)).terms]
        assert first == second
        assert np.array_equal(mol.h1, h1_before)

    def test_active_space_too_large_warns_and_clamps(self):
        mol = self._model()
        with pytest.warns(UserWarning, match="clamping the active space"):
            _, _, _, n_orbs, _ = mol._apply_active_space((2, 5))
        assert n_orbs == 2


class TestExplicitOrbitalSelection:
    """Orbitals can be named by index instead of inferred from (n_elec, n_orbs).

    The CAS arithmetic can only express "a frozen core, then a contiguous
    window".  ``active_indices`` / ``remove_orbitals`` lift that restriction;
    the numbers below are pinned against the window they are equivalent to.
    """

    @staticmethod
    def _model(**kw):
        return TestActiveSpace._model(**kw)

    def test_explicit_indices_match_the_implied_window(self):
        mol = self._model(n_orbs=4, n_elec=6)      # window {2, 3}, core {0, 1}
        window = mol._apply_active_space((2, 2))
        listed = mol._apply_active_space(None, active_indices=(2, 3), frozen_core=2)
        assert (listed[2], listed[3]) == (window[2], window[3]) == (2, 2)
        assert np.allclose(listed[0], window[0])
        assert np.allclose(listed[1], window[1])
        assert listed[4] == pytest.approx(window[4], abs=1e-12)

    def test_noncontiguous_indices_build_the_expected_integrals(self):
        mol = self._model(n_orbs=4, n_elec=4)
        h1_as, h2_as, n_elec, n_orbs, e_fc = mol._apply_active_space(
            None, active_indices=(1, 3), frozen_core=1)
        assert (n_elec, n_orbs) == (2, 2)

        active, core = [1, 3], [0]
        expected_h1 = mol.h1[np.ix_(active, active)].copy()
        for p in range(2):
            for q in range(2):
                for i in core:
                    expected_h1[p, q] += (
                        2.0 * mol.h2[active[p], active[q], i, i]
                        - mol.h2[active[p], i, active[q], i]
                    )
        assert np.allclose(h1_as, expected_h1)
        assert np.allclose(h2_as, mol.h2[np.ix_(active, active, active, active)])
        # one core orbital: 2 h_00 + (00|00)
        assert e_fc == pytest.approx(
            2.0 * mol.h1[0, 0] + mol.h2[0, 0, 0, 0], abs=1e-12)

    def test_frozen_core_keyword_alone_matches_the_window(self):
        mol = self._model(n_orbs=4, n_elec=6)
        alone = mol._apply_active_space(None, frozen_core=1)
        window = mol._apply_active_space((4, 3))
        assert (alone[2], alone[3]) == (window[2], window[3]) == (4, 3)
        assert np.allclose(alone[0], window[0])
        assert np.allclose(alone[1], window[1])
        assert alone[4] == pytest.approx(window[4], abs=1e-12)

    def test_remove_orbitals_equals_listing_what_is_kept(self):
        mol = self._model(n_orbs=4, n_elec=4)
        kept = mol._apply_active_space(None, active_indices=(1, 3), frozen_core=1)
        dropped = mol._apply_active_space(None, remove_orbitals=(2,), frozen_core=1)
        assert (kept[2], kept[3]) == (dropped[2], dropped[3]) == (2, 2)
        assert np.allclose(kept[0], dropped[0])
        assert np.allclose(kept[1], dropped[1])
        assert kept[4] == pytest.approx(dropped[4], abs=1e-12)

    def test_n_qubits_follow_the_explicit_selection(self):
        mol = self._model(n_orbs=4, n_elec=4)
        H = mol.to_hamiltonian(active_indices=(1, 3), frozen_core=1)
        assert len(H.terms[0].pauli_str) == 4
        mol.active_indices = (1, 3)
        assert mol.n_qubits == 4

    @pytest.mark.parametrize("kwargs, match", [
        (dict(active_indices=(1, 1)), "duplicates"),
        (dict(active_indices=(1, 9)), "outside"),
        (dict(active_indices=(0, 2), frozen_core=1), "overlap the"),
        (dict(active_indices=(1, 2), remove_orbitals=(3,)), "mutually exclusive"),
        (dict(active_indices=(1,)), "hold at most"),
    ])
    def test_invalid_selections_are_rejected(self, kwargs, match):
        mol = self._model(n_orbs=4, n_elec=4)
        with pytest.raises(ValueError, match=match):
            mol._apply_active_space(None, **kwargs)


class TestZ2Tapering:
    """Tapering must be exact: the fixed sectors have to tile the spectrum.

    A Hamiltonian whose terms are confined to the first ``n-2`` qubits and
    carry an even number of X/Y factors has a known symmetry group: total
    parity plus Z on each untouched qubit.  Tapering it in every sector and
    reassembling the spectra checks the Clifford rotation end to end, which is
    where a missing Pauli phase shows up as a silently shifted spectrum.
    """

    _M = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1.0, -1.0]).astype(complex),
    }

    @classmethod
    def _dense(cls, H, n):
        out = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for term in H.terms:
            block = np.array([[1.0 + 0j]])
            for ch in term.pauli_str:
                block = np.kron(block, cls._M[ch])
            out += complex(term.coeffs) * block
        return out

    @staticmethod
    def _model(n=5, n_terms=20, seed=3):
        rng = np.random.default_rng(seed)
        active = n - 2
        labels = [
            "".join(chars) + "I" * (n - active)
            for chars in itertools.product("IXYZ", repeat=active)
            if any(ch != "I" for ch in chars)
            and sum(ch in "XY" for ch in chars) % 2 == 0
        ]
        rng.shuffle(labels)
        return Hamiltonian([
            PauliString(lab, coeffs=complex(rng.normal()))
            for lab in labels[:n_terms]
        ])

    def test_only_commuting_z_strings_are_reported(self):
        H = self._model()
        syms = z2_symmetries(H)
        assert len(syms) >= 2
        for label in syms:
            assert set(label) <= {"I", "Z"}
            for term in H.terms:
                overlap = sum(1 for q, ch in enumerate(term.pauli_str)
                              if ch in "XY" and label[q] == "Z")
                assert overlap % 2 == 0, f"{label} fails on {term.pauli_str}"

    def test_sectors_tile_the_full_spectrum(self):
        H = self._model()
        n = len(H.terms[0].pauli_str)
        syms = z2_symmetries(H)
        full = np.linalg.eigvalsh(self._dense(H, n))

        seen = []
        for sector in itertools.product((1, -1), repeat=len(syms)):
            tapered, info = taper_z2(H, sector=list(sector))
            assert info["n_qubits_after"] == n - len(syms)
            seen.extend(np.linalg.eigvalsh(
                self._dense(tapered, info["n_qubits_after"])).tolist())
        assert np.allclose(np.sort(seen), full, atol=1e-10)

    def test_reference_agrees_with_an_explicit_sector(self):
        H = self._model()
        by_ref, info_ref = taper_z2(H, reference=[1, 1, 0, 0, 0])
        by_sector, info_sec = taper_z2(H, sector=info_ref["sector"])
        assert info_ref["sector"] == info_sec["sector"]

        left = {t.pauli_str: complex(t.coeffs) for t in by_ref.terms}
        right = {t.pauli_str: complex(t.coeffs) for t in by_sector.terms}
        assert left.keys() == right.keys()
        for key in left:
            assert left[key] == pytest.approx(right[key], abs=1e-12)

    def test_h2_library_sectors_partition_the_spectrum(self):
        H = get_molecular_hamiltonian("H2")
        full = np.linalg.eigvalsh(self._dense(H, 2))
        # the built-in H2 constants are stored to 7 decimals, so the FCI value
        # (-1.137306035753) is only reproduced to ~1e-8
        assert full[0] == pytest.approx(-1.13730604, abs=1e-6)

        seen = []
        for value in (1, -1):
            tapered, info = taper_z2(H, sector=[value])
            seen.extend(np.linalg.eigvalsh(
                self._dense(tapered, info["n_qubits_after"])).tolist())
        assert np.allclose(np.sort(seen), full, atol=1e-12)

    def test_clifford_gates_are_reported(self):
        H = self._model()
        n_sym = len(z2_symmetries(H))
        _, info = taper_z2(H, sector=[1] * n_sym)
        assert info["n_qubits_after"] == info["n_qubits_before"] - n_sym
        gates = info["clifford_gates"]
        # a generator that is already a single Z needs no gate at all
        assert len(gates) <= n_sym * (info["n_qubits_before"] - 1)
        for control, target in gates:
            assert 0 <= control < info["n_qubits_before"]
            assert 0 <= target < info["n_qubits_before"]
            assert control != target

    def test_malformed_sector_is_rejected(self):
        H = self._model()
        n_sym = len(z2_symmetries(H))
        with pytest.raises(ValueError, match="entries"):
            taper_z2(H, sector=[1] * (n_sym + 1))
        with pytest.raises(ValueError, match="must be"):
            taper_z2(H, sector=[2] + [1] * (n_sym - 1))
        with pytest.raises(ValueError, match="entries"):
            taper_z2(H, reference=[1, 0])
        with pytest.warns(UserWarning, match="wrong sector"):
            taper_z2(H)


class TestOpenShell:
    """Only the frozen core has to be closed-shell, and the spin must be usable.

    The old code warned on every ``spin != 0`` molecule and then froze a
    closed-shell core anyway.  That warning was noise: freezing a closed shell
    is exact whenever the active space holds all the unpaired electrons.
    The spin now appears in the one place it actually matters -- the reference
    occupation, which is what an open-shell run has to start from.
    """

    @staticmethod
    def _model(n_orbs=4, n_elec=3, spin=1, seed=0):
        """Synthetic integrals; the closed-shell numbers are checked elsewhere."""
        rng = np.random.default_rng(seed)
        h1 = np.diag(np.arange(1.0, n_orbs + 1.0))
        w = rng.normal(size=(n_orbs, n_orbs))
        w = 0.5 * (w + w.T)
        h2 = np.einsum("ab,cd->abcd", w, w)
        return MolecularData(name="model", geometry="Li 0 0 0", h1=h1, h2=h2,
                             nuclear_repulsion=0.5, n_orbitals=n_orbs,
                             n_electrons=n_elec, spin=spin)

    def test_closed_shell_core_does_not_warn(self):
        # 3 electrons, 2S = 1, nothing frozen: the core is empty, so it is
        # trivially closed-shell and there is nothing to report.
        mol = self._model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mol.to_hamiltonian(active_space=(3, 4))
        assert [str(w.message) for w in caught] == []

    def test_reference_occupation_uses_p_equals_2p_plus_s(self):
        # 3 electrons with 2S = 1  ->  n_alpha = 2, n_beta = 1, so the two
        # lowest alpha modes (0, 2) and the lowest beta mode (1) are on.
        mol = self._model()
        assert mol.reference_occupation(active_space=(3, 4)) == [
            1, 1, 1, 0, 0, 0, 0, 0,
        ]

    @pytest.mark.parametrize("n_elec, spin, cas", [
        (4, 0, (4, 3)),      # singlet, high spin == low spin
        (3, 1, (3, 4)),      # doublet
        (2, 2, (2, 3)),      # triplet, Ms = S
    ])
    def test_reference_occupation_conserves_electrons(self, n_elec, spin, cas):
        mol = self._model(n_orbs=5, n_elec=n_elec, spin=spin)
        occ = mol.reference_occupation(active_space=cas)
        assert len(occ) == 2 * cas[1]
        assert sum(occ) == n_elec
        assert set(occ) <= {0, 1}

    def test_reference_occupation_accepts_a_lower_ms(self):
        # 2 electrons with 2S = 2: Ms = 1 is (2, 0), the default; Ms = 0 is
        # (1, 1).  Both are legal and describe the same FCI problem.
        mol = self._model(n_orbs=3, n_elec=2, spin=2)
        assert mol.reference_occupation(active_space=(2, 3)) == [1, 0, 1, 0, 0, 0]
        lower = mol.reference_occupation(active_space=(2, 3),
                                         active_electrons=(1, 1))
        assert lower == [1, 1, 0, 0, 0, 0]

    @pytest.mark.parametrize("split, match", [
        ((1, 2), "n_alpha must be"),
        ((3, 0), "implies"),
        ((2, 0), "sums to"),
        ((-1, 4), "non-negative"),
        ((2,), "pair"),
    ])
    def test_active_electrons_validation(self, split, match):
        mol = self._model()               # 3 active electrons, 2S = 1
        with pytest.raises(ValueError, match=match):
            mol.reference_occupation(active_space=(3, 4),
                                     active_electrons=split)

    def test_warns_when_no_split_exists(self):
        # 2 electrons with 2S = 1: n_alpha would be 1.5
        mol = self._model(n_orbs=3, n_elec=2, spin=1)
        with pytest.warns(UserWarning, match="different parities"):
            mol.to_hamiltonian(active_space=(2, 3))

    def test_warns_when_the_active_space_cannot_hold_the_spin(self):
        # 3 electrons with 2S = 3 needs 3 alpha orbitals, only 2 are active
        mol = self._model(n_orbs=3, n_elec=3, spin=3)
        with pytest.warns(UserWarning, match="active alpha orbitals"):
            mol.to_hamiltonian(active_space=(3, 2))

    def test_reference_determinant_energy_equals_scf(self):
        """The HF determinant has no virtual occupancy.

        Dropping virtuals therefore cannot move its energy, so any deviation
        from the SCF energy is an error in the occupation convention or in the
        frozen-core folding -- not a modelling choice.
        """
        pytest.importorskip("pyscf")
        from superfermion.chemistry.pyscf_bridge import molecule_from_geometry

        mol = molecule_from_geometry("Li 0 0 0", basis="sto-3g", spin=1)
        ham = mol.to_hamiltonian(active_space=(3, 4))
        occ = mol.reference_occupation(active_space=(3, 4))

        # <ref|H|ref> is diagonal: X/Y terms vanish and each Z on an occupied
        # mode contributes -1.
        energy = 0.0
        for term in ham.terms:
            if "X" in term.pauli_str or "Y" in term.pauli_str:
                continue
            parity = sum(1 for p, ch in enumerate(term.pauli_str)
                         if ch == "Z" and occ[p])
            energy += float(np.real(term.coeffs)) * (-1) ** parity

        assert energy == pytest.approx(mol.hf_energy, abs=1e-9)
