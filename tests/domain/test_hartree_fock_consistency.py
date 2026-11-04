"""Hartree-Fock consistency, pinned.

Audit rows CH-2 / CH-3 / CH-4 / CH-5 and 72 all turn on the same question:
does a variational circuit built from the Hartree-Fock determinant stay pinned
to ``E_HF``?  That was true of the pre-fix construction, which now lives in
:func:`hardware_efficient_ansatz`; :func:`uccsd_ansatz` was rebuilt from
number-conserving rotations and reaches the exact ground state.  These tests
hold both halves of that claim down, plus every place the package *reports* a
Hartree-Fock energy, so neither can drift silently.
"""

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

import superfermion as sf
from superfermion.algorithms.variational import VQE
from superfermion.chemistry import dmet, mp2
from superfermion.chemistry import subspace as sub
from superfermion.chemistry.ansatz import hardware_efficient_ansatz, uccsd_ansatz
from superfermion.chemistry.dmet import _determinant_energy
from superfermion.chemistry.hamiltonians import get_molecular_hamiltonian
from superfermion.chemistry.library import MoleculeLibrary
from superfermion.chemistry.pyscf_bridge import _HAS_PYSCF

pytestmark = pytest.mark.domain

E_HF = -1.116998996754
E_FCI = -1.137306035753
ENUC = 0.719968994449
#: 0.735 Angstrom in Bohr -- Hartree is per Bohr, so this is the geometry's 1/R.
R_BOHR = 0.735 / 0.529177210903


def _record():
    """The single tabulated record, read without constructing a molecule."""
    return MoleculeLibrary._RECORDS[0]


def _energy_at(hamiltonian, circuit, values):
    names = list(circuit.parameters)
    bound = circuit.bind({n: float(v) for n, v in zip(names, values)})
    return float(np.real(sf.run(bound, shots=0).expectation(hamiltonian)))


def _h2_qubit():
    return get_molecular_hamiltonian("H2")


def _four_qubit():
    """The 4-spin-orbital H2 build: same molecule, explicit frozen-core-free space."""
    from superfermion.chemistry import molecule_from_geometry

    return molecule_from_geometry("H 0 0 0\nH 0 0 0.735", basis="sto-3g").to_hamiltonian()


def _expectation_matrix(hamiltonian, indices):
    """``H`` restricted to ``indices``, built only from ``H.expectation``.

    Going through the public expectation keeps the basis and qubit ordering
    identical to the statevectors ``sf.run`` returns, so this cannot disagree
    with the library about index conventions the way a hand-rolled Kronecker
    product can.
    """
    n = max(len(term.pauli_str) for term in hamiltonian.terms)
    dim = 1 << n

    def unit(i):
        vector = np.zeros(dim, dtype=complex)
        vector[i] = 1.0
        return vector

    root2 = np.sqrt(2.0)
    size = len(indices)
    matrix = np.zeros((size, size), dtype=complex)
    for a, i in enumerate(indices):
        matrix[a, a] = hamiltonian.expectation(unit(i))
        for b in range(a + 1, size):
            j = indices[b]
            real_part = (hamiltonian.expectation((unit(i) + unit(j)) / root2)
                         - hamiltonian.expectation((unit(i) - unit(j)) / root2)) / 2
            imag_part = (hamiltonian.expectation((unit(i) + 1j * unit(j)) / root2)
                         - hamiltonian.expectation((unit(i) - 1j * unit(j)) / root2)) / 2
            matrix[a, b] = real_part + 1j * imag_part
            matrix[b, a] = np.conj(matrix[a, b])
    return matrix


def _basis_index_of_qubit(qubit, n_qubits=4):
    """Which statevector bit ``x(qubit)`` sets -- measured, not assumed."""
    circuit = sf.Circuit(n_qubits)
    circuit.x(qubit)
    state = np.asarray(sf.run(circuit, shots=0).statevector, dtype=complex).ravel()
    return int(np.log2(int(np.argmax(np.abs(state)))))


class TestTabulatedRecord:
    """The numbers other modules quote must follow from the record's integrals."""

    def test_the_record_is_internally_exact(self):
        record = _record()
        h1, h2 = record["h1"], record["h2"]

        hf = 2 * h1[0, 0] + h2[0, 0, 0, 0] + record["nuclear_repulsion"]
        fci = sub.exact_energy(h1, h2, 2, 2) + record["nuclear_repulsion"]

        assert hf == pytest.approx(record["hf_energy"], abs=1e-9)
        assert fci == pytest.approx(record["fci_energy"], abs=1e-8)

    def test_nuclear_repulsion_is_one_over_r_in_bohr(self):
        """0.735 Angstrom is 1.389 Bohr; Hartree is per Bohr, not per Angstrom."""
        assert 1.0 / R_BOHR == pytest.approx(_record()["nuclear_repulsion"], abs=1e-9)

    def test_molecular_data_round_trips_every_hf_number(self):
        from superfermion.chemistry import molecule_from_geometry

        record = _record()
        molecule = molecule_from_geometry("H 0 0 0\nH 0 0 0.735", basis="sto-3g")

        assert molecule.hf_energy == pytest.approx(record["hf_energy"], abs=1e-12)
        assert molecule.nuclear_repulsion == pytest.approx(record["nuclear_repulsion"], abs=1e-12)
        if _HAS_PYSCF:
            # With PySCF installed the integrals are re-derived by a fresh SCF
            # instead of being copied out of the table, so they cannot be
            # bit-identical: entries that are analytically zero come back at the
            # 1e-17 level and the SCF adds convergence noise.  Measured
            # worst case for this record is 2.5e-13 (h1) and 4.8e-13 (h2), so
            # the bound below sits ~20x above what the solver actually does
            # while remaining far tighter than any physical significance.
            assert np.allclose(molecule.h1, record["h1"], rtol=0, atol=1e-11)
            assert np.allclose(molecule.h2, record["h2"], rtol=0, atol=1e-11)
        else:
            # No PySCF: the record is served verbatim, so this is still exact.
            assert np.array_equal(molecule.h1, record["h1"])
            assert np.array_equal(molecule.h2, record["h2"])


class TestHartreeFockState:
    """``|HF>`` is the determinant ``uccsd_ansatz`` starts from, at every size."""

    @pytest.mark.parametrize(
        "n_qubits, occupied, hamiltonian_fn",
        [
            # The reduced 2-qubit model keeps one spin orbital: |10> is HF there.
            (2, 1, _h2_qubit),
            (4, 2, _four_qubit),
        ],
        ids=["2-qubit", "4-qubit"],
    )
    def test_the_hf_determinant_expects_the_stored_hf_energy(
        self, n_qubits, occupied, hamiltonian_fn
    ):
        """Row 72: theta = 0 is HF, not "not HF" (row 79 refuted)."""
        circuit = sf.Circuit(n_qubits)
        for i in range(occupied):
            circuit.x(i)
        state = np.asarray(sf.run(circuit, shots=0).statevector, dtype=complex).ravel()

        # The 5-term 2-qubit operator is stored to ~10 decimal places, and the
        # 15-term 4-qubit build reproduces the record to 1e-13.
        assert float(np.real(hamiltonian_fn().expectation(state))) == pytest.approx(
            E_HF, abs=1e-8
        )

    def test_the_generic_ansatz_starts_at_hf(self):
        """CH-2 / CH-5: theta = 0 must be the reference, and the reference is HF."""
        hamiltonian = _four_qubit()
        ansatz = uccsd_ansatz(4, 2)
        assert _energy_at(hamiltonian, ansatz, np.zeros(ansatz.n_parameters)) == pytest.approx(
            E_HF, abs=1e-9
        )


class TestAnsatzEscapesHartreeFock:
    """CH-2 / CH-3 / CH-5 must stay fixed: the ansatz has to leave ``E_HF``."""

    def test_the_two_qubit_branch_reaches_its_hamiltonian_ground_state(self):
        hamiltonian = _h2_qubit()
        ansatz = uccsd_ansatz(2, 2)
        ground = float(np.linalg.eigvalsh(_expectation_matrix(hamiltonian, range(4)))[0])

        best = minimize_scalar(
            lambda t: _energy_at(hamiltonian, ansatz, [t]),
            bounds=(-4 * np.pi, 4 * np.pi),
            method="bounded",
            options={"xatol": 1e-12},
        )
        assert best.fun == pytest.approx(ground, abs=1e-9)
        assert best.fun < E_HF - 1e-3

    def test_the_generic_branch_reaches_the_exact_ground_state(self):
        """CH-3: singles move an electron, so the reachable set is not a frozen slice."""
        hamiltonian = _four_qubit()
        ansatz = uccsd_ansatz(4, 2)
        exact = float(np.linalg.eigvalsh(sub.pauli_matrix(hamiltonian))[0])

        best = minimize(
            lambda x: _energy_at(hamiltonian, ansatz, x),
            np.zeros(ansatz.n_parameters),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
        )
        assert best.fun == pytest.approx(exact, abs=1e-8)
        assert best.fun < E_HF - 1e-3

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift"])
    @pytest.mark.parametrize("seed", [1, 2])
    def test_vqe_escapes_hf_on_the_generic_branch(self, diff_method, seed):
        """CH-5: the *driver*, not just the ansatz, has to leave the reference."""
        vqe = VQE(uccsd_ansatz(4, 2), _four_qubit(), device="cpu",
                  method="statevector", optimizer="L-BFGS-B", shots=0,
                  diff_method=diff_method)
        result = vqe.minimize(seed=seed)

        assert result.optimal_value == pytest.approx(E_FCI, abs=1e-6)
        assert result.optimal_value < E_HF - 1e-3

    def test_the_reported_optimum_is_reproducible_from_its_own_parameters(self):
        """The value VQE reports must be the energy of the parameters it returns."""
        hamiltonian = _four_qubit()
        ansatz = uccsd_ansatz(4, 2)
        result = VQE(ansatz, hamiltonian, optimizer="L-BFGS-B").minimize(seed=3)

        values = [result.optimal_params[name] for name in ansatz.parameters]
        assert result.optimal_value == pytest.approx(
            _energy_at(hamiltonian, ansatz, values), abs=1e-9
        )

    def test_the_generic_ansatz_conserves_particle_number(self):
        """CH-3: a copying ``cx(occupied, virtual)`` changes N; a hopping rotation does not."""
        ansatz = uccsd_ansatz(4, 2)
        number = np.diag([bin(i).count("1") for i in range(16)]).astype(float)

        rng = np.random.default_rng(3)
        for _ in range(12):
            values = rng.uniform(-np.pi, np.pi, ansatz.n_parameters)
            bound = ansatz.bind(dict(zip(ansatz.parameters, values)))
            state = np.asarray(sf.run(bound, shots=0).statevector, dtype=complex).ravel()
            assert float(np.real(np.vdot(state, number @ state))) == pytest.approx(2.0, abs=1e-9)


class TestPreFixAnsatzStaysAtHartreeFock:
    """``hardware_efficient_ansatz`` is the retained pre-fix construction.

    Its docstring says its whole-parameter-space minimum equals ``E_HF``.  Both
    halves of that are pinned here: the reachable subspace cannot go lower, and
    a real optimiser really does stop at the reference.
    """

    def test_the_reachable_subspace_bottoms_out_at_hf(self):
        """CH-4: qubits 0 and 1 are only ever controls, so only 2^(4-2) states are reachable."""
        hamiltonian = _four_qubit()
        bit0 = _basis_index_of_qubit(0)
        bit1 = _basis_index_of_qubit(1)
        reachable = [index for index in range(16)
                     if (index >> bit0) & 1 and (index >> bit1) & 1]

        assert len(reachable) == 4
        lowest = float(np.linalg.eigvalsh(_expectation_matrix(hamiltonian, reachable))[0])
        assert lowest == pytest.approx(E_HF, abs=1e-6)

    def test_the_pre_fix_ansatz_leaves_the_reference_particle_number(self):
        """CH-3: the copying singles reach N = 3 and 4, which the fixed ansatz cannot."""
        ansatz = hardware_efficient_ansatz(4, 2)
        number = np.diag([bin(i).count("1") for i in range(16)]).astype(float)

        rng = np.random.default_rng(4)
        deviations = []
        for _ in range(8):
            values = rng.uniform(-np.pi, np.pi, ansatz.n_parameters)
            bound = ansatz.bind(dict(zip(ansatz.parameters, values)))
            state = np.asarray(sf.run(bound, shots=0).statevector, dtype=complex).ravel()
            deviations.append(abs(float(np.real(np.vdot(state, number @ state))) - 2.0))

        assert max(deviations) > 1e-3

    def test_an_optimiser_stops_at_hf(self):
        hamiltonian = _four_qubit()
        ansatz = hardware_efficient_ansatz(4, 2)
        best = minimize(
            lambda x: _energy_at(hamiltonian, ansatz, x),
            np.zeros(ansatz.n_parameters),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
        )
        assert best.fun == pytest.approx(E_HF, abs=1e-6)
        assert best.fun > E_FCI

    def test_the_two_qubit_instance_is_a_constant_well_above_hf(self):
        """CH-4 is a (4,2) result: on (2,2) the reference is |11>, not |10>.

        The only parameter is a global phase, so the energy cannot move at all,
        and the constant it is stuck on sits 0.7733 Ha above Hartree-Fock.
        """
        hamiltonian = _h2_qubit()
        ansatz = hardware_efficient_ansatz(2, 2)
        assert list(ansatz.parameters) == ["d_0"]

        energies = [_energy_at(hamiltonian, ansatz, [float(t)])
                    for t in (0.0, 0.5, 1.0, 2.0, np.pi)]
        assert max(energies) - min(energies) < 1e-12, "d_0 must be a global phase"
        assert energies[0] == pytest.approx(-0.3436844, abs=1e-7)
        assert energies[0] - E_HF == pytest.approx(0.7733, abs=1e-4)


class TestReportedHartreeFockEnergies:
    """Every module that reports a Hartree-Fock energy must mean the same number."""

    def test_the_determinant_energy_is_the_closed_shell_formula(self):
        record = _record()
        h1, h2 = record["h1"], record["h2"]

        determinant = _determinant_energy(h1, h2, range(1))
        assert determinant == pytest.approx(2 * h1[0, 0] + h2[0, 0, 0, 0], abs=1e-12)
        assert determinant + ENUC == pytest.approx(E_HF, abs=1e-9)

    def test_dmet_reports_the_tabulated_hf(self):
        record = _record()
        result = dmet.one_shot_dmet(
            record["h1"], record["h2"], 2, [[0, 1]], nuclear_repulsion=ENUC
        )
        assert result["hf_energy"] == pytest.approx(E_HF, abs=1e-9)
        assert result["energy"] == pytest.approx(E_FCI, abs=1e-8)

    def test_dmet_double_counts_overlapping_impurities_and_deduplicates_them(self):
        """The documented 0.0203 Ha: a localised split solves the whole correlation twice."""
        record = _record()
        theta = np.pi / 4
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        result = dmet.one_shot_dmet(
            record["h1"], record["h2"], 2, [[0], [1]],
            nuclear_repulsion=ENUC, orbitals=rotation,
        )
        raw = result["energy_electronic_raw"] + ENUC

        assert all(fragment["n_bath"] > 0 for fragment in result["fragments"])
        assert E_FCI - raw == pytest.approx(0.0203, abs=1e-3)
        assert result["energy"] == pytest.approx(E_FCI, abs=1e-8)

    def test_mp2_is_a_correction_between_hf_and_fci(self):
        record = _record()
        h1, h2 = record["h1"], record["h2"]

        fock = h1 + 2.0 * h2[:, :, 0, 0] - h2[:, 0, 0, :]
        orbital_energies = np.diag(fock)

        correlation = mp2.mp2_correlation_energy(h1, h2, orbital_energies, 1)
        assert correlation < 0.0
        assert E_HF + correlation > E_FCI
        assert E_HF + correlation < E_HF
        # The amplitude and energy routes are two spellings of one contraction.
        assert mp2.mp2_amplitudes(h1, h2, orbital_energies, 1)["e2"] == pytest.approx(
            correlation, abs=1e-14
        )
