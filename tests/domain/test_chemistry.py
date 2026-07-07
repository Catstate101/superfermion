"""Chemistry module domain tests."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.chemistry.ansatz import uccsd_ansatz
from superfermion.chemistry.hamiltonians import FermionicOperator, get_molecular_hamiltonian


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
