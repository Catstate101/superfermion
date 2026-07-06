"""Unit tests for observable construction and expectation values."""

import math

import numpy as np
import pytest

from superfermion.observables.core import (
    Hamiltonian,
    PauliString,
    SparsePauliOp,
    expval,
)


pytestmark = pytest.mark.unit


class TestPauliString:
    def test_construction_and_properties(self):
        ps = PauliString("XZ", coeff=0.5)
        assert ps.pauli_str == "XZ"
        assert ps.coeffs == 0.5
        assert "PauliString('XZ'" in repr(ps)

    def test_uppercases_pauli_string(self):
        ps = PauliString("xi")
        assert ps.pauli_str == "XI"


class TestSparsePauliOp:
    def test_from_dict(self):
        op = SparsePauliOp.from_dict({"ZZ": -1.0, "XX": 0.5})
        assert len(op._terms) == 2
        assert op._terms[0] == ("ZZ", -1.0 + 0j)

    def test_from_string_indexed(self):
        op = SparsePauliOp.from_string("Z0Z1")
        assert op._terms[0][0] == "ZZ"

    def test_list_constructor(self):
        op = SparsePauliOp(["Z", "X"], coeffs=[1.0, -0.5])
        assert len(op._terms) == 2


class TestHamiltonian:
    def test_construction_from_pauli_strings(self):
        terms = [PauliString("Z"), PauliString("X", coeff=0.5)]
        ham = Hamiltonian(terms)
        assert len(ham.terms) == 2
        assert ham.to_sparse_pauli_op()._terms[1][1] == 0.5


class TestExpval:
    def test_z_on_zero_state(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        assert expval(sv, PauliString("Z")) == pytest.approx(1.0)

    def test_x_on_plus_state(self):
        sv = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2)
        assert expval(sv, PauliString("X")) == pytest.approx(1.0)

    def test_bell_state_zz_expectation(self):
        sv = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2)
        assert expval(sv, PauliString("ZZ")) == pytest.approx(1.0)

    def test_sparse_pauli_op_expectation(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        op = SparsePauliOp.from_dict({"Z": 2.0, "I": 0.5})
        assert expval(sv, op) == pytest.approx(2.5)

    def test_hamiltonian_expectation(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        ham = Hamiltonian([PauliString("Z"), PauliString("X", coeff=0.0)])
        assert expval(sv, ham) == pytest.approx(1.0)
