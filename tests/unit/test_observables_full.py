"""Unit tests for pauli.py convenience constructors."""

import math

import numpy as np
import pytest

from superfermion.observables.core import PauliString, expval
from superfermion.observables.pauli import I, X, Y, Z


pytestmark = pytest.mark.unit


class TestPauliConvenienceConstructors:
    def test_z_with_n_qubits(self):
        ps = Z(0, n_qubits=2)
        assert isinstance(ps, PauliString)
        assert ps.pauli_str == "ZI"
        assert ps.coeffs == 1.0

    def test_x_with_n_qubits(self):
        ps = X(1, n_qubits=3)
        assert ps.pauli_str == "IXI"

    def test_y_with_n_qubits(self):
        ps = Y(2, n_qubits=3)
        assert ps.pauli_str == "IIY"

    def test_i_with_n_qubits(self):
        ps = I(1, n_qubits=2)
        assert ps.pauli_str == "II"

    def test_z_minimal_without_n_qubits(self):
        ps = Z(0)
        assert ps.pauli_str == "Z"

    def test_x_minimal_without_n_qubits(self):
        ps = X(0)
        assert ps.pauli_str == "X"

    def test_y_minimal_without_n_qubits(self):
        ps = Y(0)
        assert ps.pauli_str == "Y"

    def test_i_minimal_without_n_qubits(self):
        ps = I(0)
        assert ps.pauli_str == "I"


class TestPauliConvenienceExpectationValues:
    def test_z_on_zero_state(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        assert expval(sv, Z(0, n_qubits=1)) == pytest.approx(1.0)

    def test_x_on_plus_state(self):
        sv = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2)
        assert expval(sv, X(0, n_qubits=1)) == pytest.approx(1.0)

    def test_y_on_zero_state(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        assert expval(sv, Y(0, n_qubits=1)) == pytest.approx(0.0)

    def test_i_on_any_state(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        assert expval(sv, I(0, n_qubits=1)) == pytest.approx(1.0)

    def test_two_qubit_z_on_bell_state(self):
        sv = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2)
        assert expval(sv, Z(0, n_qubits=2)) == pytest.approx(0.0)
        assert expval(sv, Z(1, n_qubits=2)) == pytest.approx(0.0)


class TestPauliConvenienceRepr:
    def test_repr_matches_pauli_string(self):
        ps = Z(0, n_qubits=2)
        assert repr(ps) == "PauliString('ZI', coeff=1.0)"
