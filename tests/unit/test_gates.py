"""Unit tests for gate unitary matrices."""

import math

import numpy as np
import pytest

from superfermion.gates.matrices import gate_unitary_matrix


pytestmark = pytest.mark.unit


def _assert_unitary(u, tol=1e-10):
    n = u.shape[0]
    product = u @ u.conj().T
    np.testing.assert_allclose(product, np.eye(n), atol=tol)


class TestSingleQubitGates:
    @pytest.mark.parametrize("name, expected", [
        ("H", np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)),
        ("X", np.array([[0, 1], [1, 0]], dtype=np.complex128)),
        ("Y", np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
        ("Z", np.array([[1, 0], [0, -1]], dtype=np.complex128)),
        ("S", np.array([[1, 0], [0, 1j]], dtype=np.complex128)),
        ("T", np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)),
    ])
    def test_pauli_and_clifford_gates(self, name, expected):
        u = gate_unitary_matrix(name)
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)


class TestParameterizedSingleQubitGates:
    def test_rx_at_pi(self):
        u = gate_unitary_matrix("RX", [math.pi])
        expected = np.array([[0, -1j], [-1j, 0]], dtype=np.complex128)
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)

    def test_ry_at_pi_over_2(self):
        u = gate_unitary_matrix("RY", [math.pi / 2])
        expected = np.array(
            [[1 / math.sqrt(2), -1 / math.sqrt(2)],
             [1 / math.sqrt(2), 1 / math.sqrt(2)]],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)

    def test_rz_at_pi(self):
        u = gate_unitary_matrix("RZ", [math.pi])
        expected = np.array([[-1j, 0], [0, 1j]], dtype=np.complex128)
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)


class TestTwoQubitGates:
    def test_cnot_matrix(self):
        u = gate_unitary_matrix("CNOT")
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)

    def test_cz_matrix(self):
        u = gate_unitary_matrix("CZ")
        expected = np.diag([1, 1, 1, -1]).astype(np.complex128)
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)

    def test_swap_matrix(self):
        u = gate_unitary_matrix("SWAP")
        expected = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(u, expected, atol=1e-10)
        _assert_unitary(u)


class TestThreeQubitGates:
    def test_ccx_shape_and_unitarity(self):
        u = gate_unitary_matrix("CCX")
        assert u.shape == (8, 8)
        _assert_unitary(u)

    def test_toffoli_alias(self):
        u_ccx = gate_unitary_matrix("CCX")
        u_toffoli = gate_unitary_matrix("TOFFOLI")
        np.testing.assert_allclose(u_ccx, u_toffoli)


class TestUnknownGate:
    def test_unknown_gate_raises(self):
        with pytest.raises(ValueError, match="Unknown gate"):
            gate_unitary_matrix("NOT_A_GATE")
