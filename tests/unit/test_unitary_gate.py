"""Unit tests for opaque unitary gate support in Circuit."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.unit


H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
CCX_MAT = np.eye(8, dtype=complex)
CCX_MAT[6, 6], CCX_MAT[6, 7] = 0, 1
CCX_MAT[7, 7], CCX_MAT[7, 6] = 0, 1


class TestUnitaryStorage:
    def test_unitary_1q_stores_matrix(self):
        c = Circuit(1)
        c.unitary(H_MAT, [0])
        c._ensure_gates()
        assert len(c._gates) == 1
        g = c._gates[0]
        assert g.name == "UNITARY"
        assert g.matrix is not None
        assert np.allclose(g.matrix, H_MAT)

    def test_unitary_2q_stores_matrix(self):
        c = Circuit(2)
        c.unitary(CNOT_MAT, [0, 1])
        c._ensure_gates()
        assert len(c._gates) == 1
        assert c._gates[0].name == "UNITARY"
        assert c._gates[0].matrix.shape == (4, 4)

    def test_unitary_3q_stores_matrix(self):
        c = Circuit(3)
        c.unitary(CCX_MAT, [0, 1, 2])
        c._ensure_gates()
        assert len(c._gates) == 1
        assert c._gates[0].matrix.shape == (8, 8)


class TestUnitaryMetrics:
    def test_unitary_gate_count(self):
        c = Circuit(2)
        c.unitary(CNOT_MAT, [0, 1])
        c.unitary(CNOT_MAT, [0, 1])
        assert c.gate_count == 2

    def test_unitary_depth(self):
        c = Circuit(2).h(0)
        c.unitary(CNOT_MAT, [0, 1])
        c.h(1)
        assert c.depth >= 2

    def test_unitary_count_ops(self):
        c = Circuit(2).h(0)
        c.unitary(CNOT_MAT, [0, 1])
        c.unitary(H_MAT, [1])
        ops = c.count_ops()
        assert ops.get("UNITARY", 0) == 2
        assert ops.get("H", 0) == 1


class TestUnitaryValidation:
    def test_rejects_non_square(self):
        with pytest.raises((ValueError, Exception)):
            Circuit(1).unitary(np.ones((2, 3), dtype=complex), [0])

    def test_rejects_non_power_of_2(self):
        with pytest.raises((ValueError, Exception)):
            Circuit(2).unitary(np.eye(3, dtype=complex), [0, 1])

    def test_rejects_non_unitary(self):
        bad = np.array([[2, 0], [0, 2]], dtype=complex)
        with pytest.raises((ValueError, Exception)):
            Circuit(1).unitary(bad, [0])

    def test_validates_qubit_count_mismatch(self):
        with pytest.raises((ValueError, Exception)):
            Circuit(2).unitary(H_MAT, [0, 1])


class TestUnitaryMiscellaneous:
    def test_unitary_mixed_with_gates(self):
        c = Circuit(2).h(0)
        c.unitary(CNOT_MAT, [0, 1])
        c.h(1)
        assert c.gate_count == 3
        c._ensure_gates()
        assert c._gates[0].name == "H"
        assert c._gates[1].name == "UNITARY"
        assert c._gates[2].name == "H"

    def test_unitary_chaining(self):
        c = Circuit(2)
        result = c.h(0).unitary(CNOT_MAT, [0, 1])
        assert result is c

    def test_unitary_no_parameters(self):
        c = Circuit(1)
        c.unitary(H_MAT, [0])
        assert c.n_parameters == 0
        assert c.parameters == []

    def test_unitary_bind_passthrough(self):
        c = Circuit(1)
        c.unitary(H_MAT, [0])
        bound = c.bind({})
        bound._ensure_gates()
        assert len(bound._gates) == 1
        assert bound._gates[0].name == "UNITARY"
        assert bound._gates[0].matrix is not None
        assert np.allclose(bound._gates[0].matrix, H_MAT)
