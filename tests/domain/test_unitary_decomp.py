"""Compiler pass tests for UnitaryDecompositionPass."""

import numpy as np
import pytest
from scipy.stats import unitary_group

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.compiler.passes import UnitaryDecompositionPass


pytestmark = pytest.mark.domain

H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT_GATE = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
SWAP_GATE = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def _trace_fidelity(u_orig, u_decomp):
    """Process fidelity: |Tr(U_orig† @ U_decomp)|² / d²."""
    d = u_orig.shape[0]
    return abs(np.trace(u_orig.conj().T @ u_decomp)) ** 2 / d**2


def _circuit_unitary(circuit):
    return np.array(circuit.to_ir().to_unitary())


class TestUnitaryDecomp1Q:
    def test_identity_decomposes_trivially(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(1)
        c.unitary(np.eye(2, dtype=complex), [0])
        result = udp.run(c)
        u = _circuit_unitary(result)
        assert _trace_fidelity(np.eye(2), u) > 0.999

    def test_hadamard_decomposition(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(1)
        c.unitary(H_MAT, [0])
        result = udp.run(c)
        u_orig = _circuit_unitary(c)
        u_decomp = _circuit_unitary(result)
        assert _trace_fidelity(u_orig, u_decomp) > 0.999
        result._ensure_gates()
        assert all(g.name != "UNITARY" for g in result._gates)

    def test_arbitrary_1q(self):
        udp = UnitaryDecompositionPass()
        np.random.seed(42)
        for _ in range(5):
            rand_u = unitary_group.rvs(2)
            c = Circuit(1)
            c.unitary(rand_u, [0])
            result = udp.run(c)
            u_orig = _circuit_unitary(c)
            u_decomp = _circuit_unitary(result)
            assert _trace_fidelity(u_orig, u_decomp) > 0.999


class TestUnitaryDecomp2Q:
    def test_cnot_decomposition(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(2)
        c.unitary(CNOT_GATE, [0, 1])
        result = udp.run(c)
        u_orig = _circuit_unitary(c)
        u_decomp = _circuit_unitary(result)
        assert _trace_fidelity(u_orig, u_decomp) > 0.999

    def test_swap_decomposition(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(2)
        c.unitary(SWAP_GATE, [0, 1])
        result = udp.run(c)
        u_orig = _circuit_unitary(c)
        u_decomp = _circuit_unitary(result)
        assert _trace_fidelity(u_orig, u_decomp) > 0.999

    def test_arbitrary_2q(self):
        udp = UnitaryDecompositionPass()
        np.random.seed(99)
        for _ in range(10):
            rand_u = unitary_group.rvs(4)
            c = Circuit(2)
            c.unitary(rand_u, [0, 1])
            result = udp.run(c)
            u_orig = _circuit_unitary(c)
            u_decomp = _circuit_unitary(result)
            assert _trace_fidelity(u_orig, u_decomp) > 0.999


class TestUnitaryDecompMisc:
    def test_preserves_other_gates(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(2).h(0).cx(0, 1).rz(0.5, 1)
        result = udp.run(c)
        result._ensure_gates()
        names = [g.name for g in result._gates]
        assert "H" in names
        assert "CNOT" in names or "CX" in names
        assert "RZ" in names

    def test_output_contains_only_basis_gates(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(2)
        c.unitary(CNOT_GATE, [0, 1])
        result = udp.run(c)
        result._ensure_gates()
        allowed = {"RZ", "RY", "CNOT", "CX"}
        for g in result._gates:
            assert g.name.upper() in allowed, f"Unexpected gate: {g.name}"

    def test_multiple_unitaries_all_decomposed(self):
        udp = UnitaryDecompositionPass()
        c = Circuit(2)
        np.random.seed(7)
        for _ in range(5):
            c.unitary(unitary_group.rvs(4), [0, 1])
        result = udp.run(c)
        result._ensure_gates()
        assert all(g.name != "UNITARY" for g in result._gates)
        u_orig = _circuit_unitary(c)
        u_decomp = _circuit_unitary(result)
        assert _trace_fidelity(u_orig, u_decomp) > 0.99

    def test_decomp_in_compile_pipeline(self):
        from superfermion.compiler.specs import get_spec
        spec = get_spec("linear_5")
        c = Circuit(2)
        c.unitary(CNOT_GATE, [0, 1])
        compiled = sf.compile(c, target=spec)
        compiled._ensure_gates()
        assert all(g.name != "UNITARY" for g in compiled._gates)
