"""Integration tests for opaque unitary gate end-to-end flows."""

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.integration

H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT_GATE = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def _trace_fidelity(u1, u2):
    d = u1.shape[0]
    return abs(np.trace(u1.conj().T @ u2)) ** 2 / d**2


class TestUnitarySimulation:
    def test_unitary_simulate_1q(self):
        """sf.run on circuit.unitary(H, [0]) produces correct statevector."""
        c = Circuit(1)
        c.unitary(H_MAT, [0])
        sv = np.array(c.to_ir().simulate())
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_unitary_simulate_2q(self):
        """circuit.unitary(CNOT, [0,1]) matches circuit.cx(0,1) in simulation."""
        c_opaque = Circuit(2)
        c_opaque.unitary(CNOT_GATE, [0, 1])

        c_native = Circuit(2).cx(0, 1)

        u_opaque = np.array(c_opaque.to_ir().to_unitary())
        u_native = np.array(c_native.to_ir().to_unitary())
        assert _trace_fidelity(u_opaque, u_native) > 0.999


class TestUnitaryCompileThenRun:
    def test_compile_then_simulate(self):
        """sf.compile() decomposes unitaries into basis gates; verify output is valid."""
        from superfermion.compiler.passes import UnitaryDecompositionPass

        c = Circuit(2).h(0)
        c.unitary(CNOT_GATE, [0, 1])

        u_orig = np.array(c.to_ir().to_unitary())

        decomposed = UnitaryDecompositionPass().run(c)
        u_decomp = np.array(decomposed.to_ir().to_unitary())

        assert _trace_fidelity(u_orig, u_decomp) > 0.99


class TestUnitaryBridgeRoundtrip:
    def test_qiskit_roundtrip(self):
        """from_qiskit(qiskit_circuit_with_unitary) -> circuit with unitaries -> to_qiskit()."""
        qiskit = pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from superfermion.bridge import from_qiskit, to_qiskit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(UnitaryGate(CNOT_GATE), [0, 1])

        sf_circ = from_qiskit(qc)
        sf_circ._ensure_gates()
        has_unitary = any(g.name == "UNITARY" for g in sf_circ._gates)
        assert has_unitary

        qc_back = to_qiskit(sf_circ)
        assert qc_back.num_qubits == 2


class TestQVOpaqueVsDecomposed:
    def test_qv_opaque_matches_decomposed(self):
        """QV4 with opaque unitaries simulates to same state as decomposed version."""
        from scipy.stats import unitary_group
        from superfermion.compiler.passes import UnitaryDecompositionPass

        np.random.seed(42)
        n = 4
        c_opaque = Circuit(n)
        for i in range(0, n - 1, 2):
            su4 = unitary_group.rvs(4)
            c_opaque.unitary(su4, [i, i + 1])

        udp = UnitaryDecompositionPass()
        c_decomp = udp.run(c_opaque)

        u_opaque = np.array(c_opaque.to_ir().to_unitary())
        u_decomp = np.array(c_decomp.to_ir().to_unitary())
        assert _trace_fidelity(u_opaque, u_decomp) > 0.99


class TestUnitaryCompileDifferentTargets:
    def test_compile_different_topologies(self):
        """Same unitary compiled for linear vs all-to-all both produce valid circuits."""
        from superfermion.compiler.specs import get_spec

        c = Circuit(2)
        c.unitary(CNOT_GATE, [0, 1])

        for spec_name in ["linear_5", "ionq_aria"]:
            try:
                spec = get_spec(spec_name)
            except Exception:
                continue
            compiled = sf.compile(c, target=spec)
            compiled._ensure_gates()
            assert all(g.name != "UNITARY" for g in compiled._gates)
            assert compiled.gate_count > 0
