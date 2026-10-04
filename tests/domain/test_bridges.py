"""Cross-framework bridge domain tests."""

import pytest

import superfermion as sf
from superfermion.circuit import Circuit


pytestmark = pytest.mark.domain


class TestNativeQASM:
    def test_to_qasm3_produces_valid_string(self, bell_circuit):
        qasm = bell_circuit.to_qasm3()
        assert isinstance(qasm, str)
        assert qasm.startswith("OPENQASM 3.0;")
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm


class TestQASMRoundTrip:
    def test_sf_qasm_sf_roundtrip(self, bell_circuit):
        to_qasm = pytest.importorskip(
            "superfermion.bridge",
            reason="bridge module unavailable",
        ).to_qasm
        from_qasm = pytest.importorskip(
            "superfermion.bridge",
            reason="bridge module unavailable",
        ).from_qasm

        qasm_str = to_qasm(bell_circuit)
        assert "OPENQASM" in qasm_str
        restored = from_qasm(qasm_str)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert restored.gate_count == bell_circuit.gate_count


class TestQiskitBridge:
    def test_qiskit_roundtrip(self, bell_circuit):
        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")
        qiskit_circ = bridge.to_qiskit(bell_circuit)
        restored = bridge.from_qiskit(qiskit_circ)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert len(restored.to_gate_list()) == len(bell_circuit.to_gate_list())


class TestCirqBridge:
    def test_cirq_roundtrip(self, bell_circuit):
        pytest.importorskip("cirq")
        bridge = pytest.importorskip("superfermion.bridge")
        cirq_circ = bridge.to_cirq(bell_circuit)
        restored = bridge.from_cirq(cirq_circ)
        assert isinstance(restored, Circuit)
        assert restored.n_qubits == bell_circuit.n_qubits
        assert restored.gate_count >= 2


class TestStimBridge:
    def test_to_stim_ghz_sampling_agreement(self):
        pytest.importorskip("stim")
        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(4)
        c.h(0)
        for i in range(3):
            c.cx(i, i + 1)
        c.measure_all()
        ref = sf.run(c, method="stabilizer", shots=2000, seed=1).counts
        sc, order = bridge.to_stim(c)
        assert len(order) == 4
        samp = sc.compile_sampler().sample(2000)
        from collections import Counter
        got = Counter("".join(str(int(b)) for b in row[::-1]) for row in samp)
        # GHZ outcomes only, TVD small (both orders coincide on symmetric keys)
        assert set(got) <= {"0000", "1111"}
        t1 = sum(ref.values())
        t2 = sum(got.values())
        tvd = 0.5 * sum(
            abs(ref.get(k, 0) / t1 - got.get(k, 0) / t2) for k in set(ref) | set(got)
        )
        assert tvd < 0.05

    def test_to_stim_rejects_non_clifford(self):
        pytest.importorskip("stim")
        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(2)
        c.h(0)
        c.t(0)
        with pytest.raises(ValueError, match="Clifford"):
            bridge.to_stim(c)

    def test_to_stim_rep_code(self):
        pytest.importorskip("stim")
        bridge = pytest.importorskip("superfermion.bridge")
        from superfermion.qec.codes.linear import RepetitionCode

        sc, order = bridge.to_stim(RepetitionCode(n=3).build())
        assert len(order) == 2 + 3  # 2 ancilla + 3 data measures
        samp = sc.compile_sampler().sample(100)
        assert samp.shape[1] == len(order)


class TestQiskitEndianness:
    """Regression: to_qiskit must not reverse qubit labels (SF and Qiskit
    share the little-endian q0=LSB state order). Previously to_qiskit applied
    ``n-1-q`` while from_qiskit did not, so the bridge was not an involution
    for asymmetric circuits."""

    def test_to_qiskit_matches_qiskit_statevector(self):
        import numpy as np

        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")
        from qiskit.quantum_info import Statevector

        c = sf.Circuit(3)
        c.x(0)  # asymmetric: only qubit 0
        want = np.asarray(sf.simulate(c).numpy())
        got = np.asarray(Statevector.from_instruction(bridge.to_qiskit(c)))
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_qiskit_roundtrip_preserves_statevector(self):
        import numpy as np

        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")

        c = sf.Circuit(3)
        c.x(0)
        c.ry(0.37, 1)
        c.cx(1, 2)
        back = bridge.from_qiskit(bridge.to_qiskit(c))
        np.testing.assert_allclose(
            np.asarray(sf.simulate(back).numpy()),
            np.asarray(sf.simulate(c).numpy()),
            atol=1e-12,
        )

    def test_unitary_roundtrip_preserves_operator(self):
        import numpy as np

        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import Operator, random_unitary

        u = random_unitary(4, seed=7).data
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(UnitaryGate(u), [0, 1])
        back = bridge.to_qiskit(bridge.from_qiskit(qc))
        np.testing.assert_allclose(
            np.asarray(Operator(back).data), np.asarray(Operator(qc).data), atol=1e-12
        )

    def test_unitary_export_from_sf_matches_simulation(self):
        import numpy as np

        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")
        from qiskit.quantum_info import Statevector, random_unitary

        u = random_unitary(4, seed=11).data
        c = sf.Circuit(2).x(0)
        c.unitary(u, [0, 1])
        # SF state order and Qiskit are both little-endian: exporting a
        # circuit with an opaque unitary must reproduce SF's action.
        sf_sv = np.asarray(c.to_ir().simulate())
        qk_sv = np.asarray(Statevector.from_instruction(bridge.to_qiskit(c)))
        np.testing.assert_allclose(qk_sv, sf_sv, atol=1e-12)
