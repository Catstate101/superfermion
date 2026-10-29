"""Cross-framework bridge domain tests."""

import threading

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


class TestControlledUGamma:
    """SF's CU gate carries the Qiskit gamma phase natively (matrix
    ``diag(I, e^{i\\gamma}·U3(\\theta,\\phi,\\lambda))``); gamma defaults to 0 so
    the original 5-argument calls are unchanged."""

    TH, PH, LM, G = 0.7, 0.3, 0.2, 0.9

    def test_gamma_matches_qiskit(self):
        import numpy as np

        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        qc = QuantumCircuit(2)
        qc.cu(self.TH, self.PH, self.LM, self.G, 0, 1)
        sf_u = np.asarray(
            sf.Circuit(2).cu(self.TH, self.PH, self.LM, 0, 1, self.G).to_ir().to_unitary()
        )
        np.testing.assert_allclose(sf_u, np.asarray(Operator(qc).data), atol=1e-12)

    def test_default_gamma_is_zero(self):
        import numpy as np

        u5 = np.asarray(sf.Circuit(2).cu(self.TH, self.PH, self.LM, 0, 1).to_ir().to_unitary())
        u6 = np.asarray(
            sf.Circuit(2).cu(self.TH, self.PH, self.LM, 0, 1, 0.0).to_ir().to_unitary()
        )
        np.testing.assert_allclose(u5, u6, atol=1e-12)

    def test_from_qiskit_cu_stays_native_with_gamma(self):
        import numpy as np

        pytest.importorskip("qiskit")
        bridge = pytest.importorskip("superfermion.bridge")
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        qc = QuantumCircuit(2)
        qc.cu(self.TH, self.PH, self.LM, self.G, 0, 1)
        sfc = bridge.from_qiskit(qc)
        assert [g["name"] for g in sfc.to_gate_list()] == ["CU"]
        np.testing.assert_allclose(
            np.asarray(sf.simulate(sfc).numpy()),
            np.asarray(Statevector.from_instruction(qc)),
            atol=1e-12,
        )
        back = bridge.to_qiskit(sfc)
        from qiskit.quantum_info import Operator

        np.testing.assert_allclose(
            np.asarray(Operator(back).data), np.asarray(Operator(qc).data), atol=1e-12
        )

    def test_parametric_gamma_binds(self):
        import numpy as np

        c = sf.Circuit(2).cu(
            sf.param("t"), sf.param("p"), sf.param("l"), 0, 1, sf.param("g")
        )
        bound = c.bind({"t": self.TH, "p": self.PH, "l": self.LM, "g": self.G})
        ref = sf.Circuit(2).cu(self.TH, self.PH, self.LM, 0, 1, self.G)
        np.testing.assert_allclose(
            np.asarray(bound.to_ir().to_unitary()),
            np.asarray(ref.to_ir().to_unitary()),
            atol=1e-12,
        )


class TestQASMEndianness:
    """SF is little-endian pass-through (see TestQiskitEndianness), so the
    pure-Python from_qasm fallback must NOT reverse qubit labels. This test
    forces the Python fallback: a >2^53 integer placed in a *comment* trips the
    _has_bigint pre-scan (skipping the Rust parser) while the parser itself
    ignores the comment line, so it locks the corrected convention."""

    def test_qasm_python_fallback_roundtrip_preserves_statevector(self):
        import numpy as np

        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.cz(1, 2)
        c.swap(0, 2)
        qasm = bridge.to_qasm(c)
        # comment-only big int -> _has_bigint True -> Rust path skipped,
        # pure-Python parser used instead.
        qasm_fb = qasm + "\n// 99999999999999999\n"
        restored = bridge.from_qasm(qasm_fb)
        np.testing.assert_allclose(
            np.asarray(sf.simulate(restored).numpy()),
            np.asarray(sf.simulate(c).numpy()),
            atol=1e-9,
        )

    def test_qasm_rust_path_roundtrip_preserves_statevector(self):
        """Locks the Rust ``QuantumDAG.from_qasm2`` endianness fix: SF, Qiskit
        and OpenQASM all use little-endian (q0 = LSB) pass-through, so
        SF -> to_qasm -> from_qasm via the default (Rust) parser IS an
        involution for asymmetric circuits. Previously the Rust parser mapped
        q[i] -> (n-1-i) (big-endian), contradicting to_qasm/from_qiskit; fixed
        in sf-ir/src/qasm.rs."""
        import numpy as np

        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.swap(0, 2)
        restored = bridge.from_qasm(bridge.to_qasm(c))
        np.testing.assert_allclose(
            np.asarray(sf.simulate(restored).numpy()),
            np.asarray(sf.simulate(c).numpy()),
            atol=1e-9,
        )
        # The docstring above is only true if the native parser actually ran.
        # It used to be unfalsifiable: from_qasm caught BaseException, so a
        # panicking native parser fell through to the Python parser, produced
        # this same statevector, and the test passed with the Rust path dead.
        pytest.importorskip("superfermion._sf_core")
        assert bridge._RUST_QASM_FALLBACK_REASON is None, (
            "from_qasm did not use the native parser: "
            f"{bridge._RUST_QASM_FALLBACK_REASON}"
        )

    def test_a_native_parser_failure_is_not_swallowed(self, monkeypatch):
        """Regression: ``from_qasm`` wrapped the native call in
        ``except BaseException``, so ``pyo3_runtime.PanicException`` -- and every
        other native failure -- was absorbed and the Python parser returned a
        correct circuit. A broken Rust parser was invisible to all tests.
        Only ImportError/ValueError/TypeError may fall back now."""
        bridge = pytest.importorskip("superfermion.bridge")
        sf_core = pytest.importorskip("superfermion._sf_core")

        class _BrokenDAG:
            @staticmethod
            def from_qasm2(_qasm):
                raise RuntimeError("simulated native parser failure")

        monkeypatch.setattr(sf_core, "QuantumDAG", _BrokenDAG, raising=False)
        with pytest.raises(RuntimeError, match="simulated native parser failure"):
            bridge.from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n')

    def test_a_rejected_string_still_falls_back(self, monkeypatch):
        """Narrowing the except must not remove the documented fallback: a
        ValueError (what ``from_qasm2`` raises for syntax it cannot parse) still
        reaches the Python parser, and the reason is recorded for callers."""
        bridge = pytest.importorskip("superfermion.bridge")
        sf_core = pytest.importorskip("superfermion._sf_core")

        class _RejectingDAG:
            @staticmethod
            def from_qasm2(_qasm):
                raise ValueError("unsupported syntax")

        monkeypatch.setattr(sf_core, "QuantumDAG", _RejectingDAG, raising=False)
        circuit = bridge.from_qasm(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
        )
        assert len(circuit._gates) == 2
        assert bridge._RUST_QASM_FALLBACK_REASON == "ValueError: unsupported syntax"

    def test_a_conversion_failure_after_a_good_parse_propagates(self, monkeypatch):
        """The guard covers the parse only. ``to_gate_records``/``Circuit``/
        ``extend_raw_from_records`` are Rust-backed conversion, so a failure
        there is a defect and must not be absorbed into a Python re-parse."""
        bridge = pytest.importorskip("superfermion.bridge")
        sf_core = pytest.importorskip("superfermion._sf_core")

        class _BadRecordsDAG:
            @staticmethod
            def from_qasm2(_qasm):
                return _BadRecordsDAG()

            def to_gate_records(self):
                raise RuntimeError("simulated conversion failure")

            def n_qubits(self):
                return 1

            def n_cbits(self):
                return 1

        monkeypatch.setattr(sf_core, "QuantumDAG", _BadRecordsDAG, raising=False)
        with pytest.raises(RuntimeError, match="simulated conversion failure"):
            bridge.from_qasm('OPENQASM 2.0;\nqreg q[1];\nh q[0];\n')
        assert bridge._RUST_QASM_FALLBACK_REASON.startswith("conversion-failed:")

    def test_the_reason_is_reset_before_anything_can_raise(self):
        """Regression: the reset used to sit after the big-integer pre-scan, so
        a call that threw during the pre-scan left the previous call's reason in
        place, and ``None`` is documented to mean 'the native parser ran'."""
        bridge = pytest.importorskip("superfermion.bridge")

        bridge.from_qasm('OPENQASM 2.0;\nqreg q[1];\nh q[0];\n// 99999999999999999\n')
        assert bridge._RUST_QASM_FALLBACK_REASON == "bigint-parameter"

        with pytest.raises(TypeError):
            bridge.from_qasm(None)
        assert bridge._RUST_QASM_FALLBACK_REASON is None, (
            "a reason leaked from the previous call: "
            f"{bridge._RUST_QASM_FALLBACK_REASON}"
        )

    def test_the_reason_is_per_thread(self):
        """A module-level string let two threads overwrite each other's reason,
        so the value a caller read need not describe its own call."""
        bridge = pytest.importorskip("superfermion.bridge")
        bridge.from_qasm('OPENQASM 2.0;\nqreg q[1];\nh q[0];\n')
        assert bridge._RUST_QASM_FALLBACK_REASON is None

        seen = {}

        def worker():
            bridge.from_qasm(
                'OPENQASM 2.0;\nqreg q[1];\nh q[0];\n// 99999999999999999\n'
            )
            seen["reason"] = bridge._RUST_QASM_FALLBACK_REASON

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert seen["reason"] == "bigint-parameter"
        assert bridge._RUST_QASM_FALLBACK_REASON is None, (
            "the worker's reason leaked into this thread: "
            f"{bridge._RUST_QASM_FALLBACK_REASON}"
        )

    def test_the_bigint_skip_is_recorded(self):
        bridge = pytest.importorskip("superfermion.bridge")
        circuit = bridge.from_qasm(
            'OPENQASM 2.0;\nqreg q[1];\nh q[0];\n// 99999999999999999\n'
        )
        assert len(circuit._gates) == 1
        assert bridge._RUST_QASM_FALLBACK_REASON == "bigint-parameter"

    def test_an_unsupported_gate_raises_instead_of_being_dropped(self):
        """Regression: the native parser skipped unmapped gates, so the parse
        'succeeded' with operations missing and the reason still said the native
        path handled it. Both parsers are strict now."""
        bridge = pytest.importorskip("superfermion.bridge")
        with pytest.raises(ValueError, match="unsupported gate 'foo'"):
            bridge.from_qasm('OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n')
        assert bridge._RUST_QASM_FALLBACK_REASON is not None

    def test_u2_is_expanded_rather_than_dropped(self):
        # u2(phi, lam) == u3(pi/2, phi, lam); it used to vanish silently.
        bridge = pytest.importorskip("superfermion.bridge")
        circuit = bridge.from_qasm('OPENQASM 2.0;\nqreg q[1];\nu2(0.3, 0.7) q[0];\n')
        assert len(circuit._gates) == 1

    def test_measure_is_preserved(self):
        bridge = pytest.importorskip("superfermion.bridge")
        circuit = bridge.from_qasm(
            'OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[0];\n'
        )
        assert [g.name for g in circuit._gates] == ["H", "MEASURE"]


class TestToQasmDropGuard:
    """Regression: to_qasm silently `continue`d on gates missing from its
    map, emitting a circuit with operations dropped. It must raise (like
    to_qiskit / to_cirq / to_ionq) except for BARRIER, which has no OpenQASM
    2.0 equivalent."""

    def test_to_qasm_raises_on_unmappable_gate(self):
        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(2)
        c.h(0)
        c.iswap(0, 1)  # ISWAP is not in to_qasm's GATE_MAP
        with pytest.raises(ValueError, match="OpenQASM"):
            bridge.to_qasm(c)

    def test_to_qasm_still_omits_barrier(self):
        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(2)
        c.h(0)
        c.barrier()
        c.cx(0, 1)
        qasm = bridge.to_qasm(c)
        assert "barrier" not in qasm.lower()
        assert "cx q[0], q[1];" in qasm


class TestRustStorageExporters:
    """Regression: to_qiskit called _ensure_gates() but the other exporters
    iterated circuit._gates directly, so a use_rust_storage circuit exported
    as *nothing*. These pure-Python exporters must materialize first."""

    def test_to_qasm_includes_rust_stored_gates(self):
        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(3, use_rust_storage=True)
        c.h(0)
        c.cx(0, 1)
        qasm = bridge.to_qasm(c)
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm

    def test_to_ionq_includes_rust_stored_gates(self):
        bridge = pytest.importorskip("superfermion.bridge")
        c = sf.Circuit(2, use_rust_storage=True)
        c.h(0)
        c.cx(0, 1)
        names = [g["gate"] for g in bridge.to_ionq(c)]
        assert "h" in names and "cnot" in names


class TestCirqUnitaryEndianness:
    """Lock: cirq.unitary() returns big-endian matrices and circuit.unitary()
    takes big-endian matrices, so from_cirq/to_cirq pass through with NO bit
    reversal -- a round-trip must preserve the simulated state."""

    def test_cirq_unitary_roundtrip_preserves_statevector(self):
        import numpy as np

        try:
            import cirq
        except Exception:
            pytest.skip("cirq not importable in this environment")
        bridge = pytest.importorskip("superfermion.bridge")
        u = np.kron(cirq.unitary(cirq.X), cirq.unitary(cirq.H)).astype(complex)
        c = sf.Circuit(2).x(1)
        c.unitary(u, [0, 1])
        back = bridge.from_cirq(bridge.to_cirq(c))
        np.testing.assert_allclose(
            np.asarray(sf.simulate(back).numpy()),
            np.asarray(sf.simulate(c).numpy()),
            atol=1e-9,
        )


class TestPennyLaneAdditiveOps:
    """from_pennylane previously raised on CY / PhaseShift / ISWAP; the added
    PL_MAP entries route them through the same generic path as CNOT."""

    def test_from_pennylane_supports_added_ops(self):
        try:
            import pennylane as qml
        except Exception:
            pytest.skip("pennylane not importable in this environment")
        bridge = pytest.importorskip("superfermion.bridge")
        make_qscript = getattr(qml.tape, "make_qscript", None)
        if make_qscript is None:
            pytest.skip("pennylane too old for qml.tape.make_qscript")

        def ansatz():
            qml.CY(wires=[0, 1])
            qml.PhaseShift(0.3, wires=0)
            qml.ISWAP(wires=[0, 1])

        tape = make_qscript(ansatz)()
        c = bridge.from_pennylane(tape, n_qubits=2)
        assert c.gate_count == 3
