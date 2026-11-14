"""Regression tests for the defect ledger verified on SuperFermion 0.1.29.

Each test is the repro from the ledger and asserts the corrected behavior:
on 0.1.29 the test fails (wrong result, rejection, or malformed export), on
the fixed build it passes.  SF-02b is not here: bare ``cs`` is invalid
OpenQASM 2 and Qiskit rejects it too, so there is no defect to fix.
"""
import warnings

import numpy as np
import pytest

import superfermion as sf
from superfermion import bridge as sf_bridge
from superfermion.bridge import from_pennylane, from_qasm, to_qasm, to_qiskit
from superfermion.serialization import from_qasm3, to_qasm3

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit, qasm2, qasm3  # noqa: E402
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def qref(text):
    """Qiskit's reading of the same OpenQASM 2 program."""
    return qasm2.loads(text, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)


def assert_equal_up_to_phase(a, b, atol=1e-8):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    assert a.shape == b.shape
    flat = np.argmax(np.abs(b))
    phase = a.ravel()[flat] / b.ravel()[flat]
    phase = phase / abs(phase)
    assert np.allclose(a, phase * b, atol=atol), "unitaries differ"


def assert_matches_qiskit(text):
    """from_qasm must agree with Qiskit on the same program."""
    sf_unitary = np.asarray(from_qasm(text).to_unitary())
    ref_unitary = Operator(qref(text)).data
    assert_equal_up_to_phase(sf_unitary, ref_unitary)


# ── SF-09 / SF-10a / SF-10b: register model ──────────────────────────────
def test_sf09_whole_register_operand_broadcasts():
    assert_matches_qiskit(HEADER + "qreg q[2];\nh q;")
    assert_matches_qiskit(HEADER + "qreg a[2];\nqreg b[2];\ncx a, b;")


def test_sf09_whole_register_measure():
    circuit = from_qasm(HEADER + "qreg q[2];\ncreg c[2];\nmeasure q -> c;")
    assert circuit.n_qubits == 2
    assert dict(circuit.count_ops())["MEASURE"] == 2
    result = sf.run(circuit, shots=50, seed=1)
    assert set(result.counts) <= {"00", "01", "10", "11"}


def test_sf10a_multiple_registers_keep_offsets():
    text = HEADER + "qreg a[1];\nqreg b[1];\nx b[0];"
    circuit = from_qasm(text)
    assert circuit.n_qubits == 2
    assert_matches_qiskit(text)


def test_sf10a_second_register_offset_upper_index():
    text = HEADER + "qreg a[2];\nqreg b[2];\nx b[1];"
    circuit = from_qasm(text)
    assert circuit.n_qubits == 4
    assert_matches_qiskit(text)


def test_sf10b_cross_register_two_qubit_gate():
    text = HEADER + "qreg a[2];\nqreg b[2];\ncreg c[4];\nh a[0];\ncx a[0], b[0];"
    circuit = from_qasm(text)
    assert circuit.n_qubits == 4
    assert_matches_qiskit(text)


def test_mixed_register_and_indexed_operands_are_rejected():
    with pytest.raises(ValueError, match="mixes whole-register"):
        from_qasm(HEADER + "qreg a[2];\nqreg b[2];\ncx a, b[0];")


# ── SF-08: conditional statements ───────────────────────────────────────
def test_sf08_if_statement_lowers_to_c_if():
    circuit = from_qasm(
        HEADER + "qreg q[1];\ncreg c[1];\nif(c==1) x q[0];"
    )
    gate = circuit._gates[0]
    assert gate.name == "X"
    assert gate.condition == (0, 1)


def test_sf08_if_statement_is_stored_with_condition():
    text = HEADER + "qreg q[1];\ncreg c[1];\nif(c==1) x q[0];"
    circuit = from_qasm(text)
    assert circuit._gates[0].condition == (0, 1)


# ── SF-12: PennyLane controlled rotations ───────────────────────────────
pennylane = pytest.importorskip("pennylane")


@pytest.mark.parametrize(
    "op",
    [
        pennylane.CRX(0.3, wires=[0, 1]),
        pennylane.CRY(0.3, wires=[0, 1]),
        pennylane.CRZ(0.3, wires=[0, 1]),
        pennylane.ControlledPhaseShift(0.3, wires=[0, 1]),
    ],
)
def test_sf12_pennylane_controlled_rotations(op):
    tape = pennylane.tape.QuantumScript([op], [pennylane.expval(pennylane.PauliZ(0))])
    circuit = from_pennylane(tape, 2)
    # ``Circuit.to_unitary`` uses the Qiskit layout (qubit 0 = least
    # significant), while ``qml.matrix`` treats the first wire as most
    # significant, so the reference is built with the wires reversed.
    assert_equal_up_to_phase(
        circuit.to_unitary(), pennylane.matrix(tape, wire_order=[1, 0])
    )


def test_sf12_pennylane_qnode_is_accepted():
    dev = pennylane.device("default.qubit", wires=1)

    @pennylane.qnode(dev)
    def circuit():
        pennylane.Hadamard(wires=0)
        return pennylane.expval(pennylane.PauliZ(0))

    converted = from_pennylane(circuit, 1)
    assert converted.count_ops().get("H") == 1


# ── SF-17: to_qiskit mappings ───────────────────────────────────────────
@pytest.mark.parametrize("gate", ["crx", "crz", "r1", "sxdg"])
def test_sf17_to_qiskit_supports_rotations(gate):
    circuit = sf.Circuit(2)
    reference = QuantumCircuit(2)
    if gate == "r1":
        circuit.r1(0.3, 0)
        reference.p(0.3, 0)
    elif gate == "sxdg":
        circuit.sxdg(0)
        reference.sxdg(0)
    else:
        getattr(circuit, gate)(0.3, 0, 1)
        getattr(reference, gate)(0.3, 0, 1)
    qc = to_qiskit(circuit)
    assert_equal_up_to_phase(Operator(qc).data, Operator(reference).data)


# ── SF-15 / SF-16: portable OpenQASM 3 ──────────────────────────────────
def test_sf15_qasm3_includes_stdgates():
    qasm = to_qasm3(sf.Circuit(2).h(0).cx(0, 1))
    assert 'include "stdgates.inc";' in qasm
    assert qasm.index('include "stdgates.inc";') < qasm.index("qubit[2]")


def test_sf16_qasm3_uses_portable_names():
    circuit = (
        sf.Circuit(2)
        .u(0.1, 0.2, 0.3, 0)
        .r1(0.2, 1)
        .iswap(0, 1)
        .ecr(0, 1)
        .rxx(0.2, 0, 1)
        .ryy(0.3, 0, 1)
        .rzz(0.4, 0, 1)
        .sxdg(0)
    )
    qasm = to_qasm3(circuit)
    assert "u(" not in qasm
    assert "r1(" not in qasm
    assert "gate iswap" in qasm
    assert "gate rzz" in qasm
    parsed = qasm3.loads(qasm)
    assert_equal_up_to_phase(Operator(parsed).data, np.asarray(circuit.to_unitary()))


# ── SF-20: gate noise is applied on the default method ──────────────────
def test_sf20_gate_noise_applied_by_default():
    noise = sf.NoiseModel().add_depolarizing(0.5)
    circuit = sf.Circuit(1).x(0)
    result = sf.run(circuit, noise_model=noise, shots=400, seed=3)
    assert set(result.counts) == {"0", "1"}


def test_sf20_explicit_statevector_still_warns():
    noise = sf.NoiseModel().add_depolarizing(0.5)
    circuit = sf.Circuit(1).x(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sf.run(circuit, method="statevector", noise_model=noise, shots=50)
    assert any("gate noise" in str(w.message) for w in caught)
    assert result.counts == {"1": 50}


# ── SF-13 / SF-14: index coercion and error reporting ───────────────────
def test_sf13_numpy_integer_indices_are_accepted():
    for index in (np.int64(0), np.int32(0)):
        circuit = sf.Circuit(1)
        circuit.x(index)
        circuit.measure(index, index)
        assert circuit.count_ops()["X"] == 1


def test_sf14_native_parser_error_is_reported_first():
    text = HEADER + "qreg q[5];\nc3x q[0],q[1],q[2];"
    with pytest.raises(ValueError) as caught:
        from_qasm(text)
    message = str(caught.value)
    assert message.index("expects") < message.index("fallback")
    assert "c3x" in message


# ── Residual gaps closed in 0.1.30 ───────────────────────────────────────
def test_declaration_only_program_is_an_empty_circuit():
    circuit = from_qasm(HEADER + "qreg q[2];")
    assert circuit.n_qubits == 2
    assert circuit.count_ops() == {}

    fallback = sf_bridge._parse_qasm2_python(HEADER + "qreg q[2];")
    assert fallback.n_qubits == 2
    assert fallback.count_ops() == {}


def test_measure_into_explicit_classical_bit_keeps_mapping():
    text = HEADER + "qreg a[2];\ncreg c[2];\nx a[1];\nmeasure a[1] -> c[0];"
    circuit = from_qasm(text)
    measure = [g for g in circuit._gates if g.name == "MEASURE"][0]
    assert measure.qubits == [1]
    assert measure.classical_bits == [0]

    api = sf.Circuit(2, n_cbits=2).x(1).measure(1, 0)
    assert circuit.to_gate_list() == api.to_gate_list()
    assert (
        sf.run(circuit, shots=20, seed=5).counts
        == sf.run(api, shots=20, seed=5).counts
    )


def test_to_qiskit_exports_condition_as_if_test():
    circuit = sf.Circuit(1, n_cbits=1).h(0).measure(0, 0).c_if(0, 1).x(0)
    qc = to_qiskit(circuit)
    operation = qc.data[-1].operation
    assert type(operation).__name__ == "IfElseOp"
    assert operation.condition == (qc.clbits[0], 1)


def test_to_qasm_rejects_conditions_with_a_clear_error():
    # OpenQASM 2's `if(c==value)` names a whole classical register, so a
    # per-bit c_if is only expressible on a single-cbit circuit (covered by
    # test_to_qasm_conditional_measure_loads_in_qiskit).  Wider registers
    # must raise instead of emitting a wrong/register-wide guard.
    circuit = sf.Circuit(1, n_cbits=2).measure(0, 0).c_if(0, 1).x(0)
    with pytest.raises(ValueError, match="to_qasm3"):
        to_qasm(circuit)


def test_from_qiskit_keeps_measurement_clbit_mapping():
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    qc.measure(1, 0)
    circuit = sf_bridge.from_qiskit(qc)
    measure = [g for g in circuit._gates if g.name == "MEASURE"][0]
    assert measure.qubits == [1]
    assert measure.classical_bits == [0]


def test_to_qiskit_if_test_roundtrips_through_from_qiskit():
    circuit = sf.Circuit(1, n_cbits=1).measure(0, 0).c_if(0, 1).x(0)
    restored = sf_bridge.from_qiskit(to_qiskit(circuit))
    guarded = [g for g in restored._gates if g.name == "X"][0]
    assert guarded.condition == (0, 1)


@pytest.mark.parametrize(
    ("exporter_name", "module"),
    [
        ("to_braket", "braket.circuits"),
        ("to_cirq", "cirq"),
        ("to_ionq", None),
        ("to_pennylane", "pennylane"),
        ("to_stim", "stim"),
    ],
)
def test_exporters_reject_conditions_they_cannot_represent(
    exporter_name, module
):
    if module is not None:
        pytest.importorskip(module)
    circuit = sf.Circuit(2, n_cbits=1).h(0).measure(0, 0).c_if(0, 1).x(1)
    with pytest.raises(ValueError, match="to_qasm3"):
        getattr(sf_bridge, exporter_name)(circuit)


# ── Parser parity and QASM 3 feed-forward (final audit) ──────────────────
PARITY_PROGRAMS = {
    "basic": HEADER + "qreg q[3];\nh q[0];\ncx q[0], q[1];",
    "broadcast": HEADER + "qreg q[2];\nh q;",
    "barrier": HEADER + "qreg q[3];\nbarrier q;\nh q[2];",
    "legacy-u0": HEADER + "qreg q[2];\nu2(0.1, 0.2) q[0];\nu0(0.3) q[1];",
    "measure-cross": HEADER + "qreg a[2];\ncreg c[2];\nmeasure a[1] -> c[0];",
    "if-indexed": HEADER + "qreg q[1];\ncreg c[1];\nif(c[0]==0) rz(pi/4) q[0];",
    "reset-cond": HEADER + "qreg q[2];\ncreg c[1];\nif(c==1) reset q[0];",
}


def _normalized_records(circuit):
    circuit._ensure_gates()
    return [
        (
            gate.name,
            tuple(gate.qubits),
            tuple(round(float(p), 12) for p in gate.params),
            tuple(gate.classical_bits),
            gate.condition,
        )
        for gate in circuit._gates
    ]


@pytest.mark.parametrize("program", PARITY_PROGRAMS.values(), ids=PARITY_PROGRAMS)
def test_native_and_python_qasm2_parsers_agree(program):
    """Native and fallback parsers must produce identical records.

    Locks in the CNOT name, explicit-barrier retention, u0 -> U(0, 0, 0)
    reduction and condition/classical-bit mapping.
    """
    native = from_qasm(program)
    reason = sf_bridge._RUST_QASM_STATE.reason
    if reason is not None and reason.startswith(("ImportError", "TypeError")):
        pytest.skip(f"native parser unavailable: {reason}")
    assert reason is None, f"native parser rejected a supported program: {reason}"
    fallback = sf_bridge._parse_qasm2_python(program)
    assert _normalized_records(native) == _normalized_records(fallback)


def test_to_qasm3_condition_uses_boolean_literal_and_loads_in_qiskit():
    circuit = sf.Circuit(1, n_cbits=1).measure(0, 0).c_if(0, 1).x(0)
    text = to_qasm3(circuit)
    assert "if (c[0] == true) x q[0];" in text
    assert "== 1" not in text
    qc = qasm3.loads(text)
    operation = qc.data[-1].operation
    assert type(operation).__name__ == "IfElseOp"
    assert operation.condition == (qc.clbits[0], True)


def test_to_qasm3_false_condition_round_trips_through_from_qasm3():
    circuit = sf.Circuit(1, n_cbits=1).measure(0, 0).c_if(0, 0).x(0)
    text = to_qasm3(circuit)
    assert "if (c[0] == false) x q[0];" in text
    restored = from_qasm3(text)
    guarded = [g for g in restored._gates if g.name == "X"][0]
    assert guarded.condition == (0, 0)


def test_from_qasm3_still_accepts_numeric_condition_spelling():
    text = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nbit[1] c;\n'
        "c[0] = measure q[0];\nif (c[0] == 1) x q[0];\n"
    )
    restored = from_qasm3(text)
    guarded = [g for g in restored._gates if g.name == "X"][-1]
    assert guarded.condition == (0, 1)


def test_qasm2_parsers_reject_conditional_barrier():
    """OpenQASM 2 allows ``if`` only on gates, measurements and resets.

    The native parser always rejected this; the fallback used to accept it and
    silently drop the condition.  Qiskit's QASM2 importer rejects it too.
    """
    text = HEADER + "qreg q[2];\ncreg c[1];\nif(c==1) barrier q;"
    with pytest.raises(ValueError, match="barrier cannot be conditioned"):
        from_qasm(text)
    with pytest.raises(ValueError, match="barrier cannot be conditioned"):
        sf_bridge._parse_qasm2_python(text)


def test_to_qasm3_guards_barrier_and_round_trips():
    circuit = sf.Circuit(2, n_cbits=1).c_if(0, 1).barrier(0, 1)
    text = to_qasm3(circuit)
    assert "if (c[0] == true) barrier q[0], q[1];" in text
    qc = qasm3.loads(text)
    operation = qc.data[-1].operation
    assert type(operation).__name__ == "IfElseOp"
    assert operation.condition == (qc.clbits[0], True)
    restored = from_qasm3(text)
    barrier = [g for g in restored._gates if g.name == "BARRIER"][0]
    assert barrier.condition == (0, 1)


# ── Conditional measurement (final limitation sweep) ──────────────────────
def test_measure_accepts_pending_condition_and_records_it():
    circuit = sf.Circuit(2, n_cbits=2)
    circuit.x(1).h(0).measure(0, 0).c_if(0, 1).measure(1, 1)
    measure = circuit._gates[-1]
    assert measure.name == "MEASURE"
    assert measure.classical_bits == [1]
    assert measure.condition == (0, 1)


def test_conditional_measure_simulation_semantics():
    """q2 = c1, and c1 is only measured when c0 == 1: P(q2=1) = 1/4.

    0.0 would mean the guarded measurement never ran, 0.5 that it ran
    unconditionally, so the ratio pins down the intended semantics.
    """
    circuit = sf.Circuit(3, n_cbits=2)
    circuit.h(0).h(1).measure(0, 0)
    circuit.c_if(0, 1).measure(1, 1)
    circuit.c_if(1, 1).x(2)
    result = sf.run(circuit, shots=4000, seed=11)
    total = sum(result.counts.values())
    ratio = sum(v for k, v in result.counts.items() if k[0] == "1") / total
    assert 0.2 < ratio < 0.3
    assert all(k[2] == "1" for k in result.counts if k[0] == "1")


COND_MEASURE_QASM2 = HEADER + (
    "qreg q[2];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"
    "if(c==1) measure q[1] -> c[0];"
)


def test_qasm2_parsers_agree_on_conditional_measure():
    native = from_qasm(COND_MEASURE_QASM2)
    reason = sf_bridge._RUST_QASM_STATE.reason
    if reason is not None and reason.startswith(("ImportError", "TypeError")):
        pytest.skip(f"native parser unavailable: {reason}")
    assert reason is None
    fallback = sf_bridge._parse_qasm2_python(COND_MEASURE_QASM2)
    assert _normalized_records(native) == _normalized_records(fallback)
    measure = [g for g in native._gates if g.name == "MEASURE"][-1]
    assert measure.condition == (0, 1)
    assert measure.classical_bits == [0]


def test_qasm2_conditional_measure_loads_in_qiskit():
    qc = qref(COND_MEASURE_QASM2)
    operation = qc.data[-1].operation
    assert type(operation).__name__ == "IfElseOp"
    # QASM2's whole-register spelling maps back to the 1-bit register c.
    assert operation.condition == (qc.cregs[0], 1)


def test_to_qasm_conditional_measure_loads_in_qiskit():
    circuit = sf.Circuit(2, n_cbits=1)
    circuit.h(0).measure(0, 0).c_if(0, 1).measure(1, 0)
    text = to_qasm(circuit)
    assert "if(c==1) measure q[1] -> c[0];" in text
    qref(text)
    again = from_qasm(text)
    measure = [g for g in again._gates if g.name == "MEASURE"][-1]
    assert measure.condition == (0, 1)
    assert measure.classical_bits == [0]


def test_to_qasm_emits_barriers_instead_of_dropping_them():
    circuit = sf.Circuit(2).h(0).barrier(0, 1).x(1)
    text = to_qasm(circuit)
    assert "barrier q[0], q[1];" in text
    qref(text)  # Qiskit accepts the emitted program


def test_to_qasm_rejects_conditional_barrier():
    circuit = sf.Circuit(2, n_cbits=1).measure(0, 0).c_if(0, 1).barrier(0, 1)
    with pytest.raises(ValueError, match="conditional barrier"):
        to_qasm(circuit)


def test_to_qasm_conditional_reset_loads_in_qiskit():
    circuit = from_qasm(HEADER + "qreg q[2];\ncreg c[1];\nif(c==1) reset q[0];")
    emitted = to_qasm(circuit)
    assert "if(c==1) reset q[0];" in emitted
    qref(emitted)


def test_from_qiskit_multi_instruction_if_test():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], True)):
        qc.measure(1, 1)
        qc.x(1)
    circuit = sf_bridge.from_qiskit(qc)
    guarded = [g for g in circuit._gates if g.condition == (0, 1)]
    assert [g.name for g in guarded] == ["MEASURE", "X"]
    assert guarded[0].classical_bits == [1]


def test_from_qiskit_if_else_inverts_the_else_guard():
    qc = QuantumCircuit(2, 1)
    with qc.if_test((qc.clbits[0], True)) as else_block:
        qc.x(0)
    with else_block:
        qc.x(1)
    circuit = sf_bridge.from_qiskit(qc)
    assert [(g.name, g.qubits, g.condition) for g in circuit._gates] == [
        ("X", [0], (0, 1)),
        ("X", [1], (0, 0)),
    ]


def test_from_qiskit_rejects_measure_into_own_condition_bit():
    qc = QuantumCircuit(1, 1)
    with qc.if_test((qc.clbits[0], True)):
        qc.measure(0, 0)
    with pytest.raises(ValueError, match="own condition bit"):
        sf_bridge.from_qiskit(qc)


def test_from_qasm3_parses_qiskit_brace_block_and_bare_condition():
    """Qiskit's QASM3 export writes `if (c[0]) { ... }` (bare + braces)."""
    text = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[1] c;\nqubit[2] q;\n'
        "c[0] = measure q[0];\nif (c[0]) {\n  x q[1];\n  z q[1];\n}\n"
    )
    circuit = from_qasm3(text)
    guarded = [g for g in circuit._gates if g.condition == (0, 1)]
    assert [g.name for g in guarded] == ["X", "Z"]


def test_from_qasm3_parses_else_block():
    text = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[1] c;\nqubit[2] q;\n'
        "if (c[0]) {\n  x q[0];\n} else {\n  x q[1];\n}\n"
    )
    circuit = from_qasm3(text)
    assert [(g.name, g.qubits, g.condition) for g in circuit._gates] == [
        ("X", [0], (0, 1)),
        ("X", [1], (0, 0)),
    ]


def test_from_qasm3_rejects_nested_blocks():
    text = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[2] c;\nqubit[1] q;\n'
        "if (c[0]) {\n  if (c[1]) {\n    x q[0];\n  }\n}\n"
    )
    with pytest.raises(ValueError, match="Nested"):
        from_qasm3(text)


def test_to_qasm3_guarded_measure_round_trips():
    circuit = sf.Circuit(2, n_cbits=2)
    circuit.measure(0, 0).c_if(0, 1).measure(1, 1)
    text = to_qasm3(circuit)
    assert "if (c[0] == true) c[1] = measure q[1];" in text
    qc = qasm3.loads(text)
    assert type(qc.data[-1].operation).__name__ == "IfElseOp"
    restored = from_qasm3(text)
    measure = [g for g in restored._gates if g.name == "MEASURE"][-1]
    assert measure.condition == (0, 1)
    assert measure.classical_bits == [1]


# ── Guard preservation in simulation prep and compilation ────────────────
DECOMP_GUARD_QASM2 = HEADER + (
    "qreg q[3];\ncreg c[1];\n"
    "h q[2];\nmeasure q[2] -> c[0];\n"
    "if(c==1) x q[0];\n"
    "cry(0.3) q[1], q[2];\n"
)


def test_simulation_prep_preserves_guards_when_decomposing():
    """A CRY record forces decompose+fuse before dynamic simulation.

    A conditioned gate in the same circuit used to lose its c_if guard:
    single-qubit fusion folded the guarded X into an unconditional
    U(theta, phi, lambda).  q0 must stay equal to the measured q2.
    """
    from superfermion.circuit import GateRecord

    circuit = sf.Circuit(3, n_cbits=1).h(2).measure(2, 0)
    circuit.extend_raw(
        [
            GateRecord("X", [0], [], condition=(0, 1)),
            GateRecord("CRY", [1, 2], [0.3]),
        ]
    )
    result = sf.run(circuit, shots=2000, seed=7)
    assert set(result.counts) <= {"000", "101"}


def test_fusion_keeps_conditional_gates_verbatim():
    from superfermion.backends.turbo import decompose_for_rust, fuse_all_gates

    circuit = from_qasm(DECOMP_GUARD_QASM2)
    holder = sf.Circuit(circuit.n_qubits, n_cbits=circuit.n_cbits)
    holder._gates = decompose_for_rust(circuit._gates)
    fused = fuse_all_gates(holder)
    guarded = [g for g in fused._gates if g.condition == (0, 1)]
    assert [g.name for g in guarded] == ["X"]
    assert fused.n_cbits == circuit.n_cbits


def test_decomposed_conditional_gate_guards_every_subgate():
    from superfermion.backends.turbo import decompose_for_rust
    from superfermion.circuit import GateRecord

    circuit = from_qasm(DECOMP_GUARD_QASM2)
    records = decompose_for_rust(circuit._gates)
    ry_cnot = [g for g in records if g.name in ("RY", "CNOT")]
    assert ry_cnot and all(g.condition is None for g in ry_cnot)
    # A conditioned cry must carry the guard onto every factor instead.
    held = sf.Circuit(2, n_cbits=2).measure(0, 0)
    held._gates.append(GateRecord("CRY", [0, 1], [0.3], condition=(0, 1)))
    parts = [g for g in decompose_for_rust(held._gates) if g.name != "MEASURE"]
    assert parts and all(g.condition == (0, 1) for g in parts)


def test_prepare_dag_keeps_classical_width_beyond_n_qubits():
    program = HEADER + (
        "qreg q[2];\ncreg c[3];\nu1(0.5) q[0];\nmeasure q[1] -> c[2];\n"
    )
    circuit = from_qasm(program)
    result = sf.run(circuit, shots=20, seed=1)
    assert sum(result.counts.values()) == 20


@pytest.mark.parametrize(
    "kwargs", [{"level": 1}, {"level": 2}, {"target": "ibm_eagle"}]
)
def test_compile_rejects_classical_feed_forward(kwargs):
    """The Rust optimizer has no condition concept and silently unguards."""
    circuit = sf.Circuit(2, n_cbits=1).h(0).measure(0, 0).c_if(0, 1).x(1)
    with pytest.raises(ValueError, match="feed-forward"):
        sf.compile(circuit, **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"level": 1}, {"level": 2}, {"target": "ibm_eagle"}]
)
def test_compile_rejects_reset_and_mid_circuit_measure(kwargs):
    """RESET becomes an X-like U and a non-terminal measure is re-ordered."""
    reset = sf.Circuit(1).h(0).reset(0)
    with pytest.raises(ValueError, match="RESET"):
        sf.compile(reset, **kwargs)

    mid = sf.Circuit(2, n_cbits=1).h(0).measure(0, 0).x(0)
    with pytest.raises(ValueError, match="mid-circuit"):
        sf.compile(mid, **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"level": 1}, {"level": 2}, {"target": "ibm_eagle"}]
)
def test_compile_keeps_terminal_measure_mapping(kwargs):
    """A trailing measurement block may compile; wire -> cbit must survive."""
    circuit = sf.Circuit(2, n_cbits=2).x(1).measure(1, 0).measure(0, 1)
    compiled = sf.compile(circuit, **kwargs)
    mapping = {
        g.qubits[0]: g.classical_bits[0]
        for g in compiled._gates
        if g.name == "MEASURE"
    }
    assert mapping == {1: 0, 0: 1}
    assert sf.run(circuit, shots=400, seed=4).counts == sf.run(
        compiled, shots=400, seed=4
    ).counts


def test_basis_translation_keeps_guard_on_decomposed_gate():
    from superfermion.compiler.passes import BasisTranslationPass

    circuit = sf.Circuit(1, n_cbits=1).measure(0, 0).c_if(0, 1).x(0)
    out = BasisTranslationPass(["RZ", "RX"]).run(circuit)
    emitted = [g for g in out._gates if g.name != "MEASURE"]
    assert emitted and all(g.condition == (0, 1) for g in emitted)


def test_json_and_sfc_roundtrips_keep_classical_fields():
    """Both serializers rebuilt records from name/qubits/params only."""
    from superfermion.serialization.circuit_format import (
        circuit_from_bytes,
        circuit_to_bytes,
    )

    circuit = (
        sf.Circuit(2, n_cbits=2)
        .h(1)
        .measure(1, 0)
        .c_if(0, 1)
        .x(0)
        .measure(0, 1)
    )
    expected = [(1, 0, None), (0, None, (0, 1)), (0, 1, None)]
    for restored in (
        sf.Circuit.from_json(circuit.to_json()),
        circuit_from_bytes(circuit_to_bytes(circuit)),
    ):
        assert [
            (
                g.qubits[0],
                g.classical_bits[0] if g.classical_bits else None,
                g.condition,
            )
            for g in restored._gates
            if g.name in ("MEASURE", "X")
        ] == expected
        assert sf.run(restored, shots=400, seed=3).counts == sf.run(
            circuit, shots=400, seed=3
        ).counts


def test_draw_shows_measurement_target_and_guard():
    text = (
        sf.Circuit(2, n_cbits=1)
        .h(0)
        .measure(0, 0)
        .c_if(0, 1)
        .x(1)
        .draw(ascii_only=True)
    )
    assert "MEASURE->c0" in text
    assert "X if c0=1" in text


def test_twirl_passes_conditioned_gate_through():
    from superfermion.circuit import GateRecord
    from superfermion.compiler.advanced import PauliTwirlingPass

    circuit = sf.Circuit(2, n_cbits=1).measure(0, 0)
    circuit._gates.append(GateRecord("CNOT", [0, 1], [], condition=(0, 1)))
    out = PauliTwirlingPass(seed=1).run(circuit)
    assert [g.name for g in out._gates] == ["MEASURE", "CNOT"]
    assert out._gates[-1].condition == (0, 1)


def test_unitary_decomposition_guards_every_factor():
    import numpy as np

    from superfermion.circuit import GateRecord
    from superfermion.compiler.passes import UnitaryDecompositionPass

    circuit = sf.Circuit(1, n_cbits=1).measure(0, 0)
    circuit._gates.append(
        GateRecord(
            "UNITARY",
            [0],
            [],
            condition=(0, 1),
            matrix=np.array([[0, 1], [1, 0]], dtype=complex),
        )
    )
    out = UnitaryDecompositionPass().run(circuit)
    parts = [g for g in out._gates if g.name != "MEASURE"]
    assert parts and all(g.condition == (0, 1) for g in parts)


def test_folding_keeps_guard_on_every_copy():
    from superfermion.mitigation import _fold_circuit

    circuit = sf.Circuit(1, n_cbits=1).measure(0, 0).c_if(0, 1).x(0)
    folded = _fold_circuit(circuit, 3)
    xs = [g for g in folded._gates if g.name == "X"]
    assert len(xs) == 5
    assert all(g.condition == (0, 1) for g in xs)


def test_to_unitary_refuses_dynamic_circuits():
    """The Rust matrix builder skipped MEASURE/RESET and unguarded c_if gates."""
    with pytest.raises(ValueError, match="dynamic circuit"):
        sf.Circuit(2, n_cbits=1).h(0).measure(0, 0).to_unitary()
    with pytest.raises(ValueError, match="dynamic circuit"):
        sf.Circuit(1).h(0).reset(0).to_unitary()
    with pytest.raises(ValueError, match="conditioned"):
        sf.Circuit(1, n_cbits=1).measure(0, 0).c_if(0, 1).x(0).to_unitary()
    # A barrier is a unitary no-op and must stay accepted.
    assert sf.Circuit(1).h(0).barrier(0).to_unitary().shape == (2, 2)


def test_json_rejects_out_of_range_serialized_cbit():
    import json

    payload = json.loads(sf.Circuit(1, n_cbits=1).measure(0, 0).to_json())
    payload["gates"][0]["classical_bits"] = [7]
    with pytest.raises(ValueError, match="out of range"):
        sf.Circuit.from_json(json.dumps(payload))
