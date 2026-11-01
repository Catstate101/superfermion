"""OpenQASM 3.0 round-trip tests for ``to_qasm3()`` / ``from_qasm3()``.

``from_qasm3`` used to dispatch through a hard-coded gate table that did not
contain ``r1``, ``cp``, ``cu``/``cu3``, ``crx``, ``cry``, ``crz`` or ``u``/``u3``.
Unknown gates were skipped with a bare ``continue``, so a round-trip returned a
circuit with the right qubit count and *missing operations* — a wrong answer, not
an error. These tests pin the fixed behaviour and the strict contract:
anything the parser cannot represent must raise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import superfermion as sf
from superfermion.bridge import from_qasm, to_qasm
from superfermion.serialization import from_qasm3, verify_qasm3_roundtrip

pytestmark = pytest.mark.domain


def state_of(circuit):
    return np.asarray(sf.run(circuit, shots=0).statevector, dtype=complex)


# ── Gates that were silently dropped ─────────────────────────────────────────

DROPPED_BEFORE = {
    "r1": lambda: sf.Circuit(2).h(0).r1(0.5, 0),
    "cp": lambda: sf.Circuit(2).h(0).h(1).cp(0.7, 0, 1),
    "cu": lambda: sf.Circuit(2).h(0).cu(0.4, 0.3, 0.2, 0, 1),
    "cu_gamma": lambda: sf.Circuit(2).h(0).cu(0.4, 0.3, 0.2, 0, 1, 0.7),
    "crx": lambda: sf.Circuit(2).h(0).crx(0.3, 0, 1),
    "cry": lambda: sf.Circuit(2).h(0).cry(0.3, 0, 1),
    "crz": lambda: sf.Circuit(2).h(0).crz(0.2, 0, 1),
    "u": lambda: sf.Circuit(2).h(0).u(0.4, 0.3, 0.2, 1),
    "u3": lambda: sf.Circuit(2).h(0).u3(0.4, 0.3, 0.2, 1),
}


@pytest.mark.parametrize("name", sorted(DROPPED_BEFORE))
def test_formerly_dropped_gates_survive_round_trip(name):
    circuit = DROPPED_BEFORE[name]()
    restored = from_qasm3(circuit.to_qasm3())

    assert len(restored._ensure_gates() if restored._gates is None else restored._gates) == len(
        circuit._ensure_gates() if circuit._gates is None else circuit._gates
    ), f"{name}: gate count changed, a gate was dropped"
    assert np.allclose(state_of(circuit), state_of(restored), atol=1e-12), (
        f"{name}: statevector differs after round-trip"
    )


def test_cu_gamma_order_is_not_swapped():
    """QASM writes ``cu(theta, phi, lam, gamma)``; gamma follows the qubits in Python."""
    circuit = sf.Circuit(2).h(0).cu(0.4, 0.3, 0.2, 0, 1, 0.7)
    restored = from_qasm3(circuit.to_qasm3())

    # gamma lands on the gamma slot, not on a qubit index or a rotation angle.
    original = circuit._ensure_gates() if circuit._gates is None else circuit._gates
    reloaded = restored._ensure_gates() if restored._gates is None else restored._gates
    assert list(reloaded[-1].params) == pytest.approx(list(original[-1].params))
    assert list(reloaded[-1].qubits) == [0, 1]
    assert np.allclose(state_of(circuit), state_of(restored), atol=1e-12)


# ── The rest of the writer's gate set ────────────────────────────────────────

UNITARY_CASES = {
    "h": lambda: sf.Circuit(1).h(0),
    "x_y_z": lambda: sf.Circuit(1).x(0).y(0).z(0),
    "s_t": lambda: sf.Circuit(1).s(0).t(0).sdg(0).tdg(0).sx(0).id(0),
    "cnot": lambda: sf.Circuit(2).h(0).cnot(0, 1),
    "cz_cy": lambda: sf.Circuit(2).h(0).cz(0, 1).cy(0, 1),
    "swap_iswap_ecr": lambda: sf.Circuit(2).h(0).swap(0, 1).iswap(0, 1).ecr(0, 1),
    "ccx": lambda: sf.Circuit(3).h(0).h(1).ccx(0, 1, 2),
    "cswap": lambda: sf.Circuit(3).x(0).cswap(0, 1, 2),
    "rot1": lambda: sf.Circuit(2).rx(0.3, 0).ry(0.4, 0).rz(0.5, 0).p(0.6, 1),
    "rot2": lambda: sf.Circuit(2).h(0).rxx(0.4, 0, 1).ryy(0.1, 0, 1).rzz(0.2, 0, 1),
}


@pytest.mark.parametrize("name", sorted(UNITARY_CASES))
def test_writer_gate_set_round_trips(name):
    circuit = UNITARY_CASES[name]()
    restored = from_qasm3(circuit.to_qasm3())
    assert np.allclose(state_of(circuit), state_of(restored), atol=1e-12)


def test_qasm2_spellings_u1_u2_map_to_p_and_u3():
    """``u1``/``u2`` appear in hand-written QASM; they had no mapping at all."""
    text = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
u1(0.7) q[0];
u2(0.3, 0.4) q[0];
"""
    restored = from_qasm3(text)
    # u1(λ) == p(λ); u2(φ, λ) == u3(π/2, φ, λ)
    expected = sf.Circuit(1, 1).h(0).p(0.7, 0).u3(math.pi / 2, 0.3, 0.4, 0)
    assert np.allclose(state_of(restored), state_of(expected), atol=1e-12)


def test_measure_and_condition_round_trip_text():
    """Feed-forward survives: ``if (c[0] == 1) x q[1]`` comes back identical."""
    circuit = sf.Circuit(2, 2).h(0).measure(0, 0).c_if(0, 1).x(1)
    text = circuit.to_qasm3()
    assert "if (c[0] == 1) x q[1];" in text

    restored = from_qasm3(text)
    assert restored.to_qasm3().strip() == text.strip()
    # The condition must not leak onto the next gate either.
    assert restored._pending_condition is None


# ── Strictness contract ─────────────────────────────────────────────────────


def test_unknown_gate_raises_instead_of_being_dropped():
    qasm = """OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
definitely_not_a_gate q[0];
"""
    with pytest.raises(ValueError) as excinfo:
        from_qasm3(qasm)
    message = str(excinfo.value)
    assert "definitely_not_a_gate" in message
    assert "strict=False" in message

    # The escape hatch still works for callers who know what they are skipping.
    lenient = from_qasm3(qasm, strict=False)
    assert len(lenient._ensure_gates() if lenient._gates is None else lenient._gates) == 1


def test_qubit_count_matches_declaration():
    for n in (1, 2, 5):
        circuit = sf.Circuit(n).h(0)
        assert from_qasm3(circuit.to_qasm3()).n_qubits == n


# ── The two exporters agree ─────────────────────────────────────────────────


def test_bridge_from_qasm_accepts_both_qasm_versions():
    """``bridge.from_qasm`` documents QASM 2.0/3.0, so it must read both.

    The QASM 2 importer used to choke on the ``bit[2] c;`` line of its own
    QASM 3 output. QASM 3 text is now routed to ``from_qasm3`` when the native
    parser cannot read it.
    """
    circuit = sf.Circuit(2).h(0).cnot(0, 1).rx(0.3, 1)
    from_qasm2_text = from_qasm(to_qasm(circuit))
    from_qasm3_text = from_qasm(circuit.to_qasm3())

    assert np.allclose(state_of(circuit), state_of(from_qasm2_text), atol=1e-12)
    assert np.allclose(state_of(circuit), state_of(from_qasm3_text), atol=1e-12)


def test_bridge_from_qasm_reads_qasm3_registers_and_measurement():
    qasm = """OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cnot q[0], q[1];
c[0] = measure q[0];
"""
    circuit = from_qasm(qasm)

    assert circuit.n_qubits == 2
    assert circuit.n_cbits == 2
    assert int(circuit.gate_count) == 3
    assert np.allclose(
        state_of(circuit), np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), atol=1e-12
    )


def test_verify_qasm3_roundtrip_reports_structural_equality():
    circuit = sf.Circuit(2).h(0).crx(0.3, 0, 1).cp(0.2, 0, 1)
    assert verify_qasm3_roundtrip(circuit) is True
