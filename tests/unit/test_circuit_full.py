"""Unit tests for Circuit gate methods not covered in test_circuit.py."""

import math

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.gates.matrices import gate_unitary_matrix


pytestmark = pytest.mark.unit


def _run_statevector(circuit: Circuit) -> np.ndarray:
    result = sf.run(circuit, device="cpu", shots=0)
    sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
    assert sv is not None
    np.testing.assert_allclose(np.sum(np.abs(sv) ** 2), 1.0, atol=1e-10)
    return sv


def _peak_index(sv: np.ndarray) -> int:
    return int(np.argmax(np.abs(sv)))


class TestSingleQubitGates:
    def test_sdg_cancels_s(self):
        sv = _run_statevector(Circuit(1).s(0).sdg(0))
        np.testing.assert_allclose(abs(sv[0]), 1.0, atol=1e-10)

    def test_tdg_cancels_t(self):
        sv = _run_statevector(Circuit(1).t(0).tdg(0))
        np.testing.assert_allclose(abs(sv[0]), 1.0, atol=1e-10)

    def test_sx_squared_is_x(self):
        sv = _run_statevector(Circuit(1).sx(0).sx(0))
        np.testing.assert_allclose(abs(sv[1]), 1.0, atol=1e-10)

    def test_id_is_noop(self):
        sv = _run_statevector(Circuit(1).h(0).id(0))
        np.testing.assert_allclose(abs(sv[0]), 1 / math.sqrt(2), atol=1e-10)
        np.testing.assert_allclose(abs(sv[1]), 1 / math.sqrt(2), atol=1e-10)

    def test_p_zero_is_identity(self):
        sv = _run_statevector(Circuit(1).p(0.0, 0))
        np.testing.assert_allclose(abs(sv[0]), 1.0, atol=1e-10)

    def test_p_pi_on_one_state(self):
        sv = _run_statevector(Circuit(1).x(0).p(math.pi, 0))
        np.testing.assert_allclose(abs(sv[1]), 1.0, atol=1e-10)

    def test_u_zero_params_is_identity(self):
        sv = _run_statevector(Circuit(1).u(0.0, 0.0, 0.0, 0))
        np.testing.assert_allclose(abs(sv[0]), 1.0, atol=1e-10)

    def test_u3_delegates_to_u(self):
        qc_u = Circuit(1).u(math.pi / 2, 0.0, math.pi, 0)
        qc_u3 = Circuit(1).u3(math.pi / 2, 0.0, math.pi, 0)
        sv_u = _run_statevector(qc_u)
        sv_u3 = _run_statevector(qc_u3)
        np.testing.assert_allclose(sv_u, sv_u3, atol=1e-10)

    def test_u_pi_rotation_to_one(self):
        sv = _run_statevector(Circuit(1).u(math.pi, 0.0, math.pi, 0))
        np.testing.assert_allclose(abs(sv[1]), 1.0, atol=1e-10)


class TestParameterizedTwoQubitGates:
    def test_cp_leaves_non_controlled_basis_unchanged(self):
        # X(1) → |10⟩ (index 2). CP(π, control=0, target=1) with control=0 → no phase.
        sv = _run_statevector(Circuit(2).x(1).cp(math.pi, 0, 1))
        assert _peak_index(sv) == 2
        np.testing.assert_allclose(abs(sv[2]), 1.0, atol=1e-10)

    def test_cp_applies_phase_on_both_ones(self):
        sv = _run_statevector(Circuit(2).x(0).x(1).cp(math.pi, 0, 1))
        assert _peak_index(sv) == 3
        np.testing.assert_allclose(abs(sv[3]), 1.0, atol=1e-10)

    def test_cy_flips_target_when_control_is_one(self):
        sv = _run_statevector(Circuit(2).x(0).cy(0, 1))
        assert _peak_index(sv) == 3
        np.testing.assert_allclose(abs(sv[3]), 1.0, atol=1e-10)

    def test_iswap_exchanges_amplitudes(self):
        # X(0) → |01⟩ (index 1). iSWAP swaps |01⟩ → i|10⟩ (index 2).
        sv = _run_statevector(Circuit(2).x(0).iswap(0, 1))
        assert _peak_index(sv) == 2
        np.testing.assert_allclose(abs(sv[2]), 1.0, atol=1e-10)

    def test_ecr_entangles_qubits(self):
        sv = _run_statevector(Circuit(2).ecr(0, 1))
        assert np.count_nonzero(np.abs(sv) > 1e-10) > 1

    def test_rzz_zero_is_identity(self):
        sv = _run_statevector(Circuit(2).rzz(0.0, 0, 1))
        np.testing.assert_allclose(abs(sv[0]), 1.0, atol=1e-10)

    def test_rxx_pi_over_two_creates_superposition(self):
        sv = _run_statevector(Circuit(2).rxx(math.pi / 2, 0, 1))
        np.testing.assert_allclose(abs(sv[0]), 1 / math.sqrt(2), atol=1e-10)
        np.testing.assert_allclose(abs(sv[3]), 1 / math.sqrt(2), atol=1e-10)

    def test_ryy_pi_over_two_creates_superposition(self):
        sv = _run_statevector(Circuit(2).ryy(math.pi / 2, 0, 1))
        np.testing.assert_allclose(abs(sv[0]), 1 / math.sqrt(2), atol=1e-10)
        np.testing.assert_allclose(abs(sv[3]), 1 / math.sqrt(2), atol=1e-10)


class TestThreeQubitGates:
    def test_ccx_flips_target_when_both_controls_one(self):
        sv = _run_statevector(Circuit(3).x(0).x(1).ccx(0, 1, 2))
        assert _peak_index(sv) == 7
        np.testing.assert_allclose(abs(sv[7]), 1.0, atol=1e-10)

    def test_toffoli_is_alias_for_ccx(self):
        qc_ccx = Circuit(3).x(0).x(1).ccx(0, 1, 2)
        qc_toffoli = Circuit(3).x(0).x(1).toffoli(0, 1, 2)
        sv_ccx = _run_statevector(qc_ccx)
        sv_toffoli = _run_statevector(qc_toffoli)
        np.testing.assert_allclose(sv_ccx, sv_toffoli, atol=1e-10)

    def test_cswap_swaps_when_control_is_one(self):
        sv = _run_statevector(Circuit(3).x(0).x(1).cswap(0, 1, 2))
        assert _peak_index(sv) == 5
        np.testing.assert_allclose(abs(sv[5]), 1.0, atol=1e-10)

    def test_fredkin_is_alias_for_cswap(self):
        qc_cswap = Circuit(3).x(0).x(1).cswap(0, 1, 2)
        qc_fredkin = Circuit(3).x(0).x(1).fredkin(0, 1, 2)
        sv_cswap = _run_statevector(qc_cswap)
        sv_fredkin = _run_statevector(qc_fredkin)
        np.testing.assert_allclose(sv_cswap, sv_fredkin, atol=1e-10)

    def test_ccx_invalid_qubit_raises(self):
        with pytest.raises(IndexError):
            Circuit(3).ccx(0, 1, 5)


class TestUtilityMethods:
    def test_measure_all_adds_measure_gates(self):
        qc = Circuit(2).h(0).cnot(0, 1).measure_all()
        gates = qc.to_gate_list()
        measure_gates = [g for g in gates if g["name"] == "MEASURE"]
        assert len(measure_gates) == 2
        assert measure_gates[0]["qubits"] == [0]
        assert measure_gates[1]["qubits"] == [1]

    def test_measure_all_preserves_statevector(self):
        sv = _run_statevector(Circuit(2).h(0).cnot(0, 1).measure_all())
        np.testing.assert_allclose(abs(sv[0]), 1 / math.sqrt(2), atol=1e-10)
        np.testing.assert_allclose(abs(sv[3]), 1 / math.sqrt(2), atol=1e-10)

    def test_barrier_does_not_change_state(self):
        without = _run_statevector(Circuit(2).h(0).cnot(0, 1))
        with_barrier = _run_statevector(Circuit(2).h(0).barrier().cnot(0, 1))
        np.testing.assert_allclose(without, with_barrier, atol=1e-10)

    def test_barrier_all_qubits_by_default(self):
        qc = Circuit(3).barrier()
        gates = qc.to_gate_list()
        assert gates[0]["name"] == "BARRIER"
        assert gates[0]["qubits"] == [0, 1, 2]

    def test_reset_adds_reset_gate(self):
        qc = Circuit(1).x(0).reset(0)
        gates = qc.to_gate_list()
        assert gates[-1]["name"] == "RESET"
        assert gates[-1]["qubits"] == [0]

    def test_self_inverse_gate_pairs(self):
        pairs = [
            Circuit(1).h(0).h(0),
            Circuit(1).x(0).x(0),
            Circuit(1).s(0).sdg(0),
            Circuit(1).t(0).tdg(0),
        ]
        for qc in pairs:
            sv = _run_statevector(qc)
            np.testing.assert_allclose(abs(sv[0]), 1.0, atol=1e-10)

    def test_to_unitary_matches_gate_matrix(self):
        qc = Circuit(1).h(0)
        u = np.asarray(qc.to_unitary())
        expected = gate_unitary_matrix("H")
        np.testing.assert_allclose(u, expected, atol=1e-10)

    def test_to_unitary_two_qubit_cnot_matches_simulation(self):
        # CNOT(0,1) on |01⟩ (index 1, control=qubit0=1) → |11⟩ (index 3)
        qc = Circuit(2).cnot(0, 1)
        u = np.asarray(qc.to_unitary())
        initial = np.zeros(4, dtype=np.complex128)
        initial[1] = 1.0  # |01⟩: qubit0=1, qubit1=0
        expected = _run_statevector(Circuit(2).x(0).x(1))  # |11⟩
        np.testing.assert_allclose(u @ initial, expected, atol=1e-10)

    def test_draw_contains_qubit_labels(self):
        diagram = Circuit(2).h(0).cnot(0, 1).draw()
        assert "q0:" in diagram
        assert "q1:" in diagram
        assert "H" in diagram

    def test_draw_empty_circuit(self):
        diagram = Circuit(2).draw()
        assert "q0:" in diagram
        assert "q1:" in diagram

    def test_invalid_qubit_for_sdg_raises(self):
        with pytest.raises(IndexError):
            Circuit(1).sdg(3)

    def test_duplicate_qubits_in_cp_raises(self):
        with pytest.raises(ValueError, match="Duplicate qubit"):
            Circuit(2).cp(0.5, 0, 0)
