"""Compiler pass domain tests — individual pass behavior and gate-count effects."""

import math

import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.compiler.passes import (
    BasisTranslationPass,
    ConstantFoldingPass,
    GateCancellationPass,
    RotationMergingPass,
    SwapDecompositionPass,
)


pytestmark = pytest.mark.domain


def _gate_names(circuit: Circuit) -> list[str]:
    return [g.name.upper() for g in circuit._gates]


class TestGateCancellationPass:
    def test_cancels_consecutive_h_gates(self):
        c = sf.Circuit(1).h(0).h(0)
        assert c.gate_count == 2
        result = GateCancellationPass().run(c)
        assert result.gate_count == 0

    def test_cancels_s_sdg_pair(self):
        c = sf.Circuit(1).s(0).sdg(0)
        result = GateCancellationPass().run(c)
        assert result.gate_count == 0

    def test_does_not_cancel_different_qubits(self):
        c = sf.Circuit(2).h(0).h(1)
        result = GateCancellationPass().run(c)
        assert result.gate_count == 2


class TestRotationMergingPass:
    def test_merges_consecutive_rx_on_same_qubit(self):
        c = sf.Circuit(1).rx(0.3, 0).rx(0.5, 0)
        assert c.gate_count == 2
        result = RotationMergingPass().run(c)
        assert result.gate_count == 1
        assert result._gates[0].name.upper() == "RX"
        assert abs(result._gates[0].params[0] - 0.8) < 1e-12

    def test_merges_consecutive_rz(self):
        c = sf.Circuit(1).rz(math.pi / 4, 0).rz(math.pi / 4, 0)
        result = RotationMergingPass().run(c)
        assert result.gate_count == 1
        assert abs(result._gates[0].params[0] - math.pi / 2) < 1e-12


class TestConstantFoldingPass:
    def test_removes_zero_rotation_gates(self):
        c = sf.Circuit(2).rz(0.0, 0).rx(0.5, 1).ry(0.0, 1)
        assert c.gate_count == 3
        result = ConstantFoldingPass().run(c)
        assert result.gate_count == 1
        assert result._gates[0].name.upper() == "RX"

    def test_keeps_nonzero_rotations(self):
        c = sf.Circuit(1).rz(0.1, 0)
        result = ConstantFoldingPass().run(c)
        assert result.gate_count == 1


class TestSwapDecompositionPass:
    def test_decomposes_swap_into_cnots(self):
        c = sf.Circuit(2).swap(0, 1)
        assert c.gate_count == 1
        result = SwapDecompositionPass().run(c)
        assert result.gate_count == 3
        assert all(g.name.upper() in ("CX", "CNOT") for g in result._gates)


class TestBasisTranslationPass:
    def test_decomposes_h_to_native_basis(self):
        c = sf.Circuit(1).h(0)
        native = ["RZ", "RX"]
        result = BasisTranslationPass(native).run(c)
        assert result.gate_count == 3
        names = _gate_names(result)
        assert names == ["RZ", "RX", "RZ"]

    def test_translates_cnot_to_cz_basis(self):
        c = sf.Circuit(2).cnot(0, 1)
        result = BasisTranslationPass(["CZ", "H"]).run(c)
        names = _gate_names(result)
        assert names.count("H") == 2
        assert "CZ" in names

    def test_preserves_native_gates(self):
        c = sf.Circuit(1).rz(0.5, 0)
        result = BasisTranslationPass(["RZ", "RX"]).run(c)
        assert result.gate_count == 1
        assert result._gates[0].name.upper() == "RZ"


class TestPassManagerPipeline:
    def test_pass_manager_runs_sequence(self):
        from superfermion.compiler.manager import PassManager

        c = sf.Circuit(1).h(0).h(0).rx(0.0, 0).rx(0.2, 0).rx(0.3, 0)
        manager = PassManager([
            GateCancellationPass(),
            ConstantFoldingPass(),
            RotationMergingPass(),
        ])
        result = manager.run(c)
        assert result.gate_count == 1
        assert result._gates[0].name.upper() == "RX"
        assert abs(result._gates[0].params[0] - 0.5) < 1e-12
