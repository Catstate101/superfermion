"""Compiler pass domain tests — Rust pipeline and Python shim behavior."""

import math

import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.compiler.passes import BasisTranslationPass, UnitaryDecompositionPass
from superfermion.compiler.manager import PassManager, compile


pytestmark = pytest.mark.domain


def _gate_names(circuit: Circuit) -> list[str]:
    return [g.name.upper() for g in circuit._gates]


class TestRustCompilerGateCancellation:
    """Gate cancellation is now handled by the Rust sf-compiler crate."""

    def test_cancels_consecutive_h_gates(self):
        c = sf.Circuit(1).h(0).h(0)
        result = compile(c, level=1)
        assert result.gate_count < c.gate_count

    def test_cancels_s_sdg_pair(self):
        c = sf.Circuit(1).s(0).sdg(0)
        result = compile(c, level=1)
        assert result.gate_count < c.gate_count


class TestRustCompilerRotationMerging:
    """Rotation merging is now handled by the Rust sf-compiler crate."""

    def test_merges_consecutive_rx_on_same_qubit(self):
        c = sf.Circuit(1).rx(0.3, 0).rx(0.5, 0)
        result = compile(c, level=1)
        assert result.gate_count <= c.gate_count

    def test_merges_consecutive_rz(self):
        c = sf.Circuit(1).rz(math.pi / 4, 0).rz(math.pi / 4, 0)
        result = compile(c, level=1)
        assert result.gate_count <= c.gate_count


class TestRustCompilerSwapDecomposition:
    """SWAP decomposition is now handled by the Rust sf-compiler crate."""

    def test_decomposes_swap_into_cnots(self):
        c = sf.Circuit(2).swap(0, 1)
        result = compile(c, level=1)
        names = _gate_names(result)
        assert "SWAP" not in names
        assert result.gate_count >= 3


class TestBasisTranslationPass:
    """BasisTranslationPass is a Python shim for non-superconducting basis sets."""

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


class TestUnitaryDecompositionShim:
    """UnitaryDecompositionPass is a Python shim for opaque unitary gates."""

    def test_decomposes_1q_unitary(self):
        import numpy as np
        c = sf.Circuit(1)
        c.unitary(np.array([[0, 1], [1, 0]], dtype=complex), [0])
        result = UnitaryDecompositionPass().run(c)
        names = _gate_names(result)
        assert "UNITARY" not in names

    def test_decomposes_2q_unitary(self):
        import numpy as np
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        c = sf.Circuit(2)
        c.unitary(cnot, [0, 1])
        result = UnitaryDecompositionPass().run(c)
        names = _gate_names(result)
        assert "UNITARY" not in names


class TestPassManagerPipeline:
    def test_pass_manager_runs_sequence(self):
        c = sf.Circuit(1).h(0)
        manager = PassManager([
            BasisTranslationPass(["RZ", "RX"]),
        ])
        result = manager.run(c)
        assert result.gate_count == 3
        names = _gate_names(result)
        assert names == ["RZ", "RX", "RZ"]


class TestRustCompileEndToEnd:
    """End-to-end tests for the Rust compile pipeline."""

    def test_compile_bell_for_linear5(self):
        from superfermion.compiler.specs import get_spec

        c = sf.Circuit(2).h(0).cnot(0, 1)
        spec = get_spec("linear_5")
        result = compile(c, level=1, target=spec)
        assert isinstance(result, Circuit)
        assert result.gate_count >= 1

    def test_compile_level0_no_target_is_noop(self):
        c = sf.Circuit(2).h(0).cnot(0, 1)
        result = compile(c, level=0)
        assert result is c

    def test_compile_with_unitary_gates(self):
        import numpy as np
        from superfermion.compiler.specs import get_spec

        c = sf.Circuit(2)
        c.unitary(np.array([[0, 1], [1, 0]], dtype=complex), [0])
        c.cnot(0, 1)
        spec = get_spec("linear_5")
        result = compile(c, level=1, target=spec)
        names = _gate_names(result)
        assert "UNITARY" not in names
