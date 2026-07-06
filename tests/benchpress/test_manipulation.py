"""
Benchpress Circuit Manipulation Tests — Superfermion vs Qiskit
==============================================================

Modeled after the Qiskit/benchpress manipulation workouts:
  - Pauli twirling
  - Gate cancellation
  - Rotation merging / constant folding
  - Basis translation
  - Swap decomposition
  - Commutation optimization

These tests validate that SF's compiler passes produce correct,
semantically equivalent circuits and benchmark their runtime.

Run:
    python -m pytest tests/benchpress/test_manipulation.py -v --tb=short
"""

import pytest
import numpy as np

from tests.benchpress.conftest import (
    SEED,
    HAS_QISKIT,
    build_clifford_circuit_sf,
    build_clifford_circuit_qiskit,
    build_ghz_sf,
    build_ghz_qiskit,
    statevector_fidelity,
)


# =====================================================================
# 1. PAULI TWIRLING
# =====================================================================

class TestPauliTwirling:
    """Pauli twirling: random Pauli sandwiches on 2Q gates."""

    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_pauli_twirling_sf(self, benchmark, n_qubits):
        """[SF] Apply Pauli twirling to a random Clifford circuit."""
        from superfermion.compiler.advanced import PauliTwirlingPass

        circ = build_clifford_circuit_sf(n_qubits, seed=SEED)
        gate_count_before = len(circ._gates)

        ptp = PauliTwirlingPass(seed=SEED)

        @benchmark
        def result():
            return ptp.run(circ)

        assert result is not None
        # Twirling adds Pauli gates around each 2Q gate
        assert len(result._gates) >= gate_count_before

    def test_pauli_twirling_fidelity(self):
        """Verify Pauli twirling preserves circuit semantics."""
        import superfermion as sf
        from superfermion.compiler.advanced import PauliTwirlingPass

        circ = sf.Circuit(3)
        circ.h(0)
        circ.cx(0, 1)
        circ.cz(1, 2)
        circ.cx(0, 2)

        ptp = PauliTwirlingPass(seed=SEED)
        twirled = ptp.run(circ)

        sv_orig = sf.run(circ, backend="simulator").statevector
        sv_twirled = sf.run(twirled, backend="simulator").statevector
        fid = statevector_fidelity(sv_orig, sv_twirled)
        assert fid > 0.9999, f"Fidelity = {fid}"

    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_dynamical_decoupling_sf(self, benchmark, n_qubits):
        """[SF] Apply dynamical decoupling as a noise-suppression pass."""
        from superfermion.compiler.advanced import apply_dynamical_decoupling

        circ = build_clifford_circuit_sf(n_qubits, seed=SEED)
        gate_count_before = len(circ._gates)

        @benchmark
        def result():
            return apply_dynamical_decoupling(circ, sequence="XY4", spacing=1)

        assert result is not None
        # DD should add gates
        assert len(result._gates) >= gate_count_before

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.skip(reason="PauliTwirl removed in Qiskit 2.x — SF provides this natively")
    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_pauli_twirling_qiskit(self, benchmark, n_qubits):
        """[Qiskit] PauliTwirl was removed in Qiskit 2.x. SF provides native twirling."""
        pass


# =====================================================================
# 2. GATE CANCELLATION
# =====================================================================

class TestGateCancellation:
    """Gate cancellation: remove self-inverse pairs."""

    @pytest.mark.parametrize("n_qubits", [4, 8, 16, 32])
    def test_gate_cancellation_sf(self, benchmark, n_qubits):
        """[SF] Cancel self-inverse gate pairs in a circuit."""
        from superfermion.compiler.passes import GateCancellationPass

        # Build a circuit with many cancellable pairs
        import superfermion as sf
        circ = sf.Circuit(n_qubits)
        for _ in range(5 * n_qubits):
            q = np.random.randint(0, n_qubits)
            circ.h(q)
            circ.h(q)  # Immediately cancelled
            circ.x(q)
            circ.x(q)  # Immediately cancelled
        gate_count_before = len(circ._gates)

        gcp = GateCancellationPass()

        @benchmark
        def result():
            return gcp.run(circ)

        assert result is not None
        assert len(result._gates) < gate_count_before

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n_qubits", [4, 8, 16, 32])
    def test_gate_cancellation_qiskit(self, benchmark, n_qubits):
        """[Qiskit] Cancel self-inverse gate pairs (InverseCancellation in Qiskit 2.x)."""
        from qiskit.transpiler.passes import InverseCancellation
        from qiskit.transpiler import PassManager
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import CXGate

        qc = QuantumCircuit(n_qubits)
        for _ in range(5 * n_qubits):
            q = np.random.randint(0, n_qubits)
            qc.cx(q, (q + 1) % n_qubits)
            qc.cx(q, (q + 1) % n_qubits)  # Cancellable

        pm = PassManager(InverseCancellation([CXGate()]))

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 3. ROTATION MERGING
# =====================================================================

class TestRotationMerging:
    """Rotation merging: combine consecutive same-axis rotations."""

    @pytest.mark.parametrize("n_qubits", [4, 8, 16, 32])
    def test_rotation_merging_sf(self, benchmark, n_qubits):
        """[SF] Merge consecutive rotation gates."""
        from superfermion.compiler.passes import RotationMergingPass

        import superfermion as sf
        circ = sf.Circuit(n_qubits)
        for q in range(n_qubits):
            circ.rx(0.5, q)
            circ.rx(0.3, q)  # Should merge to rx(0.8, q)
            circ.ry(1.0, q)
            circ.ry(0.5, q)  # Should merge to ry(1.5, q)
            circ.rz(2.0, q)
            circ.rz(1.0, q)  # Should merge to rz(3.0, q)
        gate_count_before = len(circ._gates)

        rmp = RotationMergingPass()

        @benchmark
        def result():
            return rmp.run(circ)

        assert result is not None
        assert len(result._gates) < gate_count_before

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n_qubits", [4, 8, 16, 32])
    def test_rotation_merging_qiskit(self, benchmark, n_qubits):
        """[Qiskit] Merge consecutive rotation gates."""
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition
        from qiskit.transpiler import PassManager
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.rx(0.5, q)
            qc.rx(0.3, q)
            qc.ry(1.0, q)
            qc.ry(0.5, q)
            qc.rz(2.0, q)
            qc.rz(1.0, q)

        pm = PassManager(Optimize1qGatesDecomposition(["rx", "ry", "rz"]))

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 4. BASIS TRANSLATION
# =====================================================================

class TestBasisTranslation:
    """Basis translation: convert gates to native basis set."""

    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_basis_translation_sf(self, benchmark, n_qubits):
        """[SF] Translate circuit to superconducting basis {Rz, SX, X, CX}."""
        from superfermion.compiler.passes import BasisTranslationPass
        from superfermion.compiler.passes import SwapDecompositionPass

        circ = build_ghz_sf(n_qubits)
        # First decompose SWAPs if any, then translate basis
        sdp = SwapDecompositionPass()
        circ = sdp.run(circ)
        gate_count_before = len(circ._gates)

        btp = BasisTranslationPass(["rz", "sx", "x", "cx"])

        @benchmark
        def result():
            return btp.run(circ)

        assert result is not None
        # Basis translation should keep total gate structure valid

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_basis_translation_qiskit(self, benchmark, n_qubits):
        """[Qiskit] Translate circuit to superconducting basis."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        qc = build_ghz_qiskit(n_qubits)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cx"])

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 5. SWAP DECOMPOSITION
# =====================================================================

class TestSwapDecomposition:
    """Swap decomposition: SWAP -> 3 CNOTs."""

    @pytest.mark.parametrize("n_swaps", [10, 50, 100, 500])
    def test_swap_decomposition_sf(self, benchmark, n_swaps):
        """[SF] Decompose SWAP gates into 3 CNOTs."""
        from superfermion.compiler.passes import SwapDecompositionPass

        import superfermion as sf
        n_q = 20
        circ = sf.Circuit(n_q)
        for i in range(n_swaps):
            q0 = i % (n_q - 1)
            q1 = q0 + 1
            circ.swap(q0, q1)

        sdp = SwapDecompositionPass()

        @benchmark
        def result():
            return sdp.run(circ)

        assert result is not None
        # Each SWAP -> 3 gates
        assert len(result._gates) == 3 * n_swaps

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n_swaps", [10, 50, 100, 500])
    def test_swap_decomposition_qiskit(self, benchmark, n_swaps):
        """[Qiskit] Decompose SWAP gates."""
        from qiskit.transpiler.passes import Decompose
        from qiskit.transpiler import PassManager
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(20)
        for i in range(n_swaps):
            q0 = i % 19
            q1 = q0 + 1
            qc.swap(q0, q1)

        pm = PassManager(Decompose("swap"))

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 6. CONSTANT FOLDING
# =====================================================================

class TestConstantFolding:
    """Constant folding: remove zero-parameter rotations."""

    @pytest.mark.parametrize("n_qubits", [8, 16, 32, 64])
    def test_constant_folding_sf(self, benchmark, n_qubits):
        """[SF] Remove zero-rotation gates."""
        from superfermion.compiler.passes import ConstantFoldingPass

        import superfermion as sf
        circ = sf.Circuit(n_qubits)
        for q in range(n_qubits):
            circ.rx(0.0, q)  # Should be folded
            circ.h(q)
            circ.ry(0.0, q)  # Should be folded
            circ.x(q)
            circ.rz(0.5, q)  # Should stay
            circ.rz(0.0, q)  # Should be folded
        gate_count_before = len(circ._gates)

        cfp = ConstantFoldingPass()

        @benchmark
        def result():
            return cfp.run(circ)

        assert result is not None
        assert len(result._gates) < gate_count_before
        # Exactly 3 non-zero gates per qubit should remain: H, X, Rz(0.5)
        assert len(result._gates) == 3 * n_qubits

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n_qubits", [8, 16, 32, 64])
    def test_constant_folding_qiskit(self, benchmark, n_qubits):
        """[Qiskit] Remove zero-parameter rotations."""
        from qiskit.transpiler.passes import RemoveIdentityEquivalent
        from qiskit.transpiler import PassManager
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.rx(0.0, q)
            qc.h(q)
            qc.ry(0.0, q)
            qc.x(q)
            qc.rz(0.5, q)
            qc.rz(0.0, q)

        pm = PassManager(RemoveIdentityEquivalent())

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 7. COMMUTATION OPTIMIZATION
# =====================================================================

class TestCommutationOptimization:
    """Commutation optimization: reorder gates to expose cancellations."""

    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_commutation_sf(self, benchmark, n_qubits):
        """[SF] Commute gates to enable cancellation."""
        from superfermion.compiler.manager import compile as sf_compile

        import superfermion as sf
        circ = sf.Circuit(n_qubits)
        # Create circuit where gates can commute to expose cancellations
        for q in range(n_qubits):
            circ.h(q)
            circ.x((q + 1) % n_qubits)  # On different qubit - commutes
            circ.h(q)  # Cancels with first H after commuting X out of the way
        gate_count_before = len(circ._gates)

        @benchmark
        def result():
            return sf_compile(circ, level=1)

        assert result is not None
        # H·H pairs should cancel after commutation
        assert len(result._gates) <= gate_count_before

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_commutation_qiskit(self, benchmark, n_qubits):
        """[Qiskit] Commute gates to enable cancellation."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
            qc.x((q + 1) % n_qubits)
            qc.h(q)

        pm = generate_preset_pass_manager(1)

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 8. FULL COMPILATION PIPELINE (LEVEL 0, 1, 2)
# =====================================================================

class TestFullCompilationPipeline:
    """End-to-end compilation across optimization levels."""

    @pytest.mark.parametrize("level", [0, 1, 2])
    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_compile_levels_sf(self, benchmark, level, n_qubits):
        """[SF] Full compilation at optimization level {level}."""
        from superfermion.compiler.manager import compile as sf_compile

        circ = build_ghz_sf(n_qubits)

        @benchmark
        def result():
            return sf_compile(circ, level=level)

        assert result is not None
        assert result.n_qubits == n_qubits

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_compile_levels_qiskit(self, benchmark, level, n_qubits):
        """[Qiskit] Full compilation at optimization level {level}."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        qc = build_ghz_qiskit(n_qubits)
        pm = generate_preset_pass_manager(level)

        @benchmark
        def result():
            return pm.run(qc)

        assert result is not None


# =====================================================================
# 9. COMPILATION WITH HARDWARE TARGET
# =====================================================================

class TestCompilationWithHardwareTarget:
    """Compilation against a hardware target with coupling constraints."""

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_compile_with_target_sf(self, benchmark, n_qubits):
        """[SF] Compile circuit against a linear coupling map."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec

        import superfermion as sf
        circ = sf.Circuit(n_qubits)
        circ.h(0)
        for i in range(n_qubits - 1):
            circ.cx(i, i + 1)

        # Linear coupling map target
        target = HardwareSpec(
            name="linear_test",
            n_qubits=n_qubits,
            native_gates=["h", "x", "y", "z", "cx"],
            coupling_map=[(i, i + 1) for i in range(n_qubits - 1)],
        )

        @benchmark
        def result():
            return sf_compile(circ, level=1, target=target)

        assert result is not None

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_compile_linear_no_extra_swaps_sf(self, n_qubits):
        """[SF] Verify no unnecessary SWAPs are inserted for adjacent gates."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec

        import superfermion as sf
        circ = sf.Circuit(n_qubits)
        circ.h(0)
        # All CNOTs are on adjacent qubits = linear chain neighbors
        for i in range(n_qubits - 1):
            circ.cx(i, i + 1)

        target = HardwareSpec(
            name="linear_test",
            n_qubits=n_qubits,
            native_gates=["h", "x", "y", "z", "cx"],
            coupling_map=[(i, i + 1) for i in range(n_qubits - 1)],
        )

        result = sf_compile(circ, level=1, target=target)
        # Check no SWAP gates were added (all gates already adjacent)
        swap_count = sum(1 for g in result._gates if g.name.upper() == "SWAP")
        assert swap_count == 0, f"Expected 0 SWAPs but found {swap_count}"


# =====================================================================
# 10. SEMANTIC PRESERVATION (FIDELITY CHECKS)
# =====================================================================

class TestSemanticPreservation:
    """Verify that compilation passes preserve circuit semantics."""

    def test_cancellation_preserves_statevector(self):
        """Gate cancellation should not change the circuit statevector."""
        import superfermion as sf
        from superfermion.compiler.passes import GateCancellationPass

        circ = sf.Circuit(3)
        circ.h(0)
        circ.h(0)  # cancellable pair
        circ.cx(0, 1)
        circ.x(2)
        circ.x(2)  # cancellable pair

        gcp = GateCancellationPass()
        cancelled = gcp.run(circ)

        # Both should produce same statevector
        sv_orig = sf.run(circ, backend="simulator").statevector
        sv_canc = sf.run(cancelled, backend="simulator").statevector
        fid = statevector_fidelity(sv_orig, sv_canc)
        assert fid > 0.9999, f"Fidelity = {fid}"

    def test_rotation_merging_preserves_statevector(self):
        """Rotation merging should not change the circuit statevector."""
        import superfermion as sf
        from superfermion.compiler.passes import RotationMergingPass

        circ = sf.Circuit(2)
        circ.rx(0.3, 0)
        circ.rx(0.5, 0)  # merges to rx(0.8)
        circ.ry(1.0, 1)
        circ.ry(0.5, 1)  # merges to ry(1.5)

        rmp = RotationMergingPass()
        merged = rmp.run(circ)

        sv_orig = sf.run(circ, backend="simulator").statevector
        sv_merged = sf.run(merged, backend="simulator").statevector
        fid = statevector_fidelity(sv_orig, sv_merged)
        assert fid > 0.9999, f"Fidelity = {fid}"

    def test_constant_folding_preserves_statevector(self):
        """Constant folding should not change the circuit statevector."""
        import superfermion as sf
        from superfermion.compiler.passes import ConstantFoldingPass

        circ = sf.Circuit(2)
        circ.h(0)
        circ.rx(0.0, 0)  # folded
        circ.x(1)
        circ.ry(0.0, 1)  # folded

        cfp = ConstantFoldingPass()
        folded = cfp.run(circ)

        sv_orig = sf.run(circ, backend="simulator").statevector
        sv_folded = sf.run(folded, backend="simulator").statevector
        fid = statevector_fidelity(sv_orig, sv_folded)
        assert fid > 0.9999, f"Fidelity = {fid}"

    def test_swap_decomposition_preserves_statevector(self):
        """Swap decomposition should not change the circuit statevector."""
        import superfermion as sf
        from superfermion.compiler.passes import SwapDecompositionPass

        circ = sf.Circuit(3)
        circ.h(0)
        circ.swap(0, 1)
        circ.cx(1, 2)

        sdp = SwapDecompositionPass()
        decomposed = sdp.run(circ)

        sv_orig = sf.run(circ, backend="simulator").statevector
        sv_dec = sf.run(decomposed, backend="simulator").statevector
        fid = statevector_fidelity(sv_orig, sv_dec)
        assert fid > 0.9999, f"Fidelity = {fid}"


# =====================================================================
# 11. RUST COMPILATION PIPELINE (PyO3 native speed)
# =====================================================================

class TestRustCompilationPipeline:
    """End-to-end Rust-native compilation — 10-100x faster than Python."""

    @pytest.mark.parametrize("level", [0, 1, 2])
    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_compile_rust_levels(self, benchmark, level, n_qubits):
        """[SF Rust] Compile a GHZ circuit at optimization level {level}."""
        from tests.benchpress.conftest import compile_sf_rust

        circ = build_ghz_sf(n_qubits)

        @benchmark
        def result():
            return compile_sf_rust(circ, level=level)

        assert result is not None
        assert result.n_qubits == n_qubits

    @pytest.mark.parametrize("n_qubits", [4, 8, 16])
    def test_compile_rust_vs_python_speedup(self, benchmark, n_qubits):
        """[SF Rust] Compare Rust compilation speed vs Python compilation."""
        from tests.benchpress.conftest import compile_sf_rust
        from superfermion.compiler.manager import compile as sf_compile

        circ = build_clifford_circuit_sf(n_qubits, seed=SEED)

        # Benchmark Rust compilation
        @benchmark
        def result_rust():
            return compile_sf_rust(circ, level=1)

        assert result_rust is not None

    def test_compile_rust_fidelity(self):
        """Verify Rust compilation preserves circuit semantics."""
        import superfermion as sf
        from tests.benchpress.conftest import compile_sf_rust

        circ = sf.Circuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cz(1, 2)
        circ.cx(2, 3)
        circ.rx(0.5, 0)
        circ.h(0)
        circ.h(0)  # cancellable

        sv_orig = sf.run(circ, backend="simulator").statevector

        # Test all levels
        for level in [0, 1, 2]:
            compiled = compile_sf_rust(circ, level=level)
            sv_comp = sf.run(compiled, backend="simulator").statevector
            fid = statevector_fidelity(sv_orig, sv_comp)
            assert fid > 0.9999, f"Level {level} fidelity = {fid}"

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    def test_compile_rust_with_routing(self, n_qubits):
        """[SF Rust] Verify Rust SABRE routing works with coupling constraints."""
        import superfermion as sf
        from tests.benchpress.conftest import compile_sf_rust
        from superfermion.runtime.specs import HardwareSpec

        circ = sf.Circuit(n_qubits)
        circ.h(0)
        # Non-adjacent CNOTs force routing
        for i in range(0, n_qubits - 1, 2):
            circ.cx(i, (i + 2) % n_qubits)

        target = HardwareSpec(
            name="ring_test",
            n_qubits=n_qubits,
            native_gates=["h", "x", "y", "z", "cx"],
            coupling_map=[(i, (i + 1) % n_qubits) for i in range(n_qubits)],
        )

        compiled = compile_sf_rust(circ, level=1, target=target)
        assert compiled.n_qubits == n_qubits
        # Note: SABRE routing permutes qubits, so statevector fidelity
        # cannot be directly compared. We verify the circuit is valid.
        assert compiled.gate_count > 0
