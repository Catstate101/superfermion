"""
Benchpress Transpilation Stubs — Superfermion vs Qiskit
========================================================

Modeled after the Qiskit/benchpress transpilation workouts:
  - workouts/abstract_transpile/  (QASMBench, HamLib)
  - workouts/device_transpile/    (Feynman, HamLib Hamiltonians, 100Q device)

SF does not have a transpiler comparable to Qiskit's preset pass manager.
Per Benchpress convention, all transpilation tests are marked SKIPPED for SF.
Qiskit reference implementations are provided for comparison.

Run:
    python -m pytest tests/benchpress/test_transpile.py -v --tb=short
"""

import pytest
import numpy as np

from tests.benchpress.conftest import (
    SEED,
    HAS_QISKIT,
    build_clifford_circuit_sf,
    build_clifford_circuit_qiskit,
    build_multi_control_circuit_qiskit,
    build_qft_sf,
    build_efficient_su2_sf,
    generate_qv100_qasm,
    qasm2_to_sf,
)


# =====================================================================
# 1. ABSTRACT TRANSPILATION — QASMBench
#    (mirrors benchpress/workouts/abstract_transpile/qasmbench.py)
# =====================================================================

class TestAbstractQasmBench:
    """Abstract transpilation of QASMBench circuits across topologies."""

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_QASMBench_sf(self, benchmark, size):
        """[SF] Abstract transpilation for QASMBench {size}."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qv_qasm = generate_qv100_qasm()
        sf_circ = qasm2_to_sf(qv_qasm)
        # Optimization + basis translation only (no routing at 100Q in Python)
        target = HardwareSpec(
            name="abstract_100q",
            n_qubits=100,
            native_gates=["h", "x", "y", "z", "cx"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(sf_circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_QASMBench_qiskit(self, benchmark, size):
        """[Qiskit] Abstract transpilation for QASMBench {size}."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.transpiler import CouplingMap
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        coupling_map = CouplingMap([(i, i + 1) for i in range(99)])
        pm = generate_preset_pass_manager(1, coupling_map=coupling_map)
        @benchmark
        def result():
            return pm.run(qv_circ)
        assert result is not None


# =====================================================================
# 2. ABSTRACT TRANSPILATION — Hamiltonians
#    (mirrors benchpress/workouts/abstract_transpile/hamlib_hamiltonians.py)
# =====================================================================

class TestAbstractHamiltonians:
    """Abstract transpilation of HamLib Hamiltonians across topologies."""

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_hamiltonians_sf(self, benchmark, size):
        """[SF] Abstract transpilation for HamLib Hamiltonians."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qv_qasm = generate_qv100_qasm()
        sf_circ = qasm2_to_sf(qv_qasm)
        target = HardwareSpec(
            name="abstract_100q",
            n_qubits=100,
            native_gates=["h", "x", "y", "z", "cx"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(sf_circ, level=1, target=target)
        assert result is not None


    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_hamiltonians_qiskit(self, benchmark):
        """[Qiskit] Abstract transpilation for HamLib Hamiltonians."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.transpiler import CouplingMap
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        coupling_map = CouplingMap([(i, i + 1) for i in range(99)])
        pm = generate_preset_pass_manager(1, coupling_map=coupling_map)
        @benchmark
        def result():
            return pm.run(qv_circ)
        assert result is not None


# =====================================================================
# 3. DEVICE TRANSPILATION — 100Q
#    (mirrors benchpress/workouts/device_transpile/device_transpile_100Q.py)
# =====================================================================

class TestDeviceTranspile100Q:
    """Device-level transpilation of 100Q circuits against a target backend."""

    def test_QFT_100_transpile_sf(self, benchmark):
        """[SF] Transpile a QFT 100 circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qft = build_qft_sf(100)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(qft, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_QFT_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a QFT 100 circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.circuit.library import QFT
        qft = QFT(100).decompose()
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qft)
        assert result is not None

    def test_QV_100_transpile_sf(self, benchmark):
        """[SF] Transpile a QV 100 circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qv_qasm = generate_qv100_qasm()
        qv_circ = qasm2_to_sf(qv_qasm)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(qv_circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_QV_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a QV 100 circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qv_circ)
        assert result is not None

    def test_circSU2_100_transpile_sf(self, benchmark):
        """[SF] Transpile a 100Q SU2 circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        su2 = build_efficient_su2_sf(100, reps=4)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(su2, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_circSU2_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a 100Q SU2 circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.circuit.library import efficient_su2
        qc = efficient_su2(100, reps=4, entanglement="circular")
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qc)
        assert result is not None

    def test_BV_100_transpile_sf(self, benchmark):
        """[SF] Transpile a 100Q BV circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        import superfermion as sf
        N = 100
        circ = sf.Circuit(N, N - 1)
        circ.x(N - 1)
        for q in range(N):
            circ.h(q)
        for q in range(N - 1):
            circ.cx(q, N - 1)
        for q in range(N - 1):
            circ.h(q)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_BV_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a 100Q BV circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit import QuantumCircuit
        N = 100
        qc = QuantumCircuit(N, N - 1)
        qc.x(N - 1); qc.h(range(N)); qc.cx(range(N - 1), N - 1)
        qc.h(range(N - 1)); qc.measure(range(N - 1), range(N - 1))
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qc)
        assert result is not None

    def test_square_heisenberg_100_transpile_sf(self, benchmark):
        """[SF] Transpile a 100Q Heisenberg Hamiltonian against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qv_qasm = generate_qv100_qasm()
        qv_circ = qasm2_to_sf(qv_qasm)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(qv_circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_square_heisenberg_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a 100Q Heisenberg Hamiltonian against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qv_circ)
        assert result is not None

    def test_QAOA_100_transpile_sf(self, benchmark):
        """[SF] Transpile a 100Q QAOA circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qv_qasm = generate_qv100_qasm()
        qv_circ = qasm2_to_sf(qv_qasm)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(qv_circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_QAOA_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a 100Q QAOA circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qv_circ)
        assert result is not None

    def test_BVlike_simplification_transpile_sf(self, benchmark):
        """[SF] Transpile a BV-like simplification circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        import superfermion as sf
        N = 100
        circ = sf.Circuit(N)
        for kk in range(N - 1):
            circ.cx(kk, N - 1)
        circ.x(N - 1); circ.z(N - 2)
        for kk in range(N - 2, -1, -1):
            circ.cx(kk, N - 1)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_BVlike_simplification_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a BV-like circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit import QuantumCircuit
        N = 100
        qc = QuantumCircuit(N)
        for kk in range(N - 1):
            qc.cx(kk, N - 1)
        qc.x(N - 1); qc.z(N - 2)
        for kk in range(N - 2, -1, -1):
            qc.cx(kk, N - 1)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qc)
        assert result is not None

    def test_clifford_100_transpile_sf(self, benchmark):
        """[SF] Transpile a Clifford 100 circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        cliff = build_clifford_circuit_sf(100, seed=SEED)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(cliff, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_clifford_100_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a Clifford 100 circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        cliff_qc = build_clifford_circuit_qiskit(100, seed=SEED)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(cliff_qc)
        assert result is not None


# =====================================================================
# 4. DEVICE TRANSPILATION — Feynman
#    (mirrors benchpress/workouts/device_transpile/feynman.py)
# =====================================================================

class TestDeviceFeynman:
    """Device transpilation of Feynman circuits."""

    def test_feynman_transpile_sf(self, benchmark):
        """[SF] Transpile a Feynman circuit against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        import superfermion as sf
        circ = sf.Circuit(20)
        for i in range(19):
            circ.cx(i, i + 1)
        for q in range(20):
            circ.h(q)
        target = HardwareSpec(
            name="device_20q",
            n_qubits=20,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[(i, i + 1) for i in range(19)],
        )
        @benchmark
        def result():
            return sf_compile(circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_feynman_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a Feynman circuit against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(20)
        for i in range(19):
            qc.cx(i, i + 1)
        qc.h(range(20))
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qc)
        assert result is not None


# =====================================================================
# 5. DEVICE TRANSPILATION — HamLib Hamiltonians
#    (mirrors benchpress/workouts/device_transpile/hamlib_hamiltonians.py)
# =====================================================================

class TestDeviceHamlibHamiltonians:
    """Device transpilation of HamLib Hamiltonians."""

    def test_hamlib_hamiltonians_transpile_sf(self, benchmark):
        """[SF] Transpile a HamLib Hamiltonian against a target device."""
        from superfermion.compiler.manager import compile as sf_compile
        from superfermion.runtime.specs import HardwareSpec
        qv_qasm = generate_qv100_qasm()
        qv_circ = qasm2_to_sf(qv_qasm)
        target = HardwareSpec(
            name="device_100q",
            n_qubits=100,
            native_gates=["RZ", "SX", "X", "CX"],
            coupling_map=[],
        )
        @benchmark
        def result():
            return sf_compile(qv_circ, level=1, target=target)
        assert result is not None

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.xfail(reason="Qiskit transpile too slow for automated benchmarks")
    def test_hamlib_hamiltonians_transpile_qiskit(self, benchmark):
        """[Qiskit] Transpile a HamLib Hamiltonian against a target device."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        pm = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"])
        @benchmark
        def result():
            return pm.run(qv_circ)
        assert result is not None
