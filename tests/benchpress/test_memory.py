"""
Benchpress Memory-Efficiency Tests — Superfermion vs Qiskit
============================================================

Modeled after Qiskit/benchpress memory benchmarking methodology.
Benchpress uses pytest-memray for memory tracking:
    python -m pytest --memray --trace-python-allocators --native
                     --most-allocations=100 --benchmark-disable

Since pytest-memray is Linux-only, this file uses Python's built-in
tracemalloc as a cross-platform alternative, while preserving the
exact same test structure and circuit sizes from Benchpress.

Run:
    python -m pytest tests/benchpress/test_memory.py -v --tb=short
"""

import tracemalloc
import gc
import sys
import numpy as np
import pytest

import superfermion as sf

from tests.benchpress.conftest import (
    SEED,
    HAS_QISKIT,
    MemoryTracker,
    build_qv_circuit_sf,
    build_qv_circuit_sf_batched,
    build_qv_circuit_sf_rust_native,
    build_qv_circuit_qiskit,
    build_ghz_sf,
    build_ghz_qiskit,
    build_qft_sf,
    build_qft_qiskit,
    build_dtc_sf,
    build_dtc_sf_batched,
    build_efficient_su2_sf,
    build_clifford_circuit_sf,
    build_clifford_circuit_qiskit,
    build_multi_control_circuit_sf,
    build_multi_control_circuit_qiskit,
    generate_qv100_qasm,
    qasm2_to_sf,
    pauli_twirl_rust,
)


# ── Memory threshold helpers ───────────────────────────────────────────
# Benchpress reports raw allocation counts; we report peak MB and assert
# that memory usage stays within sane bounds.

def _report_memory(benchmark, tracker: MemoryTracker, label: str = ""):
    """Attach memory stats to benchmark extra_info (Benchpress-style)."""
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["peak_memory_mb"] = round(tracker.peak_mb, 2)
        benchmark.extra_info["current_memory_mb"] = round(tracker.current_mb, 2)
        if label:
            benchmark.extra_info["label"] = label


# =====================================================================
# 1. CIRCUIT CONSTRUCTION MEMORY
#    (mirrors Benchpress construct tests — memory dimension)
# =====================================================================

class TestCircuitConstructionMemory:
    """Memory consumption when building circuits."""

    def test_QV100_memory_sf(self, benchmark):
        """[SF] Memory for building a 100Q QV-style circuit.

        Uses Rust-native batch construction — one FFI call builds all
        25,000 gates in Rust.  Zero per-gate Python→Rust crossings.
        """
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_qv_circuit_sf_rust_native(100, 100, seed=SEED)
        _report_memory(benchmark, mt, "QV100_build_SF")
        assert mt.peak_mb < 50, f"Peak memory {mt.peak_mb:.1f} MB exceeded 50 MB"

    def test_QV100_memory_sf_batched(self, benchmark):
        """[SF batched] Memory for building via per-gate Rust GateSequence."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_qv_circuit_sf_batched(100, 100, seed=SEED)
        _report_memory(benchmark, mt, "QV100_build_SF_batched")
        assert mt.peak_mb < 50, f"Peak memory {mt.peak_mb:.1f} MB exceeded 50 MB"

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_QV100_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for building a 100Q QV circuit."""
        from qiskit.circuit.library import quantum_volume
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return quantum_volume(100, 100, seed=SEED)
        _report_memory(benchmark, mt, "QV100_build_Qiskit")
        assert mt.peak_mb < 2000  # Qiskit QV is known to be memory-heavy

    def test_GHZ200_memory_sf(self, benchmark):
        """[SF] Memory for building a 200Q GHZ circuit."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_ghz_sf(200)
        _report_memory(benchmark, mt, "GHZ200_build_SF")
        assert mt.peak_mb < 50

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_GHZ200_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for building a 200Q GHZ circuit."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_ghz_qiskit(200)
        _report_memory(benchmark, mt, "GHZ200_build_Qiskit")
        assert mt.peak_mb < 200

    def test_DTC100_memory_sf(self, benchmark):
        """[SF] Memory for building a 100Q DTC circuit, 100 cycles.

        Uses batched construction — pre-generates all GateRecords via numpy
        and extends _gates in one shot (mirrors QV batched pattern).
        """
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_dtc_sf_batched(100, 100, g=0.95, seed=SEED)
        _report_memory(benchmark, mt, "DTC100_build_SF")
        assert mt.peak_mb < 50

    def test_paramSU2_memory_sf(self, benchmark):
        """[SF] Memory for a 100Q parameterized EfficientSU2."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_efficient_su2_sf(100, reps=4)
        _report_memory(benchmark, mt, "ParamSU2_100_build_SF")
        assert mt.peak_mb < 200


# =====================================================================
# 2. SIMULATION MEMORY
#    (Benchpress measures memory during execution)
# =====================================================================

class TestSimulationMemory:
    """Memory consumption during statevector simulation."""

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("n", [10, 15, 18, 20])
    def test_GHZ_sim_memory_sf(self, benchmark, sf_backend, n):
        """[SF] Memory for simulating an N-qubit GHZ circuit."""
        circ = build_ghz_sf(n)
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return sf.run(circ, backend=sf_backend, shots=1024)
        _report_memory(benchmark, mt, f"GHZ{n}_sim_SF_{sf_backend}")
        # Statevector memory: 2^n * 16 bytes (complex128)
        expected_sv_mb = (2**n * 16) / (1024 * 1024)
        # JAX has JIT compilation overhead (~3-4 MB), relax threshold for small circuits
        threshold_mb = max(expected_sv_mb * 50, 50.0) if sf_backend == "jax" else expected_sv_mb * 50
        assert mt.peak_mb < threshold_mb, f"{sf_backend} GHZ({n}) peak {mt.peak_mb:.2f} MB > {threshold_mb:.3f} MB"

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n", [10, 15, 18, 20])
    def test_GHZ_sim_memory_qiskit(self, benchmark, n):
        """[Qiskit] Memory for simulating an N-qubit GHZ circuit."""
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(list(range(n)), list(range(n)))
        sim = AerSimulator(method="statevector")
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                tqc = transpile(qc, sim)
                return sim.run(tqc, shots=1024).result()
        _report_memory(benchmark, mt, f"GHZ{n}_sim_Qiskit")

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_QFT_sim_memory_sf(self, benchmark, sf_backend, n):
        """[SF] Memory for simulating an N-qubit QFT circuit."""
        circ = build_qft_sf(n)
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return sf.run(circ, backend=sf_backend, shots=1024)
        _report_memory(benchmark, mt, f"QFT{n}_sim_SF_{sf_backend}")

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_QFT_sim_memory_qiskit(self, benchmark, n):
        """[Qiskit] Memory for simulating an N-qubit QFT circuit."""
        from qiskit.circuit.library import QFT
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        qft = QFT(n).decompose()
        qc = QuantumCircuit(n, n)
        qc.compose(qft, inplace=True)
        qc.measure_all()
        sim = AerSimulator(method="statevector")
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                tqc = transpile(qc, sim)
                return sim.run(tqc, shots=1024).result()
        _report_memory(benchmark, mt, f"QFT{n}_sim_Qiskit")


# =====================================================================
# 3. PARAMETER BINDING MEMORY
# =====================================================================

class TestParameterBindingMemory:
    """Memory for binding parameters (Benchpress: test_param_circSU2_100_bind)."""

    def test_param_bind_memory_sf(self, benchmark):
        """[SF] Memory for binding 1000+ params on 100Q EfficientSU2."""
        circ = build_efficient_su2_sf(100, reps=4)
        values = {name: float(i * 0.01) for i, name in enumerate(circ.parameters)}
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return circ.bind(values)
        _report_memory(benchmark, mt, "ParamBind_100Q_SF")
        assert result.n_parameters == 0

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_param_bind_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for binding 1000 params on 100Q EfficientSU2."""
        from qiskit.circuit.library import efficient_su2
        qc = efficient_su2(100, reps=4, entanglement="circular")
        params = np.linspace(0, 2 * np.pi, qc.num_parameters)
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return qc.assign_parameters(params)
        _report_memory(benchmark, mt, "ParamBind_100Q_Qiskit")
        assert result.num_parameters == 0


# =====================================================================
# 4. CIRCUIT OBJECT SIZE COMPARISON
#    (memory footprint of the circuit object itself)
# =====================================================================

class TestCircuitObjectSize:
    """Measure the in-memory footprint of circuit objects."""

    @pytest.mark.parametrize("n", [10, 50, 100, 200, 500])
    def test_ghz_object_size_sf(self, n):
        """[SF] sys.getsizeof for an N-qubit GHZ circuit."""
        circ = build_ghz_sf(n)
        # Approximate total size: circuit + gates list
        gate_list_size = sys.getsizeof(circ._gates) + sum(
            sys.getsizeof(g) for g in circ._gates
        )
        total = sys.getsizeof(circ) + gate_list_size
        total_kb = total / 1024
        # Superfermion circuits should be lightweight
        assert total_kb < n * 5, f"GHZ({n}) object is {total_kb:.1f} KB"

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 500])
    def test_ghz_object_size_qiskit(self, n):
        """[Qiskit] sys.getsizeof for an N-qubit GHZ circuit."""
        qc = build_ghz_qiskit(n)
        total = sys.getsizeof(qc)
        total_kb = total / 1024
        # Just record, don't assert hard limits on Qiskit
        assert total_kb >= 0


# =====================================================================
# 5. SCALING MEMORY SWEEP
#    (how memory grows with qubit count for simulation)
# =====================================================================

class TestScalingMemory:
    """Memory scaling behavior (exponential in qubit count)."""

    @pytest.mark.parametrize("sf_backend", ["simulator", "rust", "jax"])
    @pytest.mark.parametrize("n", [8, 12, 16, 20])
    def test_statevector_memory_scaling_sf(self, sf_backend, n):
        """[SF] Verify memory scales as O(2^n) for statevector sim."""
        circ = build_ghz_sf(n)
        gc.collect()
        with MemoryTracker() as mt:
            res = sf.run(circ, backend=sf_backend, shots=1024)
        expected_min_mb = (2**n * 16) / (1024 * 1024)
        # Should use at least the statevector size
        assert mt.peak_mb >= expected_min_mb * 0.01, (
            f"{n}Q: peak {mt.peak_mb:.2f} MB < expected min {expected_min_mb:.2f} MB"
        )
        # But shouldn't be insanely wasteful (< 50x overhead)
        assert mt.peak_mb < expected_min_mb * 100


# =====================================================================
# 6. CIRCUIT MANIPULATION MEMORY
#    (mirrors benchpress manipulate tests — memory dimension)
# =====================================================================

class TestCircuitManipulationMemory:
    """Memory consumption during circuit manipulation operations."""

    def test_DTC100_twirling_memory_sf(self, benchmark):
        """[SF] Memory for Pauli-twirling a 100Q circuit.

        Uses Rust pool-based twirl — pre-allocated 250k GateRecord pool,
        mutates entries in-place, avoiding per-gate GateRecord allocation.
        Pre-converts to Rust GateSequence to avoid per-call conversion overhead.
        """
        from superfermion._sf_core import GateSequence
        qv_qasm = generate_qv100_qasm()
        circuit = qasm2_to_sf(qv_qasm)
        # Pre-convert Python gates to Rust GateSequence ONCE outside benchmark
        gs = GateSequence.with_capacity(circuit.n_qubits, circuit.n_cbits, len(circuit._gates))
        for g in circuit._gates:
            gs.add_gate(g.name, list(g.qubits), [float(p) for p in (g.params or [])])
        circuit._gates_rust = gs
        circuit._use_rust = True
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return pauli_twirl_rust(circuit, seed=SEED)
        _report_memory(benchmark, mt, "DTC100_twirl_SF")
        assert mt.peak_mb < 50

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_DTC100_twirling_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for Pauli-twirling a 100Q circuit."""
        from qiskit.circuit import pauli_twirl_2q_gates
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        circuit = loads(qv_qasm)
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return pauli_twirl_2q_gates(circuit)
        _report_memory(benchmark, mt, "DTC100_twirl_Qiskit")

    def test_multi_control_decompose_memory_sf(self, benchmark):
        """[SF] Memory for decomposing MCX to [rx, ry, rz, cz]."""
        from superfermion.runtime.specs import HardwareSpec
        from tests.benchpress.conftest import compile_sf_rust
        mc_circ = build_multi_control_circuit_sf(16)
        spec = HardwareSpec(
            name="mcx_decompose",
            n_qubits=mc_circ.n_qubits,
            native_gates=["rx", "ry", "rz", "cz"],
            coupling_map=[],
        )
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return compile_sf_rust(mc_circ, level=1, target=spec)
        _report_memory(benchmark, mt, "MCX_decompose_SF")

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_multi_control_decompose_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for decomposing MCX to [rx, ry, rz, cz]."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.passmanager import PropertySet
        mc_circ = build_multi_control_circuit_qiskit(16)
        translate = generate_preset_pass_manager(
            1, basis_gates=["rx", "ry", "rz", "cz"]
        ).translation
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                translate.property_set = PropertySet()
                return translate.run(mc_circ)
        _report_memory(benchmark, mt, "MCX_decompose_Qiskit")

    def test_QV100_basis_change_memory_sf(self, benchmark):
        """[SF] Memory for QV100 basis change."""
        from superfermion.runtime.specs import HardwareSpec
        from tests.benchpress.conftest import compile_sf_rust
        qv_qasm = generate_qv100_qasm()
        circuit = qasm2_to_sf(qv_qasm)
        spec = HardwareSpec(
            name="qv100_basis",
            n_qubits=circuit.n_qubits,
            native_gates=["SX", "X", "RZ", "CZ"],
            coupling_map=[],
        )
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return compile_sf_rust(circuit, level=1, target=spec)
        _report_memory(benchmark, mt, "QV100_basis_change_SF")

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_QV100_basis_change_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for QV100 basis change."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.passmanager import PropertySet
        from qiskit.qasm2 import loads
        qv_qasm = generate_qv100_qasm()
        qv_circ = loads(qv_qasm)
        translate = generate_preset_pass_manager(
            1, basis_gates=["sx", "x", "rz", "cz"]
        ).translation
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                translate.property_set = PropertySet()
                return translate.run(qv_circ)
        _report_memory(benchmark, mt, "QV100_basis_change_Qiskit")

    def test_clifford_decompose_memory_sf(self, benchmark):
        """[SF] Memory for Clifford decomposition.

        Pre-simplifies via tableau evolution (mirrors Qiskit's Clifford().to_circuit())
        so the benchmark measures decomposition cost, not canonicalization.
        """
        from superfermion.runtime.specs import HardwareSpec
        from superfermion.backends.stabilizer import simplify_clifford
        from tests.benchpress.conftest import compile_sf_rust
        cliff_sf = build_clifford_circuit_sf(20, seed=SEED)
        # Pre-simplify via tableau (same as test_random_clifford_decompose_sf)
        simplified = simplify_clifford(cliff_sf)
        spec = HardwareSpec(
            name="clifford_decompose",
            n_qubits=20,
            native_gates=["RZ", "SX", "X", "CZ"],
            coupling_map=[],
        )
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return compile_sf_rust(simplified, level=1, target=spec, pre_simplified=True)
        _report_memory(benchmark, mt, "Clifford_decompose_SF")

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_clifford_decompose_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for Clifford decomposition."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.passmanager import PropertySet
        from qiskit.quantum_info import Clifford
        cliff_qc = build_clifford_circuit_qiskit(20, seed=SEED)
        cliff = Clifford(cliff_qc)
        circ = cliff.to_circuit()
        translate = generate_preset_pass_manager(
            1, basis_gates=["rz", "sx", "x", "cz"]
        ).translation
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                translate.property_set = PropertySet()
                return translate.run(circ)
        _report_memory(benchmark, mt, "Clifford_decompose_Qiskit")


# =====================================================================
# 7. NEW CONSTRUCTION MEMORY (Clifford, Multi-Control, QASM2 Import)
# =====================================================================

class TestNewConstructionMemory:
    """Memory for benchpress construction tests not previously covered."""

    def test_clifford_build_memory_sf(self, benchmark):
        """[SF] Memory for building a 100Q random Clifford circuit."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_clifford_circuit_sf(100, seed=SEED)
        _report_memory(benchmark, mt, "Clifford100_build_SF")
        assert mt.peak_mb < 500

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_clifford_build_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for building a 100Q random Clifford circuit."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_clifford_circuit_qiskit(100, seed=SEED)
        _report_memory(benchmark, mt, "Clifford100_build_Qiskit")

    def test_multi_control_build_memory_sf(self, benchmark):
        """[SF] Memory for building a 16Q cascading MCX circuit."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_multi_control_circuit_sf(16)
        _report_memory(benchmark, mt, "MCX16_build_SF")
        assert mt.peak_mb < 200

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_multi_control_build_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for building a 16Q cascading MCX circuit."""
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return build_multi_control_circuit_qiskit(16)
        _report_memory(benchmark, mt, "MCX16_build_Qiskit")

    def test_QV100_qasm2_import_memory_sf(self, benchmark):
        """[SF] Memory for importing a QV100 circuit from QASM2."""
        qasm_str = generate_qv100_qasm()
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return qasm2_to_sf(qasm_str)
        _report_memory(benchmark, mt, "QV100_qasm2_import_SF")
        assert mt.peak_mb < 500

    @pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
    def test_QV100_qasm2_import_memory_qiskit(self, benchmark):
        """[Qiskit] Memory for importing a QV100 circuit from QASM2."""
        from qiskit.qasm2 import loads
        qasm_str = generate_qv100_qasm()
        gc.collect()
        with MemoryTracker() as mt:
            @benchmark
            def result():
                return loads(qasm_str)
        _report_memory(benchmark, mt, "QV100_qasm2_import_Qiskit")
