"""
Claim Verification Test — SF vs Qiskit
=======================================
Verifies these specific claims:

CLAIM SET A — "Areas Where Qiskit Leads":
  A1. QV100 circuit memory:     Qiskit 67.7x better
  A2. DTC100 twirling memory:   Qiskit 17.0x better
  A3. Clifford decompose memory: Qiskit 16.0x better
  A4. DTC100 twirling latency:  Qiskit 3.5x faster
  A5. QV100 basis change:       Qiskit 1.5x faster
  A6. QV100 circuit build:      Qiskit 1.2x faster

CLAIM SET B — "Real gaps found":
  B1. QV simulation at n>=10: SF 1.7-3.2x slower (random SU(4) gates)
  B2. MCX simulation at n>=12: SF 37x slower at n=12, crashes at n=16
      (Rust backend bug: simulate_mps method missing)

Run:
    python tests/test_claim_verification.py
"""

import time
import gc
import tracemalloc
import sys
import os
import traceback

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import superfermion as sf
from superfermion.backends.registry import BackendRegistry

# ── Qiskit availability ──────────────────────────────────────────────────
try:
    import qiskit
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

SEED = 12345
N_WARMUP = 2
N_RUNS = 5


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def bench_latency(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Return (mean_ms, min_ms, max_ms) after warmup."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.min(times), np.max(times)


def bench_memory(fn, n_warmup=1, n_runs=3):
    """Return peak memory in MB."""
    for _ in range(n_warmup):
        fn()
    gc.collect()
    tracemalloc.start()
    for _ in range(n_runs):
        fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def probe_backends():
    """Return list of working SF backend names."""
    names = [
        "statevector", "rust", "mps", "jax", "jax_mps",
        "stabilizer", "density_matrix", "singularity", "supremacy",
    ]
    probe = sf.Circuit(2).h(0).cx(0, 1)
    working = []
    for n in names:
        try:
            BackendRegistry.get_backend(n)
            sf.run(probe, backend=n, shots=128)
            working.append(n)
        except Exception:
            pass
    return working


# ══════════════════════════════════════════════════════════════════════════
# CIRCUIT BUILDERS (mirrored from conftest)
# ══════════════════════════════════════════════════════════════════════════

def build_qv_sf(n_qubits, depth, seed=SEED):
    rng = np.random.default_rng(seed)
    circ = sf.Circuit(n_qubits)
    for _ in range(depth):
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            q0, q1 = int(perm[i]), int(perm[i + 1])
            angles = rng.uniform(0, 2 * np.pi, 4)
            circ.ry(float(angles[0]), q0)
            circ.ry(float(angles[1]), q1)
            circ.cx(q0, q1)
            circ.ry(float(angles[2]), q0)
            circ.ry(float(angles[3]), q1)
    return circ


def build_qv_qiskit(n_qubits, depth, seed=SEED):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            q0, q1 = int(perm[i]), int(perm[i + 1])
            angles = rng.uniform(0, 2 * np.pi, 4)
            qc.ry(float(angles[0]), q0)
            qc.ry(float(angles[1]), q1)
            qc.cx(q0, q1)
            qc.ry(float(angles[2]), q0)
            qc.ry(float(angles[3]), q1)
    return qc


def build_dtc_sf(n_qubits, n_cycles, g=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    circ = sf.Circuit(n_qubits)
    for _ in range(n_cycles):
        for q in range(n_qubits):
            circ.rx(float(np.pi * g), q)
        for q in range(n_qubits - 1):
            angle = float(rng.uniform(0, 2 * np.pi))
            circ.rzz(angle, q, q + 1)
    return circ


def build_clifford_sf(n_qubits, seed=SEED):
    from superfermion.circuit import GateRecord
    rng = np.random.default_rng(seed)
    c = sf.Circuit(n_qubits)
    num_gates = 10 * n_qubits * n_qubits
    GATES = ["CX", "CZ", "CY", "SWAP", "X", "Y", "Z", "S", "SDG", "H"]
    choices = rng.choice(len(GATES), size=num_gates)
    two_q_mask = choices < 4
    n_two_q = int(two_q_mask.sum())
    qa = rng.integers(0, n_qubits, size=n_two_q, dtype=np.int64)
    qb = rng.integers(0, n_qubits - 1, size=n_two_q, dtype=np.int64)
    qb[qb >= qa] += 1
    n_one_q = num_gates - n_two_q
    q1 = rng.integers(0, n_qubits, size=n_one_q, dtype=np.int64)
    records = [None] * num_gates
    ti = oi = 0
    for i in range(num_gates):
        gidx = choices[i]
        gname = GATES[gidx]
        if gidx < 4:
            records[i] = GateRecord(name=gname, qubits=[int(qa[ti]), int(qb[ti])])
            ti += 1
        else:
            records[i] = GateRecord(name=gname, qubits=[int(q1[oi])])
            oi += 1
    c._gates.extend(records)
    return c


def generate_qv100_qasm():
    from qiskit.circuit.library import quantum_volume
    from qiskit.qasm2 import dumps
    from qiskit import transpile
    qc = quantum_volume(100, 100, seed=SEED)
    qc_basis = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx'])
    return dumps(qc_basis)


def qasm2_to_sf(qasm_str):
    from superfermion.bridge import from_qasm
    return from_qasm(qasm_str)


# ══════════════════════════════════════════════════════════════════════════
# MAIN VERIFICATION
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  CLAIM VERIFICATION: SF vs QISKIT")
    print("=" * 80)
    print(f"  Qiskit available: {HAS_QISKIT}")
    if HAS_QISKIT:
        print(f"  Qiskit version:   {qiskit.__version__}")
    print(f"  SF version:       {sf.__version__}")

    # Probe backends
    working = probe_backends()
    print(f"  Working SF backends: {working}")
    print()

    results = {}

    # ====================================================================
    # A1: QV100 circuit memory — claim: Qiskit 67.7x
    # ====================================================================
    print("─" * 80)
    print("A1. QV100 CIRCUIT MEMORY (claim: Qiskit uses 67.7x less memory)")
    print("─" * 80)
    # SF uses Qiskit's quantum_volume() internally for QASM generation,
    # but for "QV100 circuit build" we compare our own builders
    mem_sf = bench_memory(lambda: build_qv_sf(100, 100, seed=SEED))
    print(f"  SF QV100 build memory:    {mem_sf:.2f} MB")

    if HAS_QISKIT:
        from qiskit.circuit.library import quantum_volume
        mem_qk = bench_memory(lambda: quantum_volume(100, 100, seed=SEED))
        print(f"  Qiskit QV100 build memory: {mem_qk:.2f} MB")
        ratio = mem_sf / mem_qk if mem_qk > 0 else float("inf")
        print(f"  RATIO (SF/Qiskit):         {ratio:.1f}x")
        print(f"  CLAIM says 67.7x → {'VERIFIED' if 50 < ratio < 90 else 'NOT VERIFIED'} (actual: {ratio:.1f}x)")
        results["A1_QV100_memory"] = {"sf_mb": mem_sf, "qk_mb": mem_qk, "ratio": ratio}
    else:
        print("  Qiskit not available, cannot verify")
        results["A1_QV100_memory"] = {"sf_mb": mem_sf, "qk_mb": None, "ratio": None}

    # Also check with gate-by-gate builder (same algo as SF)
    if HAS_QISKIT:
        mem_qk_gbg = bench_memory(lambda: build_qv_qiskit(100, 100, seed=SEED))
        print(f"  Qiskit gate-by-gate memory: {mem_qk_gbg:.2f} MB")
        ratio_gbg = mem_sf / mem_qk_gbg if mem_qk_gbg > 0 else float("inf")
        print(f"  RATIO (SF/Qiskit gbg):      {ratio_gbg:.1f}x")

    print()

    # ====================================================================
    # A2: DTC100 twirling memory — claim: Qiskit 17.0x
    # ====================================================================
    print("─" * 80)
    print("A2. DTC100 TWIRLING MEMORY (claim: Qiskit uses 17.0x less memory)")
    print("─" * 80)
    try:
        from superfermion.compiler.advanced import PauliTwirlingPass
        if HAS_QISKIT:
            qv_qasm = generate_qv100_qasm()

            def sf_twirl():
                circuit = qasm2_to_sf(qv_qasm)
                return PauliTwirlingPass(seed=SEED).run(circuit)

            def qk_twirl():
                from qiskit.circuit import pauli_twirl_2q_gates
                from qiskit.qasm2 import loads
                circuit = loads(qv_qasm)
                return pauli_twirl_2q_gates(circuit)

            mem_sf_tw = bench_memory(sf_twirl)
            mem_qk_tw = bench_memory(qk_twirl)
            ratio_tw = mem_sf_tw / mem_qk_tw if mem_qk_tw > 0 else float("inf")
            print(f"  SF twirling memory:    {mem_sf_tw:.2f} MB")
            print(f"  Qiskit twirling memory: {mem_qk_tw:.2f} MB")
            print(f"  RATIO (SF/Qiskit):      {ratio_tw:.1f}x")
            print(f"  CLAIM says 17.0x → {'VERIFIED' if 10 < ratio_tw < 25 else 'NOT VERIFIED'} (actual: {ratio_tw:.1f}x)")
            results["A2_DTC100_twirl_memory"] = {"sf_mb": mem_sf_tw, "qk_mb": mem_qk_tw, "ratio": ratio_tw}
    except Exception as e:
        print(f"  ERROR: {e}")
        results["A2_DTC100_twirl_memory"] = {"error": str(e)}

    print()

    # ====================================================================
    # A3: Clifford decompose memory — claim: Qiskit 16.0x
    # ====================================================================
    print("─" * 80)
    print("A3. CLIFFORD DECOMPOSE MEMORY (claim: Qiskit uses 16.0x less)")
    print("─" * 80)
    try:
        from superfermion.runtime.specs import HardwareSpec
        from superfermion.compiler.rust_bridge import compile_rust
        from superfermion.backends.stabilizer import simplify_clifford

        cliff_sf = build_clifford_sf(20, seed=SEED)
        simplified = simplify_clifford(cliff_sf)
        spec = HardwareSpec(
            name="clifford_decompose", n_qubits=20,
            native_gates=["rz", "sx", "x", "cz"], coupling_map=[],
        )

        def sf_cliff_decomp():
            return compile_rust(simplified, level=1, target=spec, pre_simplified=True)

        mem_sf_cd = bench_memory(sf_cliff_decomp)
        print(f"  SF Clifford decompose memory: {mem_sf_cd:.2f} MB")

        if HAS_QISKIT:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.passmanager import PropertySet
            from qiskit.quantum_info import Clifford
            from qiskit.circuit.random import random_clifford_circuit

            cliff_qc = random_clifford_circuit(20, gates=["cx","cz","cy","swap","x","y","z","s","sdg","h"],
                                                num_gates=10*20*20, seed=SEED)
            cliff_obj = Clifford(cliff_qc)
            circ_cliff = cliff_obj.to_circuit()
            translate = generate_preset_pass_manager(1, basis_gates=["rz", "sx", "x", "cz"]).translation

            def qk_cliff_decomp():
                translate.property_set = PropertySet()
                return translate.run(circ_cliff)

            mem_qk_cd = bench_memory(qk_cliff_decomp)
            ratio_cd = mem_sf_cd / mem_qk_cd if mem_qk_cd > 0 else float("inf")
            print(f"  Qiskit Clifford decompose memory: {mem_qk_cd:.2f} MB")
            print(f"  RATIO (SF/Qiskit):    {ratio_cd:.1f}x")
            print(f"  CLAIM says 16.0x → {'VERIFIED' if 8 < ratio_cd < 25 else 'NOT VERIFIED'} (actual: {ratio_cd:.1f}x)")
            results["A3_clifford_decompose_memory"] = {"sf_mb": mem_sf_cd, "qk_mb": mem_qk_cd, "ratio": ratio_cd}
        else:
            results["A3_clifford_decompose_memory"] = {"sf_mb": mem_sf_cd, "qk_mb": None, "ratio": None}
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["A3_clifford_decompose_memory"] = {"error": str(e)}

    print()

    # ====================================================================
    # A4: DTC100 twirling latency — claim: Qiskit 3.5x faster
    # ====================================================================
    print("─" * 80)
    print("A4. DTC100 TWIRLING LATENCY (claim: Qiskit 3.5x faster)")
    print("─" * 80)
    try:
        from superfermion.compiler.advanced import PauliTwirlingPass
        if HAS_QISKIT:
            qv_qasm = generate_qv100_qasm()
            circuit_sf = qasm2_to_sf(qv_qasm)
            from qiskit.circuit import pauli_twirl_2q_gates
            from qiskit.qasm2 import loads
            circuit_qk = loads(qv_qasm)

            lat_sf_tw, _, _ = bench_latency(lambda: PauliTwirlingPass(seed=SEED).run(circuit_sf))
            lat_qk_tw, _, _ = bench_latency(lambda: pauli_twirl_2q_gates(circuit_qk))
            ratio_tw_lat = lat_sf_tw / lat_qk_tw if lat_qk_tw > 0 else float("inf")
            print(f"  SF twirling latency:    {lat_sf_tw:.1f} ms")
            print(f"  Qiskit twirling latency: {lat_qk_tw:.1f} ms")
            print(f"  RATIO (SF/Qiskit):       {ratio_tw_lat:.1f}x (>1 means Qiskit faster)")
            print(f"  CLAIM says 3.5x → {'VERIFIED' if 2.0 < ratio_tw_lat < 5.0 else 'NOT VERIFIED'} (actual: {ratio_tw_lat:.1f}x)")
            results["A4_DTC100_twirl_latency"] = {"sf_ms": lat_sf_tw, "qk_ms": lat_qk_tw, "ratio": ratio_tw_lat}
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["A4_DTC100_twirl_latency"] = {"error": str(e)}

    print()

    # ====================================================================
    # A5: QV100 basis change — claim: Qiskit 1.5x faster
    # ====================================================================
    print("─" * 80)
    print("A5. QV100 BASIS CHANGE (claim: Qiskit 1.5x faster)")
    print("─" * 80)
    try:
        from superfermion.runtime.specs import HardwareSpec
        from superfermion.compiler.rust_bridge import compile_rust
        if HAS_QISKIT:
            qv_qasm = generate_qv100_qasm()
            circuit_sf = qasm2_to_sf(qv_qasm)
            spec = HardwareSpec(
                name="qv100_basis", n_qubits=circuit_sf.n_qubits,
                native_gates=["sx", "x", "rz", "cz"], coupling_map=[],
            )

            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.passmanager import PropertySet
            from qiskit.qasm2 import loads
            qv_circ_qk = loads(qv_qasm)
            translate = generate_preset_pass_manager(1, basis_gates=["sx", "x", "rz", "cz"]).translation

            lat_sf_bc, _, _ = bench_latency(lambda: compile_rust(circuit_sf, level=1, target=spec))
            def _qk_basis():
                translate.property_set = PropertySet()
                return translate.run(qv_circ_qk)
            lat_qk_bc, _, _ = bench_latency(_qk_basis)
            ratio_bc = lat_sf_bc / lat_qk_bc if lat_qk_bc > 0 else float("inf")
            print(f"  SF basis change latency:    {lat_sf_bc:.1f} ms")
            print(f"  Qiskit basis change latency: {lat_qk_bc:.1f} ms")
            print(f"  RATIO (SF/Qiskit):           {ratio_bc:.1f}x (>1 means Qiskit faster)")
            print(f"  CLAIM says 1.5x → {'VERIFIED' if 1.0 < ratio_bc < 2.5 else 'NOT VERIFIED'} (actual: {ratio_bc:.1f}x)")
            results["A5_QV100_basis_change"] = {"sf_ms": lat_sf_bc, "qk_ms": lat_qk_bc, "ratio": ratio_bc}
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["A5_QV100_basis_change"] = {"error": str(e)}

    print()

    # ====================================================================
    # A6: QV100 circuit build — claim: Qiskit 1.2x faster
    # ====================================================================
    print("─" * 80)
    print("A6. QV100 CIRCUIT BUILD (claim: Qiskit 1.2x faster)")
    print("─" * 80)
    lat_sf_build, _, _ = bench_latency(lambda: build_qv_sf(100, 100, seed=SEED))
    print(f"  SF QV100 build latency: {lat_sf_build:.1f} ms")
    if HAS_QISKIT:
        lat_qk_build, _, _ = bench_latency(lambda: build_qv_qiskit(100, 100, seed=SEED))
        ratio_build = lat_sf_build / lat_qk_build if lat_qk_build > 0 else float("inf")
        print(f"  Qiskit QV100 build latency: {lat_qk_build:.1f} ms")
        print(f"  RATIO (SF/Qiskit):          {ratio_build:.1f}x (>1 means Qiskit faster)")
        print(f"  CLAIM says 1.2x → {'VERIFIED' if 0.8 < ratio_build < 2.0 else 'NOT VERIFIED'} (actual: {ratio_build:.1f}x)")
        results["A6_QV100_build"] = {"sf_ms": lat_sf_build, "qk_ms": lat_qk_build, "ratio": ratio_build}
    else:
        results["A6_QV100_build"] = {"sf_ms": lat_sf_build, "qk_ms": None, "ratio": None}

    print()

    # ====================================================================
    # B1: QV simulation at n>=10 — claim: SF 1.7-3.2x slower
    # ====================================================================
    print("─" * 80)
    print("B1. QV SIMULATION n>=10 (claim: SF 1.7-3.2x slower)")
    print("─" * 80)
    results["B1_QV_simulation"] = {}

    sf_sim_backends = [b for b in ["statevector", "rust", "jax"] if b in working]
    print(f"  Testing SF backends: {sf_sim_backends}")

    for n in [10, 12, 14]:
        print(f"\n  --- n={n} qubits ---")
        circ_sf = build_qv_sf(n, depth=n, seed=SEED)

        for bname in sf_sim_backends:
            try:
                lat_sf_sim, _, _ = bench_latency(
                    lambda b=bname: sf.run(circ_sf, backend=b, shots=1024),
                    n_warmup=2, n_runs=3,
                )
                print(f"    SF [{bname:12s}]: {lat_sf_sim:.1f} ms")
            except Exception as e:
                print(f"    SF [{bname:12s}]: ERROR — {str(e)[:60]}")
                lat_sf_sim = None

        if HAS_QISKIT and n <= 20:
            from qiskit import QuantumCircuit, transpile
            qc_sim = build_qv_qiskit(n, depth=n, seed=SEED)
            qc_sim.measure_all()
            sim = AerSimulator(method="statevector")
            tqc_sim = transpile(qc_sim, sim)
            try:
                lat_qk_sim, _, _ = bench_latency(
                    lambda: sim.run(tqc_sim, shots=1024).result(),
                    n_warmup=2, n_runs=3,
                )
                print(f"    Qiskit Aer:        {lat_qk_sim:.1f} ms")

                # Record ratios for each SF backend
                for bname in sf_sim_backends:
                    try:
                        lat_sf_b, _, _ = bench_latency(
                            lambda b=bname: sf.run(circ_sf, backend=b, shots=1024),
                            n_warmup=1, n_runs=2,
                        )
                        ratio_sim = lat_sf_b / lat_qk_sim if lat_qk_sim > 0 else float("inf")
                        key = f"n{n}_{bname}"
                        results["B1_QV_simulation"][key] = {
                            "sf_ms": lat_sf_b, "qk_ms": lat_qk_sim, "ratio": ratio_sim
                        }
                        verdict = "VERIFIED" if 1.5 < ratio_sim < 4.0 else "NOT VERIFIED"
                        print(f"    → Ratio SF/{bname}/Qiskit: {ratio_sim:.1f}x  {verdict}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"    Qiskit Aer: ERROR — {str(e)[:60]}")

    print()

    # ====================================================================
    # B2: MCX simulation at n>=12 — claim: SF 37x slower at n=12,
    #     crashes at n=16 (simulate_mps missing)
    # ====================================================================
    print("─" * 80)
    print("B2. MCX SIMULATION n>=12 (claim: SF 37x slower at n=12, crash at n=16)")
    print("─" * 80)
    results["B2_MCX_simulation"] = {}

    from tests.benchpress.conftest import build_multi_control_circuit_sf, build_multi_control_circuit_qiskit

    for n in [10, 12, 14, 16]:
        print(f"\n  --- n={n} qubits ---")

        # SF
        try:
            mcx_sf = build_multi_control_circuit_sf(n)
            print(f"    SF circuit: {mcx_sf.n_qubits} qubits, {mcx_sf.gate_count} gates")

            for bname in sf_sim_backends:
                try:
                    lat_sf_mcx, _, _ = bench_latency(
                        lambda b=bname: sf.run(mcx_sf, backend=b, shots=1024),
                        n_warmup=1, n_runs=2,
                    )
                    print(f"    SF [{bname:12s}]: {lat_sf_mcx:.1f} ms")
                    results["B2_MCX_simulation"][f"n{n}_{bname}"] = {"sf_ms": lat_sf_mcx}
                except Exception as e:
                    print(f"    SF [{bname:12s}]: ERROR — {str(e)[:80]}")
                    results["B2_MCX_simulation"][f"n{n}_{bname}"] = {"error": str(e)[:80]}

            # Check simulate_mps specifically
            try:
                from superfermion._sf_core import QuantumDAG
                dag = QuantumDAG.from_circuit(mcx_sf)
                dag.simulate_mps(32)
                print(f"    simulate_mps: OK")
            except AttributeError as ae:
                print(f"    simulate_mps: MISSING — {str(ae)[:80]}")
                results["B2_MCX_simulation"][f"n{n}_simulate_mps"] = {"error": str(ae)[:80]}
            except Exception as e:
                print(f"    simulate_mps: ERROR — {str(e)[:80]}")

        except Exception as e:
            print(f"    SF build ERROR: {str(e)[:80]}")

        # Qiskit
        if HAS_QISKIT:
            try:
                mcx_qk = build_multi_control_circuit_qiskit(n)
                mcx_qk.measure_all()
                sim = AerSimulator(method="statevector")
                tqc_mcx = transpile(mcx_qk, sim)
                lat_qk_mcx, _, _ = bench_latency(
                    lambda: sim.run(tqc_mcx, shots=1024).result(),
                    n_warmup=1, n_runs=2,
                )
                print(f"    Qiskit Aer:        {lat_qk_mcx:.1f} ms")
                results["B2_MCX_simulation"][f"n{n}_qiskit"] = {"qk_ms": lat_qk_mcx}

                # Compare with best SF backend
                for bname in sf_sim_backends:
                    key = f"n{n}_{bname}"
                    if key in results["B2_MCX_simulation"] and "sf_ms" in results["B2_MCX_simulation"][key]:
                        sf_ms = results["B2_MCX_simulation"][key]["sf_ms"]
                        ratio_mcx = sf_ms / lat_qk_mcx if lat_qk_mcx > 0 else float("inf")
                        results["B2_MCX_simulation"][key]["qk_ms"] = lat_qk_mcx
                        results["B2_MCX_simulation"][key]["ratio"] = ratio_mcx
                        print(f"    → Ratio SF/{bname}/Qiskit: {ratio_mcx:.1f}x")
            except Exception as e:
                print(f"    Qiskit Aer: ERROR — {str(e)[:80]}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n")
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()
    print("CLAIM SET A — 'Areas Where Qiskit Leads':")
    print(f"  {'Test':<35s} {'Claim':<12s} {'Actual':<12s} {'Verdict'}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*15}")

    checks = [
        ("A1 QV100 circuit memory",  "67.7x", "A1_QV100_memory"),
        ("A2 DTC100 twirling memory", "17.0x", "A2_DTC100_twirl_memory"),
        ("A3 Clifford decompose mem", "16.0x", "A3_clifford_decompose_memory"),
        ("A4 DTC100 twirling latency", "3.5x",  "A4_DTC100_twirl_latency"),
        ("A5 QV100 basis change",     "1.5x",  "A5_QV100_basis_change"),
        ("A6 QV100 circuit build",    "1.2x",  "A6_QV100_build"),
    ]

    for label, claim, key in checks:
        r = results.get(key, {})
        if "error" in r:
            print(f"  {label:<35s} {claim:<12s} {'ERROR':<12s} {r['error'][:30]}")
        elif r.get("ratio") is not None:
            actual = f"{r['ratio']:.1f}x"
            print(f"  {label:<35s} {claim:<12s} {actual:<12s} (sf={r.get('sf_mb', r.get('sf_ms', '?')):.1f}, qk={r.get('qk_mb', r.get('qk_ms', '?')):.1f})")
        else:
            print(f"  {label:<35s} {claim:<12s} {'N/A':<12s}")

    print()
    print("CLAIM SET B — 'Real gaps found':")
    print(f"  {'Test':<35s} {'Claim':<15s} {'Actual':<12s}")
    print(f"  {'─'*35} {'─'*15} {'─'*12}")

    # B1: QV simulation
    b1 = results.get("B1_QV_simulation", {})
    for key, val in sorted(b1.items()):
        if "ratio" in val:
            print(f"  B1 QV sim {key:<24s} {'1.7-3.2x slower':<15s} {val['ratio']:.1f}x")

    # B2: MCX simulation
    b2 = results.get("B2_MCX_simulation", {})
    for key, val in sorted(b2.items()):
        if "ratio" in val:
            print(f"  B2 MCX sim {key:<23s} {'37x slower':<15s} {val['ratio']:.1f}x")
        elif "error" in val and "n16" in key:
            print(f"  B2 MCX sim {key:<23s} {'crash at n=16':<15s} ERROR: {val['error'][:40]}")

    print()
    print("=" * 80)
    print("  RAW RESULTS DICT (for inspection)")
    print("=" * 80)
    import json
    # Sanitize for JSON
    safe_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            safe_results[k] = {
                sk: (sv if isinstance(sv, (int, float, str, bool, type(None)))
                     else str(sv))
                for sk, sv in v.items()
            }
        else:
            safe_results[k] = str(v)
    print(json.dumps(safe_results, indent=2))


if __name__ == "__main__":
    main()
