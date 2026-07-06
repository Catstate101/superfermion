"""
SUPERFERMION COMPARATIVE BENCHMARK: FULL SPECTRUM AUDIT
======================================================
Backends: Simulator (SV), MPS, JAX
Ground Truth: Qiskit Aer (SV/MPS)
Targets: Accuracy (TVD/Fid), Latency (ns), Memory (KB)
"""
import numpy as np
import time, tracemalloc, sys, os
import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# --- UTILS: MIRRORED QISKIT CONSTRUCTION ---
def build_mirrored_qc(n, builder_fn):
    qc = QuantumCircuit(n)
    def m(i): return n - 1 - i
    builder_fn(qc, m)
    return qc

def get_qk_sv(n, builder_fn):
    qc = build_mirrored_qc(n, builder_fn)
    qc.save_statevector()
    res = AerSimulator(method='statevector').run(qc).result()
    return np.array(res.get_statevector())

def get_qk_mps_counts(n, builder_fn, shots=1024, bond_dim=64):
    qc = build_mirrored_qc(n, builder_fn)
    qc.measure_all()
    sim = AerSimulator(method='matrix_product_state')
    sim.set_options(matrix_product_state_max_bond_dimension=bond_dim)
    res = sim.run(qc, shots=shots).result()
    # No reversal needed because builder was mirrored
    return res.get_counts()

# --- METRICS ---
def compute_tvd(c1, c2):
    tot1 = sum(c1.values()); tot2 = sum(c2.values())
    if tot1 == 0 or tot2 == 0: return 1.0
    keys = set(c1.keys()) | set(c2.keys())
    return 0.5 * sum([abs(c1.get(k, 0)/tot1 - c2.get(k, 0)/tot2) for k in keys])

def compute_fid(sv1, sv2):
    return float(np.abs(np.vdot(sv1, sv2))**2)

def audit_step(name, sf_backend, n, builder_fn, shots=1024, ground_truth=None):
    # Initialize SF circuit
    c = sf.Circuit(n)
    builder_fn(c, lambda i: i)
    
    # 2. SF Run
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    res_sf = sf.run(c, backend=sf_backend, shots=shots)
    lat_sf = time.perf_counter_ns() - t0
    _, peak_sf = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 3. Ground Truth calculation with TIMING
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    acc = 0.0
    if shots == 0:
        sv_qk = get_qk_sv(n, builder_fn)
        lat_qk = time.perf_counter_ns() - t0
        _, peak_qk = tracemalloc.get_traced_memory()
        acc = compute_fid(res_sf.statevector, sv_qk)
    else:
        # For fair comparison, we use Qiskit's equivalent simulator
        if n <= 15: # Use SV for small ones
            counts_qk = get_qk_mps_counts(n, builder_fn, shots=shots, bond_dim=1024)
        else: # Use MPS for larger ones
            counts_qk = get_qk_mps_counts(n, builder_fn, shots=shots, bond_dim=64)
        lat_qk = time.perf_counter_ns() - t0
        _, peak_qk = tracemalloc.get_traced_memory()
        acc = 1.0 - compute_tvd(res_sf.counts, counts_qk)
            
    return {
        "lat_sf": lat_sf, "mem_sf": peak_sf, 
        "lat_qk": lat_qk, "mem_qk": peak_qk, 
        "accuracy": acc
    }

if __name__ == "__main__":
    print("="*105)
    print(f"{'SF vs QISKIT AER: FULL COMPARATIVE SPECTRUM':^105}")
    print("="*105)
    print(f"{'Circuit':<12} | {'Backend':<10} | {'Acc':<6} | {'SF Lat (ms)':<12} | {'QK Lat (ms)':<12} | {'SF Mem (KB)':<12} | {'QK Mem (KB)':<12}")
    print("-" * 105)

    # 1. GHZ-12 (Mid-scale)
    def ghz_12_builder(obj, m):
        if hasattr(obj, "h"): # sf
            obj.h(m(0))
            for i in range(11): obj.cx(m(i), m(i+1))
        else: # qk
            obj.h(m(0))
            for i in range(11): obj.cx(m(i), m(i+1))

    for b in ["simulator", "mps", "jax"]:
        # Warmup for JAX
        if b == "jax": sf.run(sf.Circuit(12).h(0), backend="jax", shots=0)
        
        m = audit_step("GHZ-12", b, 12, ghz_12_builder, shots=1024)
        print(f"{'GHZ-12':<12} | {b:<10} | {m['accuracy']:<6.4f} | {m['lat_sf']/1e6:<12.2f} | {m['lat_qk']/1e6:<12.2f} | {m['mem_sf']/1024:<12.2f} | {m['mem_qk']/1024:<12.2f}")


    # 2. Random Circ 20 Qubits (Large scale for MPS)
    def rand_20_builder(obj, m):
        import random; r = random.Random(123)
        for i in range(20): obj.h(m(i))
        for i in range(30): obj.rx(r.uniform(0, 0.5), m(i % 20))
        for i in range(0, 19, 2): obj.cx(m(i), m(i+1))

    for b in ["mps", "jax"]:
        m = audit_step("Rand-20", b, 20, rand_20_builder, shots=1024)
        print(f"{'Rand-20':<12} | {b:<10} | {m['accuracy']:<6.4f} | {m['lat_sf']/1e6:<12.2f} | {m['lat_qk']/1e6:<12.2f} | {m['mem_sf']/1024:<12.2f} | {m['mem_qk']/1024:<12.2f}")

    print("-" * 105)

    # 3. Random Circ 32 Qubits (Very large scale)
    def rand_32_builder(obj, m):
        import random; r = random.Random(456)
        for i in range(32): obj.h(m(i))
        for i in range(32): obj.ry(r.uniform(0, 0.5), m(i))
        for i in range(0, 31, 2): obj.cx(m(i), m(i+1))

    m = audit_step("Rand-32", "mps", 32, rand_32_builder, shots=1024)
    print(f"{'Rand-32':<12} | {'mps':<10} | {m['accuracy']:<6.4f} | {m['lat_sf']/1e6:<12.2f} | {m['lat_qk']/1e6:<12.2f} | {m['mem_sf']/1024:<12.2f} | {m['mem_qk']/1024:<12.2f}")
    print("=" * 105)
