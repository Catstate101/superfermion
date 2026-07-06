"""
FINAL INDUSTRIAL VALIDATION REDO
================================
Validating SuperFermion against Qiskit Aer Ground Truths.
Fixes: 2-qubit gate index bug, MPS dynamic truncation, Endianness alignment.
"""
import numpy as np
import time, tracemalloc, sys, os
import superfermion as sf
from superfermion.simulator import simulate_statevector
from superfermion.backends.mps import MPSSimulatorBackend
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def get_qk_sv(qc_fn, n):
    qc = QuantumCircuit(n)
    qc_fn(qc)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    sv_qk = np.array(sim.run(qc).result().get_statevector())
    # Qiskit (LE) to SF (BE) conversion
    sv_tensor = sv_qk.reshape([2] * n)
    sv_tensor = sv_tensor.transpose(list(range(n-1, -1, -1)))
    return sv_tensor.flatten()

def get_qk_mps_counts(qc_fn, n, shots=4096, bond_dim=32):
    qc = QuantumCircuit(n)
    qc_fn(qc)
    qc.measure_all()
    sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=bond_dim)
    res = sim.run(qc, shots=shots).result()
    counts_le = res.get_counts()
    # Reverse bitstrings for BE
    return {k[::-1]: v for k, v in counts_le.items()}

def compute_fidelity(sv1, sv2):
    return float(np.abs(np.vdot(sv1, sv2))**2)

def compute_tvd(c1, c2, n):
    tot1 = sum(c1.values()); tot2 = sum(c2.values())
    keys = set(c1.keys()) | set(c2.keys())
    tvd = 0.5 * sum([abs(c1.get(k, 0)/tot1 - c2.get(k, 0)/tot2) for k in keys])
    return tvd

def benchmark(name, n, sf_fn, qk_fn, test_type='sv'):
    print(f"--- {name} ({n} qubits) ---")
    
    # SF run
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    res_sf = sf_fn()
    lat_sf = time.perf_counter_ns() - t0
    _, peak_sf = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # QK run
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    res_qk = qk_fn()
    lat_qk = time.perf_counter_ns() - t0
    _, peak_qk = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if test_type == 'sv':
        fid = compute_fidelity(res_sf, res_qk)
        print(f"  Fidelity: {fid:.10f}")
        passed = fid > 0.99999
    else:
        tvd = compute_tvd(res_sf.counts, res_qk, n)
        print(f"  TVD:      {tvd:.6f}")
        passed = tvd < 0.05
        
    print(f"  Latency: SF={lat_sf:>12,} ns | QK={lat_qk:>12,} ns")
    print(f"  Memory:  SF={peak_sf/1024:>10.2f} KB | QK={peak_qk/1024:>10.2f} KB")
    return passed

if __name__ == "__main__":
    print("==============================================================================")
    print("INDUSTRIAL VALIDATION REDO: QISKIT GROUND TRUTH PARITY")
    print("==============================================================================")
    
    # Test 01: Multi-Qubit Gates (Testing the fix for Q1 > Q0)
    # Circuit: H(0), CNOT(1, 0)
    print("\n[TEST 01] ASYMMETRIC GATE PHYSICS (CNOT 1->0)")
    def build_01(c): c.h(1).cx(1, 0)
    def sf_01(): return simulate_statevector(sf.Circuit(2).h(1).cx(1, 0))
    def qk_01(): return get_qk_sv(lambda qc: (qc.h(1), qc.cx(1, 0)), 2)
    p1 = benchmark("CNOT(1,0)", 2, sf_01, qk_01, 'sv')

    # Test 02: GHZ-10 Accuracy
    print("\n[TEST 02] GHZ-10 ENTANGLEMENT")
    def build_ghz(c, n):
        c.h(0)
        for i in range(n-1): c.cx(i, i+1)
    def sf_02(): return simulate_statevector(sf.Circuit(10).h(0).cx(0,1).cx(1,2).cx(2,3).cx(3,4).cx(4,5).cx(5,6).cx(6,7).cx(7,8).cx(8,9))
    def qk_02(): return get_qk_sv(lambda qc: (qc.h(0), [qc.cx(i,i+1) for i in range(9)]), 10)
    p2 = benchmark("GHZ-10", 10, sf_02, qk_02, 'sv')

    # Test 03: MPS Accuracy & Memory Scaling
    # Using dynamic truncation fix
    print("\n[TEST 03] MPS SCALING & DYNAMIC TRUNCATION (20 Qubits)")
    def build_ladder(c, n):
        for i in range(n): c.h(i)
        for i in range(n-1): c.cx(i, i+1)
    def sf_03(): 
        c = sf.Circuit(20)
        for i in range(20): c.h(i)
        for i in range(19): c.cx(i, i+1)
        sim = MPSSimulatorBackend(options={"max_bond_dim": 32})
        return sim.run(c, shots=1024)
    def qk_03(): return get_qk_mps_counts(lambda qc: ( [qc.h(i) for i in range(20)], [qc.cx(i,i+1) for i in range(19)]), 20, 1024, 32)
    p3 = benchmark("Ladder-20 MPS", 20, sf_03, qk_03, 'mps')

    # Test 04: JAX Optimization (Minimal latency check)
    print("\n[TEST 04] JAX KERNEL LATENCY (XLA vs Qiskit Cache)")
    # Warmup first to subtract JIT time if we want to show raw speed
    f_jax = sf.qml.circuit_to_jax(sf.Circuit(2).h(0).cx(0,1))
    f_jax(1.0) # Warmup
    def sf_04(): return f_jax(1.0)
    def qk_04(): 
        qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1); qc.save_statevector()
        return AerSimulator().run(qc).result().get_statevector()
    p4 = benchmark("JAX 2q", 2, sf_04, qk_04, 'sv')

    all_p = p1 and p2 and p3 and p4
    print("\n" + "=" * 78)
    print(f"REDO RESULT: {'PASS' if all_p else 'FAIL'}")
    print("=" * 78)
    sys.exit(0 if all_p else 1)
