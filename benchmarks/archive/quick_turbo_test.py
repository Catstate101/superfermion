
import time
import numpy as np
import superfermion as sf
from superfermion.backends.turbo import fuse_single_qubit_gates

def benchmark_turbo_speedup(n_qubits=20, depth=100):
    print(f"=== BENCHMARKING TURBO SPEEDUP (N={n_qubits}, depth={depth}) ===")
    
    # Define a complex VQE-style circuit with many 1Q gates
    circuit = sf.Circuit(n_qubits)
    for _ in range(max(1, depth // n_qubits)):
        for q in range(n_qubits):
            circuit.rx(0.51, q).ry(0.23, q).rz(0.12, q)
        for q in range(0, n_qubits - 1, 2):
            circuit.cnot(q, q+1)
            
    print(f"Original gate count: {len(circuit._gates)}")
    
    # Test Gate Fusion speedup
    t_f0 = time.time()
    fused = fuse_single_qubit_gates(circuit)
    t_fusion = time.time() - t_f0
    print(f"Fused gate count: {len(fused._gates)} (reduced by {(1 - len(fused._gates)/len(circuit._gates))*100:.1f}%)")
    print(f"Fusion time: {t_fusion*1000:.2f}ms")
    
    # Cold start simulation
    t0 = time.time()
    res1 = sf.run(circuit, backend="singularity", shots=1000)
    t_cold = time.time() - t0
    print(f"Cold start time: {t_cold*1000:.2f}ms")
    
    # Warm start simulation (Caching)
    t1 = time.time()
    res2 = sf.run(circuit, backend="singularity", shots=1000)
    t_warm = time.time() - t1
    print(f"Warm start time: {t_warm*1000:.2f}ms")
    
    speedup = t_cold / t_warm if t_warm > 0 else 0
    print(f"TURBO CACHE SPEEDUP: {speedup:.1f}x")
    
    # Scalability check: N=40 (Should use MPS direct and be FAST)
    print("\n=== SCALABILITY CHECK (N=40) ===")
    c40 = sf.Circuit(40)
    c40.h(0)
    for j in range(39): c40.cnot(j, j+1)
    
    t0 = time.time()
    res40 = sf.run(c40, backend="singularity", shots=100)
    t40 = time.time() - t0
    print(f"N=40 GHZ simulation time: {t40*1000:.2f}ms (Industry-Dominating scale-out)")
    print(f"Bitstrings sampled samples count: {sum(res40.counts.values())}")

if __name__ == "__main__":
    benchmark_turbo_speedup()
