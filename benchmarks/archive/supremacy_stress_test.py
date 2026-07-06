"""
=== THE ULTIMATE SUPREMACY TEST: RANDOM CIRCUIT SAMPLING (RCS) ===
This test simulates random quantum circuits (Sycamore-style) 
to push the Superfermion JAX Engine to its absolute hardware limit.
Comparing against Qiskit Aer (C++).
"""

import time
import numpy as np
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator

def create_random_circuit(n, depth):
    c = Circuit(n)
    for d in range(depth):
        # 1. Single Qubit Layer
        for i in range(n):
            gate = np.random.choice(["h", "rx", "ry", "rz"])
            if gate == "h": c.h(i)
            elif gate == "rx": c.rx(np.random.rand() * np.pi, i)
            elif gate == "ry": c.ry(np.random.rand() * np.pi, i)
            elif gate == "rz": c.rz(np.random.rand() * np.pi, i)
        
        # 2. Entangling Layer (Randomized Couplers)
        pairs = list(range(n))
        np.random.shuffle(pairs)
        for i in range(0, n-1, 2):
            c.cx(pairs[i], pairs[i+1])
    return c

def ultimate_supremacy_test():
    N_QUBITS = 15 # Pushing towards memory limits
    DEPTH = 20
    print(f"=== QUANTUM SUPREMACY STRESS TEST: {N_QUBITS} QUBITS, {DEPTH} LAYERS ===")
    
    rcs_c = create_random_circuit(N_QUBITS, DEPTH)
    print(f"Generated Random Circuit with {len(rcs_c._gates)} gates.")

    # 1. Qiskit Aer (C++)
    print("\n[1/2] Computing with Qiskit Aer...")
    qc = to_qiskit(rcs_c)
    qc.save_statevector()
    aer_sim = AerSimulator()
    
    t0 = time.time()
    _ = aer_sim.run(qc).result()
    t_aer = time.time() - t0
    print(f"Qiskit Aer Time: {t_aer:.4f}s")

    # 2. Superfermion JAX (XLA)
    print("\n[2/2] Computing with Superfermion JAX...")
    # First Run (Warmup + JIT)
    t1 = time.time()
    _ = sf.run(rcs_c, backend="jax", shots=0)
    t_sf_first = time.time() - t1
    print(f"SF JAX First Run (JIT + Bake): {t_sf_first:.4f}s")
    
    # Cached Run
    t2 = time.time()
    _ = sf.run(rcs_c, backend="jax", shots=0)
    t_sf_cached = time.time() - t2
    print(f"SF JAX Cached Run:            {t_sf_cached:.4f}s")

    print("\n" + "="*50)
    print(f"CACHED SPEEDUP: {t_aer / t_sf_cached:.2f}x")
    if t_sf_cached < t_aer:
        print("STATUS: TOTAL SUPERFERMION DOMINANCE")
    print("="*50)

if __name__ == "__main__":
    ultimate_supremacy_test()
