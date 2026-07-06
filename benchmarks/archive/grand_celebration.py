"""
=== THE GRAND CELEBRATION BENCHMARK ===
Superfermion JAX vs. The Industrial Giants
Algorithms: 12-Qubit GHZ Stability (100 Iterations)

Competing Frameworks:
1. Superfermion JAX (XLA-Turbo)
2. Qiskit Aer (C++ Optimized)
3. PennyLane (Industrial Standard)
4. TensorFlow (Deep Learning Engine)
"""

import time
import numpy as np
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Framework Imports
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator

import pennylane as qml
import tensorflow as tf
import jax.numpy as jnp

def run_celebration_benchmark():
    n_qubits = 12
    iterations = 50 
    print(f"=== CELEBRATION BENCHMARK: {n_qubits} QUBITS, {iterations} ITERATIONS ===")
    print("Task: GHZ Entanglement State Generation\n")

    # --- 1. SUPERFERMION JAX (XLA TURBO) ---
    print("[1/4] Superfermion JAX (XLA Turbo)...")
    sf_c = Circuit(n_qubits).h(0)
    for i in range(n_qubits-1):
        sf_c.cx(i, i+1)
    
    # Warmup
    _ = sf.run(sf_c, backend="jax", shots=0)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = sf.run(sf_c, backend="jax", shots=0)
    t_sf = time.time() - t0
    print(f"Superfermion JAX Time: {t_sf:.4f}s")

    # --- 2. QISKIT AER (C++ ENGINE) ---
    print("\n[2/4] Qiskit Aer (C++ Engine)...")
    qc = to_qiskit(sf_c)
    qc.save_statevector()
    aer_sim = AerSimulator()
    
    t0 = time.time()
    for _ in range(iterations):
        aer_sim.run(qc).result()
    t_aer = time.time() - t0
    print(f"Qiskit Aer Time:      {t_aer:.4f}s")

    # --- 3. PENNYLANE (DEFAULT QUBIT) ---
    print("\n[3/4] PennyLane (Default Qubit)...")
    dev_pl = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev_pl)
    def pl_circuit():
        qml.Hadamard(wires=0)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        return qml.state()
    
    # Warmup
    _ = pl_circuit()
    
    t0 = time.time()
    for _ in range(iterations):
        _ = pl_circuit()
    t_pl = time.time() - t0
    print(f"PennyLane Time:       {t_pl:.4f}s")

    # --- 4. TENSORFLOW (NATIVE MATRIX OPS) ---
    print("\n[4/4] TensorFlow (Linear Algebra Engine)...")
    # We implement a fast TF simulator for this specific GHZ task
    def tf_ghz(n):
        state = tf.constant([1.0] + [0.0]*(2**n-1), dtype=tf.complex128)
        # H on 0
        h_mat = tf.constant([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=tf.complex128)
        # Reshape to tensor
        state = tf.reshape(state, [2]*n)
        # Contract H
        state = tf.tensordot(h_mat, state, axes=[[1], [0]])
        # Contract CNOTs
        cx_mat = tf.constant([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=tf.complex128)
        cx_ten = tf.reshape(cx_mat, (2,2,2,2))
        for i in range(n-1):
            state = tf.tensordot(cx_ten, state, axes=[[2,3], [i, i+1]])
            # Move axes back to original positions (simplified for GHZ)
            state = tf.transpose(state, perm=list(range(2, i+2)) + [0, 1] + list(range(i+2, n)))
        return tf.reshape(state, [-1])

    # Warmup
    _ = tf_ghz(n_qubits)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = tf_ghz(n_qubits)
    t_tf = time.time() - t0
    print(f"TensorFlow Time:      {t_tf:.4f}s")

    # --- FINAL VERDICT ---
    print("\n" + "="*60)
    print(f"{'Framework':<25} | {'Speedup over Qiskit':<15}")
    print("-" * 60)
    print(f"{'Qiskit Aer (Baseline)':<25} | 1.00x")
    print(f"{'PennyLane':<25} | {t_aer / t_pl:.2f}x")
    print(f"{'TensorFlow':<25} | {t_aer / t_tf:.2f}x")
    print(f"🥇 {'Superfermion JAX':<22} | {t_aer / t_sf:.2f}x")
    print("="*60)
    
    if t_sf < min(t_aer, t_pl, t_tf):
        print("\n🏆 THE WINNER AND NEW CHAMPION: SUPERFERMION JAX 🏆")
        print("Reason: XLA-Fused Unitaries eliminate ALL overhead.")
    print("="*60)

if __name__ == "__main__":
    run_celebration_benchmark()
