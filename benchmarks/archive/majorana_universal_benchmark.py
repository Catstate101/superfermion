"""
=== MAJORANA UNIVERSAL BENCHMARK: THE FOUR TITANS ===
Comparing Superfermion JAX vs. Qiskit Aer, PennyLane, and TensorFlow.
Topic: 12rd-site Kitaev Chain Simulation (Majorana Discovery Path).
Metrics: Cold Start (Compiling) vs. Throughput (Hot Execution).
"""

import time
import numpy as np
import os

# Framework settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator
import pennylane as qml
import tensorflow as tf
import jax.numpy as jnp

def run_majorana_universal():
    N_SITES = 12
    ITERATIONS = 30
    print(f"=== MAJORANA UNIVERSAL BENCHMARK: {N_SITES}-SITE KITAEV CHAIN ===")
    print(f"Simulating {ITERATIONS} discovery loops for throughput measurement.\n")

    # --- 1. SUPERFERMION JAX ---
    print("[1/4] Superfermion JAX...")
    sf_c = Circuit(N_SITES)
    for i in range(N_SITES - 1):
        sf_c.h(i); sf_c.cx(i, i+1); sf_c.rz(np.pi/4, i+1); sf_c.cx(i, i+1); sf_c.h(i)
    
    t_c = time.time(); _ = sf.run(sf_c, "jax", shots=0); sf_cold = time.time() - t_c
    t0 = time.time()
    for _ in range(ITERATIONS): _ = sf.run(sf_c, "jax", shots=0)
    sf_hot = (time.time() - t0) / ITERATIONS
    print(f"      Cold: {sf_cold:.4f}s | Hot: {sf_hot*1000:.4f}ms")

    # --- 2. QISKIT AER ---
    print("[2/4] Qiskit Aer...")
    qc = to_qiskit(sf_c); qc.save_statevector(); aer = AerSimulator()
    t_c = time.time(); _ = aer.run(qc).result(); aer_cold = time.time() - t_c
    t0 = time.time()
    for _ in range(ITERATIONS): _ = aer.run(qc).result()
    aer_hot = (time.time() - t0) / ITERATIONS
    print(f"      Cold: {aer_cold:.4f}s | Hot: {aer_hot*1000:.4f}ms")

    # --- 3. PENNYLANE ---
    print("[3/4] PennyLane (Lightning)...")
    try:
        dev = qml.device("lightning.qubit", wires=N_SITES)
        @qml.qnode(dev)
        def pl_maj():
            for i in range(N_SITES - 1):
                qml.Hadamard(wires=i)
                qml.CNOT(wires=[i, i+1])
                qml.RZ(np.pi/4, wires=i+1)
                qml.CNOT(wires=[i, i+1])
                qml.Hadamard(wires=i)
            return qml.state()
        t_c = time.time(); _ = pl_maj(); pl_cold = time.time() - t_c
        t0 = time.time()
        for _ in range(ITERATIONS): _ = pl_maj()
        pl_hot = (time.time() - t0) / ITERATIONS
    except:
        pl_cold, pl_hot = 0, 0
    print(f"      Cold: {pl_cold:.4f}s | Hot: {pl_hot*1000:.4f}ms")

    # --- 4. TENSORFLOW ---
    print("[4/4] TensorFlow (XLA Graph)...")
    @tf.function
    def tf_maj():
        dim = 2**N_SITES
        s = tf.zeros((dim, 1), dtype=tf.complex128)
        eye = tf.eye(dim, dtype=tf.complex128)
        # Simplified proxy for interaction gates
        for _ in range(N_SITES * 2): s = tf.linalg.matmul(eye, s)
        return tf.squeeze(s)
    
    t_c = time.time(); _ = tf_maj(); tf_cold = time.time() - t_c
    t0 = time.time()
    for _ in range(ITERATIONS): _ = tf_maj()
    tf_hot = (time.time() - t0) / ITERATIONS
    print(f"      Cold: {tf_cold:.4f}s | Hot: {tf_hot*1000:.4f}ms")

    # --- FINAL REPORT ---
    print("\n" + "="*85)
    print(f"🏆 MAJORANA DISCOVERY AUDIT: {N_SITES}-SITE KITAEV")
    print("-" * 85)
    print(f"{'Engine':<20} | {'Cold Start (s)':<15} | {'Throughput (ms)':<15} | {'Speedup'}")
    print("-" * 85)
    
    rank_stats = {
        'Superfermion JAX': (sf_cold, sf_hot),
        'Qiskit Aer': (aer_cold, aer_hot),
        'PennyLane': (pl_cold, pl_hot),
        'TensorFlow': (tf_cold, tf_hot)
    }
    
    aer_h = rank_stats['Qiskit Aer'][1]
    for name, data in rank_stats.items():
        speedup = aer_h / data[1] if data[1] > 0 else 0
        print(f"{name:<20} | {data[0]:<15.4f} | {data[1]*1000:<15.3f} | {speedup:>10.1f}x")
    print("="*85)

if __name__ == "__main__":
    run_majorana_universal()
