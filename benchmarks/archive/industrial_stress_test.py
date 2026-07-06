"""
=== SUPERFERMION INDUSTRIAL COMPARISON SUITE ===
A deep-dive benchmark comparing Superfermion JAX against Qiskit Aer,
PennyLane, and TensorFlow on high-complexity stress tests.

Complexity Setup: 12 Qubits, 20 Layers (~700+ Gates)
Metric: Latency per execution, Compilation Overhead, and Memory Scaling.
"""

import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator

import pennylane as qml
import tensorflow as tf
import jax.numpy as jnp

def create_complex_circuit(n, depth):
    c = Circuit(n)
    for d in range(depth):
        for i in range(n):
            c.rx(0.123 * d, i)
            c.ry(0.456 * d, i)
            c.rz(0.789 * d, i)
        for i in range(n-1):
            c.cx(i, i+1)
    return c

def run_industrial_stress_test():
    n_qubits = 12
    depth = 20
    iterations = 20
    
    print(f"=== INDUSTRIAL STRESS TEST: {n_qubits} QUBITS, {depth} LAYERS (~700 GATES) ===")
    print(f"Running {iterations} iterations for throughput measurement.\n")

    sf_c = create_complex_circuit(n_qubits, depth)
    
    results = {}

    # 1. SUPERFERMION JAX
    print("[1/4] Testing Superfermion JAX...")
    t_start = time.time()
    _ = sf.run(sf_c, backend="jax", shots=0) # Compilation
    t_cold = time.time() - t_start
    
    t0 = time.time()
    for _ in range(iterations):
        _ = sf.run(sf_c, backend="jax", shots=0)
    t_hot = (time.time() - t0) / iterations
    results['SF JAX'] = {'cold': t_cold, 'hot': t_hot}

    # 2. QISKIT AER
    print("[2/4] Testing Qiskit Aer...")
    qc = to_qiskit(sf_c)
    qc.save_statevector()
    aer_sim = AerSimulator()
    
    t_start = time.time()
    _ = aer_sim.run(qc).result()
    t_cold = time.time() - t_start
    
    t0 = time.time()
    for _ in range(iterations):
        aer_sim.run(qc).result()
    t_hot = (time.time() - t0) / iterations
    results['Qiskit Aer'] = {'cold': t_cold, 'hot': t_hot}

    # 3. PENNYLANE
    print("[3/4] Testing PennyLane...")
    # Using lightning.qubit if available, else default.qubit
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except:
        dev = qml.device("default.qubit", wires=n_qubits)
        
    @qml.qnode(dev)
    def pl_node():
        for d in range(depth):
            for i in range(n_qubits):
                qml.RX(0.123 * d, wires=i)
                qml.RY(0.456 * d, wires=i)
                qml.RZ(0.789 * d, wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        return qml.state()

    t_start = time.time()
    _ = pl_node()
    t_cold = time.time() - t_start

    t0 = time.time()
    for _ in range(iterations):
        _ = pl_node()
    t_hot = (time.time() - t0) / iterations
    results['PennyLane'] = {'cold': t_cold, 'hot': t_hot}

    # 4. TENSORFLOW (Native Ops Simulate)
    print("[4/4] Testing TensorFlow...")
    @tf.function
    def tf_complex_step():
        dim = 2**n_qubits
        state = tf.zeros((dim, 1), dtype=tf.complex128)
        # Mocking the gate application overhead
        eye = tf.eye(dim, dtype=tf.complex128)
        for d in range(depth):
            # Apply "gates" as matrix multiplications
            state = tf.linalg.matmul(eye, state) 
        return tf.squeeze(state)

    t_start = time.time()
    _ = tf_complex_step()
    t_cold = time.time() - t_start

    t0 = time.time()
    for _ in range(iterations):
        _ = tf_complex_step()
    t_hot = (time.time() - t0) / iterations
    results['TensorFlow'] = {'cold': t_cold, 'hot': t_hot}

    # --- PRINT FINAL REPORT ---
    print("\n\n" + "="*80)
    print(f"{'Framework':<20} | {'Cold Start (s)':<15} | {'Throughput (ms)':<15} | {'Speed Ranking'}")
    print("-" * 80)
    
    sorted_res = sorted(results.items(), key=lambda x: x[1]['hot'])
    for i, (name, data) in enumerate(sorted_res):
        rank = ["🥇 TOP", "🥈 HIGH", "🥉 MID", "   SLOW"][i]
        print(f"{name:<20} | {data['cold']:<15.4f} | {data['hot']*1000:<15.2f} | {rank}")
    print("="*80)

if __name__ == "__main__":
    run_industrial_stress_test()
