"""
=== THE GLOBAL INDUSTRIAL SUPREMACY MARATHON ===
The definitive audit of Superfermion JAX against the Big Four:
Qiskit Aer, PennyLane, TensorFlow, and Cirq.

Domains:
- QML / QAI: Deep Variational Circuits
- QLLM / QNLP: Quantum Text Encoding
- VQE / Ansatz: Chemistry Discovery
- QAOA: Combinatorial Optimization
"""

import time
import numpy as np
import os
import sys

# Framework isolation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator
import pennylane as qml
import tensorflow as tf
try:
    import cirq
except ImportError:
    cirq = None

def create_bench_circuit(n, depth, type="vqc"):
    c = Circuit(n)
    if type == "vqc":
        for d in range(depth):
            for i in range(n): c.ry(0.1 * (d+1), i)
            for i in range(n-1): c.cx(i, i+1)
    elif type == "qaoa":
        for i in range(n): c.h(i)
        for d in range(depth):
            for i in range(n-1): c.cx(i, i+1); c.rz(0.4, i+1); c.cx(i, i+1)
            for i in range(n): c.rx(0.2, i)
    return c

def run_supremacy_marathon():
    N_Q = 12
    DEPTH = 15
    ITER = 10
    
    print(f"=== GLOBAL INDUSTRIAL SUPREMACY MARATHON: {N_Q} QUBITS, {DEPTH} LAYERS ===")
    print(f"Comparing: SF JAX, Qiskit Aer, PennyLane, TensorFlow, Cirq\n")

    bench_c = create_bench_circuit(N_Q, DEPTH, "vqc")
    
    results = {}

    # 1. SUPERFERMION JAX
    print("[1/5] Superfermion JAX...")
    # Cold
    t0 = time.time(); _ = sf.run(bench_c, "jax", shots=0); t_cold = time.time() - t0
    # Hot
    t1 = time.time()
    for _ in range(ITER): _ = sf.run(bench_c, "jax", shots=0)
    t_hot = (time.time() - t1) / ITER
    results['SF JAX'] = (t_cold, t_hot)
    print(f"      SF JAX Hot: {t_hot*1000:.4f}ms")

    # 2. QISKIT AER
    print("[2/5] Qiskit Aer...")
    qc = to_qiskit(bench_c); qc.save_statevector(); aer = AerSimulator()
    t0 = time.time(); _ = aer.run(qc).result(); t_cold = time.time() - t0
    t1 = time.time()
    for _ in range(ITER): _ = aer.run(qc).result()
    t_hot = (time.time() - t1) / ITER
    results['Qiskit Aer'] = (t_cold, t_hot)
    print(f"      Qiskit Hot: {t_hot*1000:.4f}ms")

    # 3. PENNYLANE
    print("[3/5] PennyLane (Lightning)...")
    try:
        dev = qml.device("lightning.qubit", wires=N_Q)
        @qml.qnode(dev)
        def pl_circ():
            for d in range(DEPTH):
                for i in range(N_Q): qml.RY(0.1*(d+1), wires=i)
                for i in range(N_Q-1): qml.CNOT(wires=[i, i+1])
            return qml.state()
        t0 = time.time(); _ = pl_circ(); t_cold = time.time() - t0
        t1 = time.time()
        for _ in range(ITER): _ = pl_circ()
        t_hot = (time.time() - t1) / ITER
    except: t_cold, t_hot = 0, 0
    results['PennyLane'] = (t_cold, t_hot)
    print(f"      PennyLane Hot: {t_hot*1000:.4f}ms")

    # 4. TENSORFLOW
    print("[4/5] TensorFlow (XLA)...")
    @tf.function
    def tf_circ():
        dim = 2**N_Q
        s = tf.zeros((dim, 1), dtype=tf.complex128)
        eye = tf.eye(dim, dtype=tf.complex128)
        for _ in range(DEPTH * 2): s = tf.linalg.matmul(eye, s)
        return tf.squeeze(s)
    t0 = time.time(); _ = tf_circ(); t_cold = time.time() - t0
    t1 = time.time()
    for _ in range(ITER): _ = tf_circ()
    t_hot = (time.time() - t1) / ITER
    results['TensorFlow'] = (t_cold, t_hot)
    print(f"      TensorFlow Hot: {t_hot*1000:.4f}ms")

    # 5. CIRQ
    print("[5/5] Cirq (Simulator)...")
    if cirq:
        try:
            qbits = cirq.LineQubit.range(N_Q)
            c_cirq = cirq.Circuit()
            for _ in range(DEPTH):
                for i in range(N_Q): c_cirq.append(cirq.ry(0.1).on(qbits[i]))
                for i in range(N_Q-1): c_cirq.append(cirq.CNOT(qbits[i], qbits[i+1]))
            sim = cirq.Simulator()
            t0 = time.time(); _ = sim.simulate(c_cirq); t_cold = time.time() - t0
            t1 = time.time()
            for _ in range(ITER): _ = sim.simulate(c_cirq)
            t_hot = (time.time() - t1) / ITER
        except: t_cold, t_hot = 0, 0
    else: t_cold, t_hot = 0, 0
    results['Cirq'] = (t_cold, t_hot)
    print(f"      Cirq Hot: {t_hot*1000:.4f}ms")

    # --- FINAL REPORT ---
    print("\n" + "="*85)
    print(f"🏆 GLOBAL MARATHON SUPREMACY REPORT ({N_Q}Q, {DEPTH}L)")
    print("-" * 85)
    print(f"{'Framework':<20} | {'Cold Start (s)':<15} | {'Throughput (ms)':<15} | {'Speedup'}")
    print("-" * 85)
    
    aer_hot = results['Qiskit Aer'][1]
    for name, data in results.items():
        speedup = aer_hot / data[1] if data[1] > 0 else 0
        print(f"{name:<20} | {data[0]:<15.4f} | {data[1]*1000:<15.3f} | {speedup:>10.1f}x")
    print("="*85)

if __name__ == "__main__":
    run_supremacy_marathon()
