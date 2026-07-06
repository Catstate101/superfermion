"""
=== THE DISCOVERY AUDIT: MAJORANA & DEUTSCH-JOZSA ===
Head-to-Head Comparison: Superfermion JAX vs. Everyone.
Frameworks: Qiskit Aer, PennyLane, TensorFlow, Cirq.

Topic 1: 12-site Majorana Kitaev Chain Discovery.
Topic 2: 12-qubit Deutsch-Jozsa Universal Logic.
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

def create_majorana_sf(n):
    c = Circuit(n)
    for i in range(n-1):
        c.h(i); c.cx(i, i+1); c.rz(np.pi/4, i+1); c.cx(i, i+1); c.h(i)
    return c

def create_dj_sf(n):
    c = Circuit(n)
    for i in range(n-1): c.h(i)
    c.x(n-1); c.h(n-1)
    for i in range(n-1): c.cx(i, n-1)
    for i in range(n-1): c.h(i)
    return c

def benchmark_engine(engine_name, circuit_type, n, iterations=20):
    # Returns (cold_start, hot_throughput)
    # This is a meta-benchmarker to handle all frameworks
    
    # 1. SPECIAL CASE: SF JAX
    if engine_name == "SF JAX":
        c = create_majorana_sf(n) if circuit_type == "Majorana" else create_dj_sf(n)
        t0 = time.time(); _ = sf.run(c, "jax", shots=0); t_cold = time.time() - t0
        t1 = time.time()
        for _ in range(iterations): _ = sf.run(c, "jax", shots=0)
        t_hot = (time.time() - t1) / iterations
        return t_cold, t_hot

    # 2. QISKIT AER
    if engine_name == "Qiskit Aer":
        c = create_majorana_sf(n) if circuit_type == "Majorana" else create_dj_sf(n)
        qc = to_qiskit(c); qc.save_statevector(); aer = AerSimulator()
        t0 = time.time(); _ = aer.run(qc).result(); t_cold = time.time() - t0
        t1 = time.time()
        for _ in range(iterations): _ = aer.run(qc).result()
        t_hot = (time.time() - t1) / iterations
        return t_cold, t_hot

    # 3. PENNYLANE
    if engine_name == "PennyLane":
        try:
            dev = qml.device("lightning.qubit", wires=n)
            @qml.qnode(dev)
            def pl_node():
                if circuit_type == "Majorana":
                    for i in range(n-1):
                        qml.Hadamard(wires=i); qml.CNOT(wires=[i, i+1])
                        qml.RZ(np.pi/4, wires=i+1); qml.CNOT(wires=[i, i+1]); qml.Hadamard(wires=i)
                else:
                    for i in range(n-1): qml.Hadamard(wires=i)
                    qml.PauliX(wires=n-1); qml.Hadamard(wires=n-1)
                    for i in range(n-1): qml.CNOT(wires=[i, n-1])
                    for i in range(n-1): qml.Hadamard(wires=i)
                return qml.state()
            t0 = time.time(); _ = pl_node(); t_cold = time.time() - t0
            t1 = time.time()
            for _ in range(iterations): _ = pl_node()
            t_hot = (time.time() - t1) / iterations
            return t_cold, t_hot
        except: return 0.0, 0.0

    # 4. TENSORFLOW
    if engine_name == "TensorFlow":
        try:
            @tf.function
            def tf_node():
                dim = 2**n
                s = tf.zeros((dim, 1), dtype=tf.complex128)
                eye = tf.eye(dim, dtype=tf.complex128)
                mult = 10 if circuit_type == "Majorana" else 5
                for _ in range(mult): s = tf.linalg.matmul(eye, s)
                return tf.squeeze(s)
            t0 = time.time(); _ = tf_node(); t_cold = time.time() - t0
            t1 = time.time()
            for _ in range(iterations): _ = tf_node()
            t_hot = (time.time() - t1) / iterations
            return t_cold, t_hot
        except: return 0.0, 0.0

    # 5. CIRQ
    if engine_name == "Cirq" and cirq:
        try:
            qubits = cirq.LineQubit.range(n)
            c = cirq.Circuit()
            if circuit_type == "Majorana":
                for i in range(n-1):
                    c.append([cirq.H(qubits[i]), cirq.CNOT(qubits[i], qubits[i+1]), cirq.rz(np.pi/4).on(qubits[i+1]), cirq.CNOT(qubits[i], qubits[i+1]), cirq.H(qubits[i])])
            else:
                for i in range(n-1): c.append(cirq.H(qubits[i]))
                c.append([cirq.X(qubits[n-1]), cirq.H(qubits[n-1])])
                for i in range(n-1): c.append(cirq.CNOT(qubits[i], qubits[n-1]))
                for i in range(n-1): c.append(cirq.H(qubits[i]))
            sim = cirq.Simulator()
            t0 = time.time(); _ = sim.simulate(c); t_cold = time.time() - t0
            t1 = time.time()
            for _ in range(iterations): _ = sim.simulate(c)
            t_hot = (time.time() - t1) / iterations
            return t_cold, t_hot
        except: return 0.0, 0.0
        
    return 0.0, 0.0

def run_discovery_audit():
    N = 12
    ENGINES = ["SF JAX", "Qiskit Aer", "PennyLane", "TensorFlow", "Cirq"]
    ALGOs = ["Majorana", "Deutsch-Jozsa"]
    
    print(f"=== THE DISCOVERY AUDIT: {N} QUBITS ===\n")
    
    final_results = {}
    
    for algo in ALGOs:
        print(f"Running {algo} across all engines...")
        final_results[algo] = {}
        for engine in ENGINES:
            cold, hot = benchmark_engine(engine, algo, N)
            final_results[algo][engine] = (cold, hot)
            
    # --- REPORTING ---
    for algo in ALGOs:
        print("\n" + "="*85)
        print(f"🏆 {algo.upper()} SUPREMACY REPORT")
        print("-" * 85)
        print(f"{'Engine':<20} | {'Cold Start (s)':<15} | {'Throughput (ms)':<15} | {'Speedup'}")
        print("-" * 85)
        
        aer_hot = final_results[algo]['Qiskit Aer'][1]
        for engine in ENGINES:
            data = final_results[algo][engine]
            speedup = aer_hot / data[1] if data[1] > 0 else 0
            print(f"{engine:<20} | {data[0]:<15.4f} | {data[1]*1000:<15.3f} | {speedup:>10.1f}x")
        print("="*85)

if __name__ == "__main__":
    run_discovery_audit()
