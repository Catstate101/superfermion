import time
import numpy as np

# Framework Imports
import qiskit
from qiskit import QuantumCircuit

try:
    import pennylane as qml
except ImportError:
    qml = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

import superfermion as sf
from superfermion.circuit import Circuit as SFCircuit

def benchmark_creation():
    n_qubits = 100
    depth = 100 # 10,000 gates total
    
    print(f"--- Circuit Creation Benchmark ({n_qubits} Qubits, {depth*n_qubits} Gates) ---")
    
    # 1. Superfermion
    t0 = time.perf_counter()
    c_sf = SFCircuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            c_sf.h(i)
    t_sf = (time.perf_counter() - t0) * 1000
    print(f"Superfermion:  {t_sf:>10.4f} ms")

    # 2. Qiskit
    t0 = time.perf_counter()
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            qc.h(i)
    t_qs = (time.perf_counter() - t0) * 1000
    print(f"Qiskit:        {t_qs:>10.4f} ms")

    # 3. PennyLane
    if qml:
        t0 = time.perf_counter()
        def pl_circuit():
            for _ in range(depth):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
        # PennyLane creates "tapes" when recorded
        with qml.queuing.AnnotatedQueue() as q:
            pl_circuit()
        tape = qml.tape.QuantumScript.from_queue(q)
        t_pl = (time.perf_counter() - t0) * 1000
        print(f"PennyLane:     {t_pl:>10.4f} ms")

    # 4. TensorFlow (Pure Keras/TF layer creation analogy)
    if tf:
        # TFQ is usually used but if not found, we measure TF overhead for large objects
        t0 = time.perf_counter()
        # Simulating TF operation graph build
        @tf.function
        def tf_sim(val):
            return val * 2.0
        tf_sim(tf.constant(1.0)) # Trace
        t_tf = (time.perf_counter() - t0) * 1000
        print(f"TensorFlow:    {t_tf:>10.4f} ms (Graph Build)")

if __name__ == "__main__":
    benchmark_creation()
