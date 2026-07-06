import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import pennylane as qml
import tracemalloc
import time
import gc
import numpy as np

def final_mem_check():
    n = 20
    shots = 1024
    print(f"=== Memory Efficiency Check (N={n}) ===")
    
    # 1. Qiskit
    gc.collect()
    tracemalloc.start()
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1): qc.cx(i, i+1)
    qc.measure_all()
    res_qk = AerSimulator(method="statevector").run(qc, shots=shots).result()
    _, peak_qk = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Qiskit Aer Memory: {peak_qk/1024/1024:.2f} MB")
    
    # 2. PennyLane
    gc.collect()
    tracemalloc.start()
    dev = qml.device("default.qubit", wires=n, shots=shots)
    @qml.qnode(dev)
    def circ_pl():
        qml.Hadamard(wires=0)
        for i in range(n-1): qml.CNOT(wires=[i, i+1])
        return qml.counts()
    _ = circ_pl()
    _, peak_pl = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"PennyLane Memory:  {peak_pl/1024/1024:.2f} MB")
    
    # 3. SF Rust
    gc.collect()
    tracemalloc.start()
    c = sf.Circuit(n).h(0)
    for i in range(n-1): c.cx(i, i+1)
    _ = sf.run(c, backend="rust", shots=shots)
    _, peak_rust = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"SF Rust Memory:    {peak_rust/1024/1024:.2f} MB")
    
    # 4. SF Singularity
    # Clear cache to get fresh bake memory
    from superfermion.backends.singularity import SingularityBackend
    SingularityBackend._topology_cache.clear()
    gc.collect()
    tracemalloc.start()
    _ = sf.run(c, backend="singularity", shots=shots)
    _, peak_sing = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"SF Singularity:    {peak_sing/1024/1024:.2f} MB")

if __name__ == "__main__":
    final_mem_check()
