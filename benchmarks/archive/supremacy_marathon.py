import time
import tracemalloc
import numpy as np
import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import pennylane as qml
import os
import json

# =============================================================================
# CIRCUIT GENERATORS
# =============================================================================

def get_qc_circuit(n):
    """Complex QC: Quantum Fourier Transform (QFT)."""
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / 2**(k - j), k, j)
    qc.measure_all()
    return qc

def get_sf_qc_circuit(n):
    c = sf.Circuit(n)
    for j in range(n):
        c.h(j)
        for k in range(j + 1, n):
            c.cp(np.pi / 2**(k - j), k, j)
    return c

def get_qml_circuit(n, depth=2):
    """Complex QML: Variational Circuit with TwoLocal-like structure."""
    qc = QuantumCircuit(n)
    params = np.random.rand(depth, n, 3) * 2 * np.pi
    for d in range(depth):
        for i in range(n):
            qc.rx(params[d, i, 0], i)
            qc.ry(params[d, i, 1], i)
            qc.rz(params[d, i, 2], i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def get_sf_qml_circuit(n, depth=2):
    c = sf.Circuit(n)
    params = np.random.rand(depth, n, 3) * 2 * np.pi
    for d in range(depth):
        for i in range(n):
            c.rx(params[d, i, 0], i)
            c.ry(params[d, i, 1], i)
            c.rz(params[d, i, 2], i)
        for i in range(n - 1):
            c.cx(i, i + 1)
    return c

def get_qai_circuit(n, depth=1):
    """Complex QAI: Simplified Quantum Attention Head."""
    qc = QuantumCircuit(n)
    # Query/Key mapping (Rotations)
    for i in range(n):
        qc.ry(np.pi/4, i)
    # Entanglement 
    for i in range(0, n-1, 2):
        qc.cx(i, i+1)
    for i in range(1, n-1, 2):
        qc.cx(i, i+1)
    # Value mapping (Rotations)
    for i in range(n):
        qc.rz(np.pi/2, i)
    qc.measure_all()
    return qc

def get_sf_qai_circuit(n, depth=1):
    c = sf.Circuit(n)
    for i in range(n):
        c.ry(np.pi/4, i)
    for i in range(0, n-1, 2):
        c.cx(i, i+1)
    for i in range(1, n-1, 2):
        c.cx(i, i+1)
    for i in range(n):
        c.rz(np.pi/2, i)
    return c

# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

def run_benchmark():
    qubit_counts = [10, 20, 50, 100, 200, 500, 1000]
    scenarios = ["QC (QFT)", "QML (Var)", "QAI (Attn)"]
    shots = 1000
    
    results = []

    print("-" * 160)
    print(f"{'N':<4} | {'Scenario':<12} | {'Backend':<12} | {'S (Cold) ms':<12} | {'S (Hot) ms':<12} | {'Mem (MB)':<10} | {'Acc (GT)'}")
    print("-" * 160)

    for n in qubit_counts:
        for scenario in scenarios:
            # Generate circuits
            if scenario == "QC (QFT)":
                qc_qiskit = get_qc_circuit(n)
                qc_sf = get_sf_qc_circuit(n)
            elif scenario == "QML (Var)":
                qc_qiskit = get_qml_circuit(n)
                qc_sf = get_sf_qml_circuit(n)
            else:
                qc_qiskit = get_qai_circuit(n)
                qc_sf = get_sf_qai_circuit(n)

            # Ground Truth (Aer or PL)
            # Aer MPS for N > 30
            gt_counts = {}
            gt_backend = "Aer MPS"
            if n <= 100:
                try:
                    sim = AerSimulator(method='matrix_product_state' if n > 25 else 'statevector')
                    res = sim.run(qc_qiskit, shots=shots).result()
                    gt_counts = res.get_counts()
                except:
                    gt_counts = {}
            
            # Backends to test
            backends = ["singularity", "rust", "pennylane"]
            
            for b_name in backends:
                if b_name == "pennylane":
                    # PennyLane benchmarking
                    if n > 100 and scenario != "QAI (Attn)": continue # PL might be slow for complex n>100
                    try:
                        dev = qml.device("default.qubit" if n <= 25 else "lightning.qubit", wires=n)
                        @qml.qnode(dev)
                        def circuit_pl():
                            # Replicate circuit
                            if scenario == "QC (QFT)":
                                qml.QFT(wires=range(n))
                            elif scenario == "QML (Var)":
                                # Simplified version for benchmark
                                for i in range(n): qml.RY(0.5, wires=i)
                                for i in range(n-1): qml.CNOT(wires=[i, i+1])
                            else:
                                for i in range(n): qml.RY(0.5, wires=i)
                            return qml.counts()
                        
                        # Cold start
                        tracemalloc.start()
                        t0 = time.perf_counter()
                        _ = circuit_pl()
                        dt_cold = (time.perf_counter() - t0) * 1000
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        # Hot start
                        t0 = time.perf_counter()
                        _ = circuit_pl()
                        dt_hot = (time.perf_counter() - t0) * 1000
                        
                        acc = "N/A" # Skip accuracy for PL in this simple mode
                        print(f"{n:<4} | {scenario:<12} | {'PL-Lite':<12} | {dt_cold:<12.2f} | {dt_hot:<12.2f} | {peak/1024/1024:<10.2f} | {acc}")
                        results.append({"n":n, "scenario":scenario, "backend":"PL", "cold":dt_cold, "hot":dt_hot, "mem":peak/1024/1024})
                    except Exception as e:
                        print(f"{n:<4} | {scenario:<12} | {'PL-FAIL':<12} | {'--':<12} | {'--':<12} | {'--':<10} | {'--'}")
                    continue

                # SF Backends
                try:
                    # Cold start
                    tracemalloc.start()
                    t0 = time.perf_counter()
                    res_sf_cold = sf.run(qc_sf, backend=b_name, shots=shots)
                    dt_cold = (time.perf_counter() - t0) * 1000
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    # Hot start
                    t0 = time.perf_counter()
                    res_sf_hot = sf.run(qc_sf, backend=b_name, shots=shots)
                    dt_hot = (time.perf_counter() - t0) * 1000
                    
                    # Accuracy (TVD)
                    acc = 0.0
                    if gt_counts:
                        tvd = 0.0
                        all_k = set(gt_counts.keys()) | set(res_sf_hot.counts.keys())
                        for k in all_k:
                            tvd += abs(gt_counts.get(k, 0) - res_sf_hot.counts.get(k, 0))
                        acc = 1.0 - (tvd / (2 * shots))
                    else:
                        acc = "N/A"
                    
                    acc_str = f"{acc:.4f}" if isinstance(acc, float) else "N/A"
                    print(f"{n:<4} | {scenario:<12} | {b_name:<12} | {dt_cold:<12.2f} | {dt_hot:<12.2f} | {peak/1024/1024:<10.2f} | {acc_str}")
                    results.append({"n":n, "scenario":scenario, "backend":b_name, "cold":dt_cold, "hot":dt_hot, "mem":peak/1024/1024, "acc":acc})
                except Exception as e:
                    print(f"DEBUG ERROR {b_name} @ N={n}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"{n:<4} | {scenario:<12} | {b_name:<12} | {'FAIL':<12} | {'--':<12} | {'--':<10} | {'--'}")
                    tracemalloc.stop()
        print("-" * 160)

    # Save results to JSON
    with open("supremacy_marathon_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
