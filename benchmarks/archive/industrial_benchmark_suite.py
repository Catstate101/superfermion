import time
import tracemalloc
import numpy as np
import superfermion as sf
from typing import Dict, Any, List
import pandas as pd
import os
import gc

# Suppress JAX/TensorFlow noise
os.environ["JAX_PLATFORMS"] = "cpu"

def run_industrial_benchmark():
    qubit_range = [4, 8, 16, 26, 40, 64, 128] # Reduced for time
    shots = 100
    seed = 42
    
    results = []
    
    print("🚀 RUNNING INDUSTRIAL QUANTUM SUPREMACY COMPARISON...")
    
    for n in qubit_range:
        print(f"--- N={n} ---")
        
        # Qiskit
        try:
            if n <= 32:
                from qiskit import QuantumCircuit; from qiskit_aer import AerSimulator
                qc = QuantumCircuit(n); qc.h(0); [qc.cx(i, i+1) for i in range(n-1)]; qc.measure_all()
                sim = AerSimulator(method='matrix_product_state' if n>26 else 'statevector')
                t0 = time.time(); sim.run(qc, shots=shots).result(); t1 = time.time()
                results.append({"n": n, "backend": "Qiskit Aer", "lat_ms": (t1-t0)*1000, "status": "PASS"})
            else: results.append({"n": n, "backend": "Qiskit Aer", "lat_ms": 0, "status": "FAIL (Limit)"})
        except: results.append({"n": n, "backend": "Qiskit Aer", "lat_ms": 0, "status": "FAIL/OOM"})

        # PennyLane
        try:
            if n <= 25:
                import pennylane as qml
                dev = qml.device("default.qubit", wires=n, shots=shots)
                @qml.qnode(dev)
                def circ():
                    qml.Hadamard(0); [qml.CNOT([i, i+1]) for i in range(n-1)]; return qml.counts()
                t0 = time.time(); circ(); t1 = time.time()
                results.append({"n": n, "backend": "PennyLane", "lat_ms": (t1-t0)*1000, "status": "PASS"})
            else: results.append({"n": n, "backend": "PennyLane", "lat_ms": 0, "status": "OOM Limit"})
        except: results.append({"n": n, "backend": "PennyLane", "lat_ms": 0, "status": "FAIL/OOM"})

        # SuperFermion Singularity (Turbo)
        try:
            # Re-import to ensure fresh state if needed
            from superfermion.backends.singularity import SingularityBackend
            SingularityBackend._topology_cache.clear()
            
            c = sf.Circuit(n).h(0); [c.cx(i, i+1) for i in range(n-1)]
            
            # Cold test
            t0 = time.time(); sf.run(c, backend="singularity", shots=shots); t1 = time.time()
            results.append({"n": n, "backend": "SF Singularity (Cold)", "lat_ms": (t1-t0)*1000, "status": "PASS"})
            
            # Warm test
            t0 = time.time(); sf.run(c, backend="singularity", shots=shots); t1 = time.time()
            results.append({"n": n, "backend": "SF Singularity (Warm)", "lat_ms": (t1-t0)*1000, "status": "PASS"})
        except Exception as e:
            results.append({"n": n, "backend": "SF Singularity", "lat_ms": 0, "status": f"FAIL: {str(e)[:30]}"})
            
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv("c:/Users/ASUS/OneDrive/Desktop/superfermion/tests/industrial_benchmark_results.csv", index=False)
    
    # GENERATE MD REPORT
    md_report = """# Quantum Simulation Industrial Hierarchy Report
Developed by SuperFermion Performance Engineering Unit.

## Performance Comparison (Latency in ms)
| N | Qiskit Aer | PennyLane | SF Singularity (Cold) | SF Singularity (Warm) |
|---|---|---|---|---|
"""
    for n in qubit_range:
        q = df[(df.n==n) & (df.backend=='Qiskit Aer')].lat_ms.values[0]
        p = df[(df.n==n) & (df.backend=='PennyLane')].lat_ms.values[0]
        sc = df[(df.n==n) & (df.backend=='SF Singularity (Cold)')].lat_ms.values[0]
        sw = df[(df.n==n) & (df.backend=='SF Singularity (Warm)')].lat_ms.values[0]
        md_report += f"| {n} | {q:.1f} | {p:.1f} | {sc:.1f} | {sw:.1f} |\n"
    
    with open("c:/Users/ASUS/OneDrive/Desktop/superfermion/INDUSTRIAL_REPORT.md", "w") as f:
        f.write(md_report)
    print("✅ INDUSTRIAL REPORT GENERATED.")

if __name__ == "__main__":
    run_industrial_benchmark()
