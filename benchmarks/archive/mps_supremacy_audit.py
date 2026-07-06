import time
import tracemalloc
import numpy as np
import superfermion as sf
import pandas as pd
import os
import gc
import csv

# Suppress JAX noise
os.environ["JAX_PLATFORMS"] = "cpu"

def build_entangled_circuit(n):
    """Highly entangled circuit: H layer + All-to-all CNOTs."""
    c = sf.Circuit(n)
    for i in range(n): c.h(i)
    # Entangle with many 2Q gates (Hard for MPS)
    for i in range(n - 1):
        c.cx(i, i + 1)
    for i in range(0, n - 2, 2):
        c.cx(i, i + 2)
    return c

def run_mps_supremacy_audit():
    # Sweep from 30 to 100 qubits
    qubit_range = [30, 40, 50, 64, 100]
    shots = 500
    seed = 42
    
    csv_file = "c:/Users/ASUS/OneDrive/Desktop/superfermion/tests/mps_supremacy_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "backend", "ms", "mem", "status"])

    print("🚀 STARTING MPS SUPREMACY STRESS TEST: SF vs QISKIT AER MPS...")
    
    for n in qubit_range:
        print(f"--- N={n} QUBITS ---")
        
        # 1. Qiskit Aer MPS (Dedicated MPS simulator)
        try:
            from qiskit import QuantumCircuit; from qiskit_aer import AerSimulator
            qc = QuantumCircuit(n); qc.h(0); [qc.cx(i, i+1) for i in range(n-1)]; qc.measure_all()
            sim = AerSimulator(method='matrix_product_state')
            t0 = time.time(); sim.run(qc, shots=shots).result(); t1 = time.time()
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([n, "Qiskit Aer (MPS)", (t1-t0)*1000, 0, "PASS"])
        except Exception as e:
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([n, "Qiskit Aer (MPS)", 0, 0, f"FAIL: {str(e)[:20]}"])

        # 2. SF Singularity (Warm)
        try:
            c = sf.Circuit(n).h(0); [c.cx(i, i+1) for i in range(n-1)]
            # Pre-bake to warm the cache
            sf.run(c, backend="singularity", shots=0)
            
            t0 = time.time(); sf.run(c, backend="singularity", shots=shots); t1 = time.time()
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([n, "SF Singularity (Warm)", (t1-t0)*1000, 0, "PASS"])
        except Exception as e:
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([n, "SF Singularity", 0, 0, f"FAIL: {str(e)[:20]}"])

        # 3. SF MPS (Python implementation)
        try:
            c = sf.Circuit(n).h(0); [c.cx(i, i+1) for i in range(n-1)]
            t0 = time.time(); sf.run(c, backend="mps", shots=shots); t1 = time.time()
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([n, "SF MPS", (t1-t0)*1000, 0, "PASS"])
        except Exception as e:
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([n, "SF MPS", 0, 0, f"FAIL: {str(e)[:20]}"])
            
        gc.collect()

    print("✅ MPS SUPREMACY AUDIT COMPLETE.")

if __name__ == "__main__":
    run_mps_supremacy_audit()
