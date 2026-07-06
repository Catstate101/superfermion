import time
import numpy as np
import superfermion as sf
from superfermion.backends.singularity import SingularityBackend
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
import pandas as pd

def run_mps_supremacy_comparison():
    # Sweep range for heavy qubits
    qubit_range = [30, 40, 60, 80, 100]
    shots = 1000
    
    results = []
    
    print("🚀 QUANTUM SUPREMACY MPS AUDIT: SUPERFERMION vs QISKIT AER C++ MPS")
    
    for n in qubit_range:
        print(f"--- Testing N = {n} Qubits ---")
        
        # 1. CIRCUIT DEFINITION (Entangled Chain)
        # SF Circuit
        c = sf.Circuit(n).h(0)
        for i in range(n-1): c.cx(i, i+1)
        
        # Qiskit Circuit
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n-1): qc.cx(i, i+1)
        qc.measure_all()
        
        # 2. QISKIT AER MPS (Cold Start)
        sim = AerSimulator(method='matrix_product_state')
        t0 = time.time()
        res_qk = sim.run(qc, shots=shots).result()
        lat_qk = (time.time() - t0) * 1000
        print(f"  Qiskit Aer MPS Latency: {lat_qk:.2f} ms")
        
        # 3. SF SINGULARITY (Cold Start)
        SingularityBackend._topology_cache.clear()
        t0 = time.time()
        res_sf_cold = sf.run(c, backend="singularity", shots=shots)
        lat_sf_cold = (time.time() - t0) * 1000
        print(f"  SF Singularity Cold Latency: {lat_sf_cold:.2f} ms")
        
        # 4. SF SINGULARITY (Warm Start - Industrial Case)
        # Re-run the same circuit
        t0 = time.time()
        res_sf_warm = sf.run(c, backend="singularity", shots=shots)
        lat_sf_warm = (time.time() - t0) * 1000
        print(f"  SF Singularity Warm Latency: {lat_sf_warm:.2f} ms")
        
        # 5. GROUND TRUTH FIDELITY (approx check via counts bitstrings)
        # For GHZ-style, only '0'*n and '1'*n should exist.
        match_sf = (res_sf_warm.counts.get('0'*n, 0) + res_sf_warm.counts.get('1'*n, 0)) / shots
        match_qk = (res_qk.get_counts().get('0'*n, 0) + res_qk.get_counts().get('1'*n, 0)) / shots
        
        results.append({
            "n": n,
            "qk_lat_ms": lat_qk,
            "sf_cold_ms": lat_sf_cold,
            "sf_warm_ms": lat_sf_warm,
            "sf_fidelity_proxy": match_sf,
            "qk_fidelity_proxy": match_qk
        })
        
    df = pd.DataFrame(results)
    df.to_csv("c:/Users/ASUS/OneDrive/Desktop/superfermion/tests/mps_supremacy_final.csv", index=False)
    
    # Generate MD Report Snippet
    report = f"## MPS Stress Test Results (N=30 to N=100)\n\n"
    report += "| N | Qiskit MPS (Cold) | SF Singularity (Cold) | SF Singularity (Warm) | Fidelity Match |\n"
    report += "|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        report += f"| {int(r.n)} | {r.qk_lat_ms:.1f}ms | {r.sf_cold_ms:.1f}ms | {r.sf_warm_ms:.1f}ms | {r.sf_fidelity_proxy*100:.1f}% |\n"
        
    with open("c:/Users/ASUS/OneDrive/Desktop/superfermion/MPS_SUPREMACY_REPORT.md", "w") as f:
        f.write(report)
    print("✅ MPS SUPREMACY AUDIT COMPLETE.")

if __name__ == "__main__":
    run_mps_supremacy_comparison()
