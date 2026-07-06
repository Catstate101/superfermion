
import time
import tracemalloc
import numpy as np
import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def audit_rust_scaling():
    counts = [10, 20, 30, 50, 100, 250, 500, 1000]
    shots = 1024
    
    print("="*120)
    print(f"{'SF RUST CORE vs QISKIT AER: INDUSTRIAL QML SCALING SWEEP':^120}")
    print("="*120)
    print(f"{'N':<5} | {'Acc (1-TVD)':<12} | {'SF Rust Mem (MB)':<18} | {'SF Rust Lat (ms)':<18} | {'QK Aer Lat (ms)':<18} | {'QK Aer Mem (MB)':<18}")
    print("-" * 120)

    for n in counts:
        # --- 1. QISKIT AER GROUND TRUTH ---
        tracemalloc.start()
        t0 = time.perf_counter_ns()
        qc = QuantumCircuit(n)
        qc.ry(0.5, 0)
        for i in range(n-1): qc.cx(i, i+1)
        qc.measure_all()
        sim_qk = AerSimulator(method='matrix_product_state' if n > 25 else 'statevector')
        res_qk = sim_qk.run(qc, shots=shots).result()
        counts_qk = res_qk.get_counts()
        lat_qk = (time.perf_counter_ns() - t0) / 1e6
        _, peak_qk = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- 2. SF RUST CORE ---
        tracemalloc.start()
        t0 = time.perf_counter_ns()
        c = sf.Circuit(n)
        c.ry(0.5, 0)
        for i in range(n-1): c.cx(i, i+1)
        res_sf = sf.run(c, backend="rust", shots=shots)
        lat_sf = (time.perf_counter_ns() - t0) / 1e6
        _, peak_sf = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- 3. ACCURACY (TVD) ---
        # Remap Qiskit keys if needed
        # Qiskit: MSB is q[n-1]
        # SF: MSB is q[0]
        # The bitstrings should be compared carefully.
        # However, for GHZ/Ladder, they are mostly symmetric states.
        # For simplicity, we compare TVD directly as a statistical measure.
        tvd = 0.0
        all_keys = set(res_sf.counts.keys()) | set(counts_qk.keys())
        for k in all_keys:
            tvd += abs(res_sf.counts.get(k, 0) - counts_qk.get(k, 0))
        tvd = tvd / (2 * shots)
        acc = 1.0 - tvd

        print(f"{n:<5} | {acc:<12.4f} | {peak_sf/1024/1024:<18.2f} | {lat_sf:<18.2f} | {lat_qk:<18.2f} | {peak_qk/1024/1024:<18.2f}")

    print("="*120)

if __name__ == "__main__":
    audit_rust_scaling()
