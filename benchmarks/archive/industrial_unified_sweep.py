
import time
import tracemalloc
import numpy as np
import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def unified_benchmark():
    # Gradual Sweep points
    counts = [10, 20, 25, 30, 40, 50, 100, 200, 500, 1000]
    shots = 1024
    
    print("="*140)
    print(f"{'SUPERFERMION INDUSTRIAL UNIFIED SWEEP: ALL BACKENDS vs QISKIT AER':^140}")
    print("="*140)
    print(f"{'N':<4} | {'Backend':<12} | {'Acc (TVD)':<10} | {'SF Lat (ms)':<12} | {'QK Lat (ms)':<12} | {'SF Mem (MB)':<12} | {'QK Mem (MB)':<12} | {'Baking'}")
    print("-" * 140)

    for n in counts:
        # 1. PRE-CALCULATE QK GROUND TRUTH FOR N
        tracemalloc.start()
        t0 = time.perf_counter_ns()
        try:
            qc = QuantumCircuit(n)
            qc.h(0)
            for i in range(n-1): qc.cx(i, i+1)
            qc.measure_all()
            meth = 'matrix_product_state' if n > 25 else 'statevector'
            sim = AerSimulator(method=meth)
            res_qk = sim.run(qc, shots=shots).result()
            counts_qk = res_qk.get_counts()
            lat_qk = (time.perf_counter_ns() - t0) / 1e6
            _, peak_qk = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        except:
            tracemalloc.stop()
            lat_qk, peak_qk, counts_qk = 0, 0, {}

        # 2. RUN SF BACKENDS
        # Decide which backends to run for N
        b_to_run = ["simulator", "jax", "rust", "mps", "singularity"]
        if n > 25: b_to_run = ["mps", "singularity"]

        for b in b_to_run:
            tracemalloc.start()
            t0 = time.perf_counter_ns()
            try:
                c = sf.Circuit(n).h(0)
                for i in range(n-1): c.cx(i, i+1)
                
                # We skip to_ir for plain 'rust' if n > 25 to avoid OOM check in constructor
                # but 'singularity' will handle it correctly
                res_sf = sf.run(c, backend=b, shots=shots)
                lat_sf = (time.perf_counter_ns() - t0) / 1e6
                _, peak_sf = tracemalloc.get_traced_memory()
                
                # Accuracy calc (Statistical TVD)
                tvd = 0.0
                all_ks = set(res_sf.counts.keys()) | set(counts_qk.keys())
                for k in all_ks: tvd += abs(res_sf.counts.get(k, 0) - counts_qk.get(k, 0))
                acc = 1.0 - (tvd / (2 * shots))
                
                baking = "YES" if res_sf.metadata.get("singularity_mode") else "NO"
                print(f"{n:<4} | {b:<12} | {acc:<10.4f} | {lat_sf:<12.2f} | {lat_qk:<12.2f} | {peak_sf/1024/1024:<12.2f} | {peak_qk/1024/1024:<12.2f} | {baking}")
            except Exception as e:
                print(f"{n:<4} | {b:<12} | {'FAIL':<10} | {'--':<12} | {'--':<12} | {'--':<12} | {'--':<12} | {'--'}")
            tracemalloc.stop()
        print("-" * 140)

if __name__ == "__main__":
    unified_benchmark()
