
import time
import tracemalloc
import numpy as np
import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def get_qk_stats(n, shots=1024, method='statevector'):
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    try:
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n-1): qc.cx(i, i+1)
        qc.measure_all()
        sim = AerSimulator(method=method)
        res = sim.run(qc, shots=shots).result()
        lat = (time.perf_counter_ns() - t0) / 1e6
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return lat, peak/1024/1024, "PASS"
    except Exception as e:
        tracemalloc.stop()
        return 0, 0, f"FAIL ({str(e)[:15]})"

def get_sf_stats(n, backend, shots=1024):
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    try:
        c = sf.Circuit(n)
        c.h(0)
        for i in range(n-1): c.cx(i, i+1)
        res = sf.run(c, backend=backend, shots=shots)
        lat = (time.perf_counter_ns() - t0) / 1e6
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return lat, peak/1024/1024, "PASS"
    except Exception as e:
        tracemalloc.stop()
        return 0, 0, f"FAIL ({str(e)[:15]})"

def scaling_sweep():
    counts = [10, 15, 20, 25, 30, 50, 100, 500, 1000]
    backends = ["simulator", "mps", "jax"]
    
    print("="*120)
    print(f"{'SUPERFERMION INDUSTRIAL SCALING SWEEP vs QISKIT AER':^120}")
    print("="*120)
    print(f"{'N':<4} | {'Backend':<12} | {'SF Lat (ms)':<12} | {'SF Mem (MB)':<12} | {'QK Lat (ms)':<12} | {'QK Mem (MB)':<12} | {'Status'}")
    print("-" * 120)

    for n in counts:
        for b in backends:
            # Skip SV-based for high qubit counts 
            if b in ["simulator", "jax"] and n > 25:
                continue
                
            # Determine QK method for comparison
            qk_method = 'statevector' if n <= 25 else 'matrix_product_state'
            
            # Get stats
            sf_lat, sf_mem, sf_stat = get_sf_stats(n, b)
            qk_lat, qk_mem, qk_stat = get_qk_stats(n, method=qk_method)
            
            status = "MATCH" if sf_stat == qk_stat else f"DIFF ({sf_stat}/{qk_stat})"
            
            print(f"{n:<4} | {b:<12} | {sf_lat:<12.2f} | {sf_mem:<12.2f} | {qk_lat:<12.2f} | {qk_mem:<12.2f} | {status}")
        print("-" * 120)

if __name__ == "__main__":
    scaling_sweep()
