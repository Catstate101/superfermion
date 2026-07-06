
import time
import tracemalloc
import numpy as np
import superfermion as sf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def audit_rust_production():
    # Gradual increase as requested
    counts = [10, 15, 20, 25, 30, 50, 100, 200]
    shots = 1024
    
    print("="*120)
    print(f"{'SF RUST INDUSTRIAL QML COMPARISON vs QISKIT AER':^120}")
    print("="*105)
    print(f"{'Qubits':<8} | {'SF Rust Mem (MB)':<18} | {'QK Aer Mem (MB)':<18} | {'SF Lat (ms)':<15} | {'QK Lat (ms)':<15} | {'Status'}")
    print("-" * 120)

    for n in counts:
        # Use MPS for both if n > 25
        qk_method = 'matrix_product_state' if n > 25 else 'statevector'
        
        # 1. QISKIT AER
        tracemalloc.start()
        t0 = time.perf_counter_ns()
        try:
            qc = QuantumCircuit(n)
            qc.ry(0.5, 0)
            for i in range(n-1): qc.cx(i, i+1)
            qc.measure_all()
            sim = AerSimulator(method=qk_method)
            res_qk = sim.run(qc, shots=shots).result()
            lat_qk = (time.perf_counter_ns() - t0) / 1e6
            _, peak_qk = tracemalloc.get_traced_memory()
            stat_qk = "PASS"
        except Exception as e:
            lat_qk, peak_qk, stat_qk = 0, 0, f"FAIL"
        tracemalloc.stop()

        # 2. SF RUST
        tracemalloc.start()
        t0 = time.perf_counter_ns()
        try:
            c = sf.Circuit(n)
            c.ry(0.5, 0)
            for i in range(n-1): c.cx(i, i+1)
            # We use 'rust' backend which should adaptively use MPS for n > 24
            res_sf = sf.run(c, backend="rust", shots=shots)
            lat_sf = (time.perf_counter_ns() - t0) / 1e6
            _, peak_sf = tracemalloc.get_traced_memory()
            stat_sf = "PASS"
        except Exception as e:
            # If Rust fails, we report the error (likely OOM)
            lat_sf, peak_sf = 0, 0
            stat_sf = f"FAIL ({str(e)[:10]})"
        tracemalloc.stop()

        print(f"{n:<8} | {peak_sf/1024/1024:<18.2f} | {peak_qk/1024/1024:<18.2f} | {lat_sf:<15.2f} | {lat_qk:<15.2f} | {stat_sf}/{stat_qk}")

    print("="*120)

if __name__ == "__main__":
    audit_rust_production()
