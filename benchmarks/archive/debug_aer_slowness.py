import time
import qiskit
from qiskit_aer import AerSimulator

def debug_aer():
    n = 200
    depth = 5
    
    print(f"--- Debugging Qiskit Aer MPS Slowness (n={n}) ---")
    
    # 1. Timing Circuit Creation
    t0 = time.time()
    qc = qiskit.QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    for _ in range(depth):
        for i in range(0, n, 10):
            qc.x(i)
    qc.measure_all()
    t_create = time.time() - t0
    print(f"Circuit Creation: {t_create:.4f}s")
    
    # 2. Timing Simulation Run (Cold)
    sim = AerSimulator(method='matrix_product_state')
    t0 = time.time()
    sim.run(qc).result()
    t_run_cold = time.time() - t0
    print(f"Aer Run (Cold):  {t_run_cold:.4f}s")
    
    # 3. Timing Simulation Run (Warm)
    t0 = time.time()
    sim.run(qc).result()
    t_run_warm = time.time() - t0
    print(f"Aer Run (Warm):  {t_run_warm:.4f}s")

if __name__ == "__main__":
    debug_aer()
