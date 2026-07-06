import time
import numpy as np
import jax
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.backends.jax_mps import JAXMPSBackend
from superfermion.backends.mps import MPSSimulatorBackend

import qiskit
from qiskit_aer import AerSimulator

def build_qiskit_circuit(n, depth):
    qc = qiskit.QuantumCircuit(n)
    for _ in range(depth):
        for i in range(n): qc.x(i)
    qc.measure_all()
    return qc

def test_turbo_scaling():
    log_path = "turbo_mps_log.txt"
    with open(log_path, "w") as f_log:
        def log_print(msg):
            print(msg)
            f_log.write(msg + "\n")
            f_log.flush()

        log_print("="*105)
        log_print("      QUANTUM BENCHMARK: SF-JAX-MPS TURBO VS QISKIT AER MPS")
        log_print("="*105)
        
        qubit_counts = [100, 200, 500, 1000]
        depth = 5
        
        # Initialize backends
        turbo_backend = JAXMPSBackend()
        standard_backend = MPSSimulatorBackend()
        aer_sim = AerSimulator(method='matrix_product_state')
        
        header = f"{'Qubits':<8} | {'SF-JAX-Turbo (s)':<18} | {'Qiskit Aer MPS (s)':<18} | {'Standard MPS (s)':<18} | {'Win Factor':<10}"
        log_print(header)
        log_print("-" * 105)
        
        for n in qubit_counts:
            # Build circuits
            c_sf = Circuit(n)
            for _ in range(depth):
                for i in range(n): c_sf.x(i)
                
            c_qs = build_qiskit_circuit(n, depth)
            
            # 1. Qiskit Aer MPS
            t0 = time.time()
            aer_sim.run(c_qs).result()
            t_qs = time.time() - t0
            
            # 2. SF-JAX-Turbo (XLA-Fused)
            # Warmup (First compile)
            turbo_backend.run(c_sf, shots=1)
            
            t0 = time.time()
            turbo_backend.run(c_sf, shots=1)
            t_turbo = time.time() - t0
            
            # 3. Standard MPS (Python)
            t0 = time.time()
            standard_backend.run(c_sf, shots=1)
            t_std = time.time() - t0
            
            win_factor = t_qs / t_turbo if t_turbo > 0 else 0
            
            log_print(f"{n:<8} | {t_turbo:<18.6f} | {t_qs:<18.6f} | {t_std:<18.6f} | {win_factor:.2f}x")

        log_print("="*105)
        log_print("Note: SF-JAX-Turbo utilizes XLA 'lax.scan' Fusion to minimize Python dispatch latency.")
        log_print(f"Hardware: {jax.devices()[0]}")
        log_print("="*105)

if __name__ == "__main__":
    test_turbo_scaling()
