import time
import numpy as np
import jax
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.simulator import simulate_statevector as sf_np_sim
from superfermion.backends.jax_sim import JAXBackend as SF_JAX
from superfermion.backends.jax_mps import JAXMPSBackend as SF_JAX_MPS

import qiskit
from qiskit_aer import AerSimulator

def bench_task(name, n_qubits, build_fn, shots=0):
    print(f"\n--- TASK: {name} ({n_qubits} Qubits) ---")
    
    # Pre-build circuits
    c_sf = build_fn(n_qubits, "sf")
    c_qs = build_fn(n_qubits, "qiskit")
    
    # Backends
    jax_sv = SF_JAX()
    jax_mps = SF_JAX_MPS()
    aer_sv = AerSimulator(method='statevector')
    aer_mps = AerSimulator(method='matrix_product_state')

    results = {}

    # 1. SF NumPy (Minimal dependency latency)
    t0 = time.perf_counter()
    sf_np_sim(c_sf)
    results["SF-NumPy"] = (time.perf_counter() - t0) * 1000

    # 2. SF JAX (Warm)
    jax_sv.run(c_sf, shots=0) # Warmup
    t0 = time.perf_counter()
    jax_sv.run(c_sf, shots=0)
    results["SF-JAX-SV"] = (time.perf_counter() - t0) * 1000

    # 3. SF JAX MPS (Warm)
    jax_mps.run(c_sf, shots=0) # Warmup
    t0 = time.perf_counter()
    jax_mps.run(c_sf, shots=0)
    results["SF-JAX-MPS"] = (time.perf_counter() - t0) * 1000

    # 4. Qiskit Aer SV (Warm)
    aer_sv.run(c_qs).result() # Warmup
    t0 = time.perf_counter()
    aer_sv.run(c_qs).result()
    results["Qiskit-Aer-SV"] = (time.perf_counter() - t0) * 1000

    # 5. Qiskit Aer MPS (Warm)
    aer_mps.run(c_qs).result() # Warmup
    t0 = time.perf_counter()
    aer_mps.run(c_qs).result()
    results["Qiskit-Aer-MPS"] = (time.perf_counter() - t0) * 1000

    # Print Table
    print(f"{'Backend':<20} | {'Latency (ms)':<15}")
    print("-" * 38)
    for b, t in results.items():
        print(f"{b:<20} | {t:>12.4f} ms")
    
    winner = min(results, key=results.get)
    print(f"WINNER: {winner}")

# --- Task Generators ---

def build_bell(n, framework):
    if framework == "sf":
        return Circuit(2).h(0).cnot(0, 1)
    else:
        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

def build_ghz(n, framework):
    if framework == "sf":
        c = Circuit(n).h(0)
        for i in range(n-1): c.cnot(i, i+1)
        return c
    else:
        qc = qiskit.QuantumCircuit(n)
        qc.h(0)
        for i in range(n-1): qc.cx(i, i+1)
        qc.measure_all()
        return qc

def build_random(n, framework):
    if framework == "sf":
        c = Circuit(n)
        for i in range(n): c.h(i).rx(0.5, i)
        for i in range(n-1): c.cnot(i, i+1)
        return c
    else:
        qc = qiskit.QuantumCircuit(n)
        for i in range(n): qc.h(i); qc.rx(0.5, i)
        for i in range(n-1): qc.cx(i, i+1)
        qc.measure_all()
        return qc

if __name__ == "__main__":
    print("="*50)
    print("      SMALL TASK BENCHMARK: OVERHEAD TEST")
    print("="*50)
    
    bench_task("2-Qubit Bell State", 2, build_bell)
    bench_task("5-Qubit GHZ State", 5, build_ghz)
    bench_task("12-Qubit Random Circuit", 12, build_random)
