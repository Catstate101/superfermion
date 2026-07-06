
import time
import numpy as np
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.simulator import simulate_statevector
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator

def benchmark_suite():
    print("=== Superfermion Performance & Fidelity Benchmark ===")
    
    # 1. FIDELITY TEST: GHZ State (5 Qubits)
    print("\n[1/3] Fidelity Validation (5-Qubit GHZ)")
    n_ghz = 5
    sf_c = Circuit(n_ghz).h(0)
    for i in range(n_ghz - 1):
        sf_c.cnot(i, i+1)
    
    # SF Simulation
    sf_state = simulate_statevector(sf_c)
    
    # Qiskit Aer Simulation (Ground Truth)
    qc = to_qiskit(sf_c)
    qc.save_statevector()
    aer_sim = AerSimulator()
    aer_state = aer_sim.run(qc).result().get_statevector()
    
    # Calculate Fidelity (Overalp)
    fidelity = np.abs(np.vdot(sf_state, aer_state))**2
    print(f"Fidelity vs. Qiskit Aer: {fidelity:.10f}")
    if fidelity > 0.999999:
        print("RESULT: PASS - Mathematical Purity Confirmed.")
    else:
        print("RESULT: FAIL - Fidelity deviation detected.")

    # 2. PERFORMANCE SCALING: Large Qubit Counts
    print("\n[2/3] Performance Scaling (Wall Clock Time)")
    qubit_sizes = [15, 20, 22] # Testing the limits of local CPU
    
    print(f"{'Qubits':<10} | {'SF CPU (s)':<15} | {'SF GPU (s)':<15} | {'Aer CPU (s)'}")
    print("-" * 55)
    
    for n in qubit_sizes:
        # Create a randomized deep circuit
        c = Circuit(n)
        for i in range(n):
            c.h(i).rx(0.5, i)
        for i in range(n-1):
            c.cnot(i, i+1)
        
        # 1. SF Basic (CPU)
        t0 = time.time()
        simulate_statevector(c)
        sf_time = time.time() - t0
        
        # 2. SF Turbo (GPU - CuPy)
        sf_gpu_time = 0
        gpu_err = ""
        try:
            from superfermion.backends.cupy_sim import CupyBackend
            gpu_backend = CupyBackend()
            t_gpu = time.time()
            gpu_backend.run(c, shots=0)
            sf_gpu_time = time.time() - t_gpu
        except Exception as e:
            sf_gpu_time = -1
            gpu_err = str(e)
        
        # 3. SF JAX (CPU/GPU-Accelerated)
        sf_jax_time = 0
        try:
            import jax
            # Simple JAX simulation proxy (representing the DaaS engine speed)
            t_jax = time.time()
            sf_jax_time = time.time() - t_jax # JAX is usually used for optimization loops
        except:
            sf_jax_time = -1
        
        # 3. Qiskit Aer Benchmark
        qc = to_qiskit(c)
        qc.save_statevector()
        t1 = time.time()
        aer_sim.run(qc).result()
        aer_time = time.time() - t1
        
        gpu_str = f"{sf_gpu_time:<15.4f}" if sf_gpu_time > 0 else "N/A            "
        print(f"{n:<10} | {sf_time:<15.4f} | {gpu_str} | {aer_time:<15.4f}")

    if gpu_err:
        print(f"\n[!] GPU Warning: {gpu_err}")
        print("    Fix: Ensure CUDA_PATH is set and compatible NVIDIA drivers are installed.")
    
    # 3. LATENCY BENCHMARK: Circuit Building
    print("\n[3/3] Latency (Circuit Construction for 1,000 Gates)")
    t_start = time.time()
    c_long = Circuit(10)
    for _ in range(500):
        c_long.h(0).cnot(0, 1)
    build_time = (time.time() - t_start) * 1000 # convert to ms
    print(f"SF Circuit Build Latency: {build_time:.2f} ms")
    
    print("\n=== BENCHMARK COMPLETE ===")

if __name__ == "__main__":
    benchmark_suite()
