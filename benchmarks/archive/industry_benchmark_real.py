import sys
import time
import numpy as np
import superfermion as sf
from superfermion.circuit import Circuit as SFCircuit
from superfermion.simulator import simulate_statevector as sf_sim
from superfermion.backends.jax_sim import JAXBackend

# Framework imports
try:
    import qiskit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PL_AVAILABLE = True
except ImportError:
    PL_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def run_sf_numpy(n_qubits, depth=10):
    c = SFCircuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            c.h(i).rz(0.5, i)
        for i in range(n_qubits - 1):
            c.cnot(i, i+1)
    
    t0 = time.time()
    res = sf_sim(c)
    return time.time() - t0

def run_sf_jax(n_qubits, depth=10):
    c = SFCircuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            c.h(i).rz(0.5, i)
        for i in range(n_qubits - 1):
            c.cnot(i, i+1)
    
    backend = JAXBackend()
    # First run to JIT compile
    backend.simulate(c)
    
    t0 = time.time()
    res = backend.simulate(c)
    return time.time() - t0

def run_qiskit(n_qubits, depth=10):
    if not QISKIT_AVAILABLE: return -1
    qc = qiskit.QuantumCircuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            qc.h(i)
            qc.rz(0.5, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)
    
    sim = AerSimulator()
    qc.save_statevector()
    t0 = time.time()
    try:
        sim.run(qc).result().get_statevector()
    except Exception:
        return -1
    return time.time() - t0

def run_pennylane(n_qubits, depth=10):
    if not PL_AVAILABLE: return -1
    try:
        dev = qml.device('lightning.qubit', wires=n_qubits)
    except Exception:
        dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        for _ in range(depth):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(0.5, wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        return qml.state()
    
    # Pennylane lightning often needs a warm up too for best benchmark
    circuit()
    
    t0 = time.time()
    try:
        circuit()
    except Exception:
        return -1
    return time.time() - t0

def run_cirq(n_qubits, depth=10):
    if not CIRQ_AVAILABLE: return -1
    qubits = cirq.LineQubit.range(n_qubits)
    c = cirq.Circuit()
    for _ in range(depth):
        for i in range(n_qubits):
            c.append(cirq.H(qubits[i]))
            c.append(cirq.rz(0.5).on(qubits[i]))
        for i in range(n_qubits - 1):
            c.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    s = cirq.Simulator()
    t0 = time.time()
    try:
        s.simulate(c)
    except Exception:
        return -1
    return time.time() - t0

def benchmark():
    with open('bench_log_manual.txt', 'w') as f:
        f.write("="*85 + "\n")
        f.write("      QUANTUM FRAMEWORK BENCHMARK: SUPERFERMION (JAX-TURBO) VS INDUSTRY\n")
        f.write("="*85 + "\n")
        f.write(f"SF version: {sf.__version__}\n")
        if QISKIT_AVAILABLE: f.write(f"Qiskit Aer available: {qiskit.__version__}\n")
        if PL_AVAILABLE: f.write(f"PennyLane available: {qml.__version__}\n")
        if CIRQ_AVAILABLE: f.write(f"Cirq available: {cirq.__version__}\n")
        if TF_AVAILABLE: f.write(f"TensorFlow available: {tf.__version__}\n")
        f.write("="*85 + "\n")
        f.flush()
        
        results = {"SF_NP": [], "SF_JAX": [], "Qiskit": [], "PL": [], "Cirq": []}
        
        qubit_range = [4, 8, 12, 16]
        for n in qubit_range:
            f.write(f"\nBenchmarking {n} Qubits (depth=20)...\n")
            f.flush()
            
            t_sf_np = run_sf_numpy(n, 20)
            f.write(f"  SF NumPy: {t_sf_np:.4f}s\n")
            results["SF_NP"].append(t_sf_np)
            f.flush()

            t_sf_jax = run_sf_jax(n, 20)
            f.write(f"  SF JAX:   {t_sf_jax:.4f}s (WARM)\n")
            results["SF_JAX"].append(t_sf_jax)
            f.flush()

            t_qs = run_qiskit(n, 20)
            f.write(f"  Qiskit:  {t_qs:.4f}s\n")
            results["Qiskit"].append(t_qs)
            f.flush()

            t_pl = run_pennylane(n, 20)
            f.write(f"  PennyLane: {t_pl:.4f}s\n")
            results["PL"].append(t_pl)
            f.flush()

            t_cq = run_cirq(n, 20)
            f.write(f"  Cirq:    {t_cq:.4f}s\n")
            results["Cirq"].append(t_cq)
            f.flush()
        
        f.write("\nSummary Results (Time in s):\n")
        f.write("-" * 105 + "\n")
        f.write(f"{'Qubits':<8} | {'SF NumPy':<12} | {'SF JAX':<12} | {'Qiskit Aer':<15} | {'PennyLane':<15} | {'Cirq':<15}\n")
        f.write("-" * 105 + "\n")
        for i, n in enumerate(qubit_range):
            sf_np_t = f"{results['SF_NP'][i]:.4f}"
            sf_jax_t = f"{results['SF_JAX'][i]:.4f}"
            qs_t = f"{results['Qiskit'][i]:.4f}" if results['Qiskit'][i] > 0 else "N/A"
            pl_t = f"{results['PL'][i]:.4f}" if results['PL'][i] > 0 else "N/A"
            cq_t = f"{results['Cirq'][i]:.4f}" if results['Cirq'][i] > 0 else "N/A"
            f.write(f"{n:<8} | {sf_np_t:<12} | {sf_jax_t:<12} | {qs_t:<15} | {pl_t:<15} | {cq_t:<15}\n")
        f.flush()

        # Efficiency Multiplier
        f.write("\nJAX-TURBO Performance Boost vs NumPy:\n")
        for i, n in enumerate(qubit_range):
            boost = results['SF_NP'][i] / results['SF_JAX'][i]
            f.write(f"  {n:>2} Qubits: {boost:.1f}x speedup\n")
        f.flush()

        f.write("\n" + "="*85 + "\n")
        f.write("      USER EXPERIENCE / API SIMPLICITY COMPARISON\n")
        f.write("="*85 + "\n")
        f.write(f"{'Metric':<25} | {'SF':<10} | {'Qiskit':<8} | {'PL':<8} | {'Cirq':<8}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Import count':<25} | {'1':<10} | {'4+':<8} | {'2':<8} | {'1':<8}\n")
        f.write(f"{'Chainable API':<25} | {'Yes':<10} | {'No':<8} | {'No':<8} | {'No':<8}\n")
        f.write(f"{'JAX/GPU Support':<25} | {'Native':<10} | {'External':<8} | {'Plugin':<8} | {'Limited':<8}\n")
        f.write(f"{'DaaS Integration':<25} | {'Built-in':<10} | {'No':<8} | {'No':<8} | {'No':<8}\n")
        f.write("-" * 85 + "\n")
        f.flush()

if __name__ == "__main__":
    benchmark()
