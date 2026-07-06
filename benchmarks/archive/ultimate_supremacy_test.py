import os
import sys
import time
import tracemalloc
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import superfermion as sf
from superfermion.bridge import to_qiskit
from qiskit_aer import AerSimulator
import pennylane as qml
from collections import defaultdict
import numpy as np

def create_sf_circuit(n_qubits, depth=2):
    c = sf.Circuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            c.h(i)
            c.ry(0.1, i)
        for i in range(n_qubits - 1):
            c.cx(i, i + 1)
    return c

def create_pl_circuit(n_qubits, depth=2):
    def circuit():
        for _ in range(depth):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(0.1, wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        return qml.probs(wires=range(min(n_qubits, 10))) # Only probs of first 10 for memory/speed
    
    dev = qml.device("default.tensor", wires=n_qubits)
    qnode = qml.QNode(circuit, dev)
    return qnode

def measure_execution(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.time()
    try:
        res = func(*args, **kwargs)
        success = True
    except Exception as e:
        res = str(e)
        success = False
    
    t1 = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "success": success,
        "time_req": t1 - t0,
        "peak_mem_mb": peak_mem / (1024 * 1024),
        "result": res
    }

def run_sf_mps(circuit, backend="jax_mps", shots=1000):
    return sf.run(circuit, backend=backend, shots=shots, max_bond_dim=16)

def run_qiskit_mps(circuit, shots=1000):
    qc = to_qiskit(circuit)
    qc.measure_all()
    sim = AerSimulator(method="matrix_product_state")
    return sim.run(qc, shots=shots).result().get_counts()

def run_pl_tensor(qnode):
    return qnode()

def main():
    qubits_to_test = [20, 30, 40]
    depth = 2
    shots = 1000
    
    print("="*60)
    print("ULTIMATE SUPREMACY TEST: SuperFermion vs Qiskit vs PennyLane")
    print("="*60)
    
    log_content = ""
    
    for n in qubits_to_test:
        print(f"\\n--- Testing N={n} Qubits ---")
        log_content += f"\\n--- Testing N={n} Qubits ---\\n"
        
        # 1. SF
        sf_circ = create_sf_circuit(n, depth)
        print(f"Running SuperFermion (jax_mps)...")
        sf_stats = measure_execution(run_sf_mps, sf_circ, "jax_mps", shots)
        if sf_stats["success"]:
            msg = f"  [+] Time: {sf_stats['time_req']:.3f} s | Mem: {sf_stats['peak_mem_mb']:.2f} MB"
            print(msg)
            log_content += f"SF: {msg}\\n"
            sf_counts = sf_stats["result"].counts
            top_sf = sorted(sf_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [+] Top states: {top_sf}")
            log_content += f"SF Top: {top_sf}\\n"
        else:
            print(f"  [-] Failed: {sf_stats['result']}")
        
        # 2. Qiskit
        print(f"Running Qiskit Aer (mps)...")
        qiskit_stats = measure_execution(run_qiskit_mps, sf_circ, shots)
        if qiskit_stats["success"]:
            msg = f"  [+] Time: {qiskit_stats['time_req']:.3f} s | Mem: {qiskit_stats['peak_mem_mb']:.2f} MB"
            print(msg)
            log_content += f"Qk: {msg}\\n"
            qk_counts = qiskit_stats["result"]
            top_qk = sorted(qk_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [+] Top states: {top_qk}")
            log_content += f"Qk Top: {top_qk}\\n"
        else:
            print(f"  [-] Failed: {qiskit_stats['result']}")
            
        # 3. PennyLane
        print(f"Running PennyLane (default.tensor)...")
        pl_circ = create_pl_circuit(n, depth)
        pl_stats = measure_execution(run_pl_tensor, pl_circ)
        if pl_stats["success"]:
            msg = f"  [+] Time: {pl_stats['time_req']:.3f} s | Mem: {pl_stats['peak_mem_mb']:.2f} MB"
            print(msg)
            log_content += f"PL: {msg}\\n"
            print("  [+] Completed tensor network contraction")
        else:
            print(f"  [-] Failed: {pl_stats['result']}")
            
        print("-" * 60)
        
        # Scientific Ground truth check
        if sf_stats["success"] and qiskit_stats["success"]:
            sf_best = top_sf[0][0] if len(top_sf) > 0 else None
            qk_best = top_qk[0][0] if len(top_qk) > 0 else None
            print(f"Fidelity check / Ground Truth (SF vs QK): SF={sf_best}, QK={qk_best}")
            log_content += f"Gt: SF={sf_best}, QK={qk_best}\\n"
    
    with open("ultimate_supremacy_results.md", "w") as f:
        f.write("# Ultimate Benchmark Results\\n\\n")
        f.write("## Ground truth testing against Qiskit Aer and PennyLane\\n")
        f.write(log_content)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"CRASH CANCELLED: {e}")
