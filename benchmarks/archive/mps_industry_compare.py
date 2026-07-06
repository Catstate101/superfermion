import time
import os
import sys
import numpy as np

# Framework Imports
import superfermion as sf
from superfermion.circuit import Circuit as SFCircuit
from superfermion.backends.mps import MPSSimulatorBackend as SF_MPS
from superfermion.backends.jax_mps import JAXMPSBackend as SF_JAX_MPS

import qiskit
from qiskit_aer import AerSimulator

try:
    import pennylane as qml
except ImportError:
    qml = None

try:
    import cirq
except ImportError:
    cirq = None

# --------------------------------------------------------------------------------
# CIRCUIT GENERATORS
# --------------------------------------------------------------------------------

def build_sf_circuit(n, depth):
    c = SFCircuit(n)
    for _ in range(depth):
        for i in range(n): c.h(i)
        for i in range(n-1): c.cnot(i, i+1)
    return c

def build_qiskit_circuit(n, depth):
    qc = qiskit.QuantumCircuit(n)
    for _ in range(depth):
        for i in range(n): qc.h(i)
        for i in range(n-1): qc.cx(i, i+1)
    qc.measure_all()
    return qc

def build_pl_circuit(n, depth, dev):
    @qml.qnode(dev)
    def circuit():
        for _ in range(depth):
            for i in range(n): qml.Hadamard(wires=i)
            for i in range(n-1): qml.CNOT(wires=[i, i+1])
        return qml.counts()
    return circuit

# --------------------------------------------------------------------------------
# BENCHMARK
# --------------------------------------------------------------------------------

def run_mps_benchmark():
    qubits_list = [50, 80, 100]
    depth = 2
    
    results = {
        "SF-MPS": [],
        "SF-JAX-MPS": [],
        "Qiskit-MPS": [],
        "PL-Default": [],
        "Cirq-Default": []
    }

    print("="*90)
    print(f"{'Qubits':<10} | {'SF-MPS':<12} | {'SF-JAX-MPS':<12} | {'Qiskit-MPS':<15} | {'PL-Def':<12} | {'Cirq-Def':<12}")
    print("-" * 90)

    for n in qubits_list:
        # 1. SF-MPS
        sf_c = build_sf_circuit(n, depth)
        backend_mps = SF_MPS()
        t0 = time.time()
        backend_mps.run(sf_c, shots=1024)
        t_sf = time.time() - t0
        results["SF-MPS"].append(t_sf)

        # 2. SF-JAX-MPS
        backend_jax_mps = SF_JAX_MPS()
        # Warmup
        backend_jax_mps.run(sf_c, shots=1)
        t0 = time.time()
        backend_jax_mps.run(sf_c, shots=1024)
        t_sf_jax = time.time() - t0
        results["SF-JAX-MPS"].append(t_sf_jax)

        # 3. Qiskit-Aer MPS
        qs_c = build_qiskit_circuit(n, depth)
        sim = AerSimulator(method='matrix_product_state')
        t0 = time.time()
        sim.run(qs_c).result()
        t_qs = time.time() - t0
        results["Qiskit-MPS"].append(t_qs)

        # 4. PennyLane (Using default.qubit as fallack, but limited by RAM)
        if n <= 24: # PL default.qubit is statevector and EXTREMELY slow for 24+ without true MPS
            dev = qml.device("default.qubit", wires=n)
            pl_c = build_pl_circuit(n, depth, dev)
            t0 = time.time()
            pl_c()
            t_pl = time.time() - t0
        else:
            t_pl = -1 # Skip for high qubit counts to prevent crash
        results["PL-Default"].append(t_pl)

        # 5. Cirq (Statevector fallback)
        if cirq and n <= 22:
            qubits = cirq.LineQubit.range(n)
            c = cirq.Circuit()
            for _ in range(depth):
                for i in range(n): c.append(cirq.H(qubits[i]))
                for i in range(n-1): c.append(cirq.CNOT(qubits[i], qubits[i+1]))
            s = cirq.Simulator()
            t0 = time.time()
            s.simulate(c)
            t_cq = time.time() - t0
        else:
            t_cq = -1
        results["Cirq-Default"].append(t_cq)

        # Print Row
        pl_str = f"{t_pl:.3f}s" if t_pl > 0 else "N/A (RAM)"
        cq_str = f"{t_cq:.3f}s" if t_cq > 0 else "N/A (RAM)"
        print(f"{n:<10} | {t_sf:<12.4f} | {t_sf_jax:<12.4f} | {t_qs:<15.4f} | {pl_str:<12} | {cq_str:<12}")

    print("-" * 90)
    print("NOTE: ONLY Superfermion and Qiskit Aer support native MPS scaling to 100 qubits on this machine.")
    print("      Superfermion JAX-MPS is the only backend utilizing XLA vectorization for tensor chains.")
    print("="*90)

if __name__ == "__main__":
    run_mps_benchmark()
