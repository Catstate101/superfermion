
import time
import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional

try:
    import superfermion as sf
    from superfermion.circuit import Circuit
    from superfermion.backends.registry import get_backend
except ImportError:
    pass

class IndustrialBenchmark:
    """Production-grade benchmarks for industry use cases."""
    
    @staticmethod
    def get_vqe_ansatz(n_qubits: int):
        c = sf.Circuit(n_qubits)
        for i in range(n_qubits):
            c.h(i).ry(0.1, i).rz(0.2, i)
        for i in range(n_qubits - 1):
            c.cx(i, i+1)
        for i in range(n_qubits):
            c.ry(0.1, i)
        return c

    @staticmethod
    def get_qaoa_circuit(n_qubits: int):
        c = sf.Circuit(n_qubits)
        for i in range(n_qubits): c.h(i)
        for i in range(n_qubits - 1):
            c.cx(i, i+1).rz(0.5, i+1).cx(i, i+1) 
        for i in range(n_qubits): c.rx(0.3, i)
        return c

    @staticmethod
    def get_qml_circuit(n_qubits: int):
        c = sf.Circuit(n_qubits)
        for i in range(n_qubits): c.h(i).rx(0.1, i).ry(0.2, i).rz(0.3, i)
        for i in range(0, n_qubits - 1, 2): c.cx(i, i+1)
        for i in range(1, n_qubits - 1, 2): c.cx(i, i+1)
        return c

    @classmethod
    def run_showdown(cls, qubit_counts: List[int] = [10, 20, 30]):
        """Run the framework comparison showdown."""
        # Framework availability
        try:
            import qiskit
            from qiskit_aer import AerSimulator
            QISKIT_OK = True
        except: QISKIT_OK = False
        
        try:
            import pennylane as qml
            PL_OK = True
        except: PL_OK = False

        backends = ["jax_mps", "cuda_mps", "mps", "jax"]
        use_cases = [
            ("VQE (Chemistry)", cls.get_vqe_ansatz),
            ("QAOA (Finance)", cls.get_qaoa_circuit),
            ("QML (Neural Nets)", cls.get_qml_circuit)
        ]

        results = {}
        
        for name, gen in use_cases:
            case_results = {}
            for n in qubit_counts:
                n_results = {}
                # SF Backends
                for b_name in backends:
                    try:
                        if n > 25 and b_name == "jax":
                            n_results[b_name] = "Skip"
                            continue
                        circuit = gen(n)
                        backend = get_backend(b_name)
                        backend.run(circuit, shots=0) # Warmup
                        t0 = time.time()
                        backend.run(circuit, shots=0)
                        n_results[b_name] = (time.time() - t0) * 1000
                    except:
                        n_results[b_name] = "Fail"
                
                # Qiskit
                if QISKIT_OK:
                    try:
                        # Convert SF circuit to Qiskit (simplified for benchmark)
                        qc = qiskit.QuantumCircuit(n)
                        # We just rebuild the same pattern to be fair
                        if name == "VQE (Chemistry)":
                            for i in range(n): qc.h(i); qc.ry(0.1, i); qc.rz(0.2, i)
                            for i in range(n-1): qc.cx(i, i+1)
                            for i in range(n): qc.ry(0.1, i)
                        elif name == "QAOA (Finance)":
                            for i in range(n): qc.h(i)
                            for i in range(n-1): qc.rzz(0.5, i, i+1)
                            for i in range(n): qc.rx(0.3, i)
                        else:
                            for i in range(n): qc.h(i); qc.rx(0.1, i); qc.ry(0.2, i); qc.rz(0.3, i)
                            for i in range(0, n-1, 2): qc.cx(i, i+1)
                            for i in range(1, n-1, 2): qc.cx(i, i+1)
                        
                        method = 'matrix_product_state' if n > 16 else 'statevector'
                        sim = AerSimulator(method=method)
                        sim.run(qc, shots=0).result() # Warmup
                        t0 = time.time()
                        sim.run(qc, shots=0).result()
                        n_results["Qiskit"] = (time.time() - t0) * 1000
                    except: n_results["Qiskit"] = "Fail"
                
                # Pennylane
                if PL_OK:
                    try:
                        def pl_circuit():
                            if name == "VQE (Chemistry)":
                                for i in range(n): qml.Hadamard(i); qml.RY(0.1, i); qml.RZ(0.2, i)
                                for i in range(n-1): qml.CNOT([i, i+1])
                                for i in range(n): qml.RY(0.1, i)
                            elif name == "QAOA (Finance)":
                                for i in range(n): qml.Hadamard(i)
                                for i in range(n-1): qml.IsingZZ(0.5, [i, i+1])
                                for i in range(n): qml.RX(0.3, i)
                            else:
                                for i in range(n): qml.Hadamard(i); qml.RX(0.1, i); qml.RY(0.2, i); qml.RZ(0.3, i)
                                for i in range(0, n-1, 2): qml.CNOT([i, i+1])
                                for i in range(1, n-1, 2): qml.CNOT([i, i+1])
                            return qml.expval(qml.PauliZ(0))

                        dev = qml.device("default.qubit", wires=n)
                        qnode = qml.QNode(pl_circuit, dev)
                        qnode() # Warmup
                        t0 = time.time()
                        qnode()
                        n_results["PennyLane"] = (time.time() - t0) * 1000
                    except: n_results["PennyLane"] = "Fail"
                
                case_results[n] = n_results
            results[name] = case_results
        return results
