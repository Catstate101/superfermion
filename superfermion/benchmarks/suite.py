"""
Superfermion Standard Benchmarking Suite.

Provides systematic performance evaluation for VQE and QAOA algorithms
across various circuit sizes and backend configurations.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any

import superfermion as sf
from superfermion.algorithms.variational import VQE, QAOA
from superfermion.observables.core import PauliString, Hamiltonian
# Use the canonical function from ansatz/__init__.py — importing the submodule
# with the same name shadows the function in the package namespace.
from superfermion.qml.ansatz import hardware_efficient as hardware_efficient_ansatz


class BenchmarkSuite:
    """Orchestrates standard quantum benchmarks for Superfermion."""

    @staticmethod
    def run_vqe_benchmark(n_qubits: int = 4, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark VQE on a transverse-field Ising-like Hamiltonian."""
        print(f"Running VQE Benchmark ({n_qubits} qubits)...")
        
        # H = sum Z_i Z_{i+1} + sum X_i
        terms = []
        for i in range(n_qubits - 1):
            s = ["I"] * n_qubits
            s[i] = "Z"
            s[i+1] = "Z"
            terms.append(PauliString("".join(s), coeffs=1.0))
        for i in range(n_qubits):
            s = ["I"] * n_qubits
            s[i] = "X"
            terms.append(PauliString("".join(s), coeffs=0.5))
            
        h = Hamiltonian(terms)
        ansatz = hardware_efficient_ansatz(n_qubits, layers=1)
        vqe = VQE(ansatz, h, optimizer="L-BFGS-B")

        start_time = time.perf_counter()
        results = vqe.minimize(iterations=iterations)
        duration = time.perf_counter() - start_time

        n_iters = len(results.history) if results.history else 1
        return {
            "n_qubits": n_qubits,
            "final_energy": results.optimal_value,
            "duration_sec": duration,
            "ms_per_iter": (duration / n_iters) * 1000,
        }

    @staticmethod
    def run_qaoa_benchmark(n_qubits: int = 4, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark QAOA on a random MaxCut instance."""
        print(f"Running QAOA Benchmark ({n_qubits} qubits)...")
        
        # H_cost = sum Z_i Z_j for a linear chain
        terms = []
        for i in range(n_qubits - 1):
            s = ["I"] * n_qubits
            s[i] = "Z"
            s[i+1] = "Z"
            terms.append(PauliString("".join(s), coeffs=1.0))
            
        edges = [(i, i + 1, 1.0) for i in range(n_qubits - 1)]
        qaoa = QAOA(n_qubits, edges, p_layers=1, optimizer="L-BFGS-B")

        start_time = time.perf_counter()
        results = qaoa.minimize(iterations=iterations)
        duration = time.perf_counter() - start_time

        n_iters = len(results.history) if results.history else 1
        return {
            "n_qubits": n_qubits,
            "min_energy": results.optimal_value,
            "duration_sec": duration,
            "ms_per_iter": (duration / n_iters) * 1000,
        }

def run_all_benchmarks():
    print("=== Superfermion Performance Report ===\n")
    
    # 2, 4, 8 Qubit VQE
    vqe_results = []
    for n in [2, 4, 8]:
        res = BenchmarkSuite.run_vqe_benchmark(n)
        vqe_results.append(res)
        print(f"  {n} Qubits: {res['ms_per_iter']:.2f} ms/iter")
        
    # 2, 4, 8 Qubit QAOA
    qaoa_results = []
    for n in [2, 4, 8]:
        res = BenchmarkSuite.run_qaoa_benchmark(n)
        qaoa_results.append(res)
        print(f"  {n} Qubits: {res['ms_per_iter']:.2f} ms/iter")
        
    return vqe_results, qaoa_results

if __name__ == "__main__":
    run_all_benchmarks()
