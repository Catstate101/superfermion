"""Tutorial 1 — VQE on H2.

Finds the ground-state energy of molecular hydrogen using the scipy-backed
VQE in ``superfermion.algorithms.variational`` and the UCCSD ansatz.
"""
from __future__ import annotations

import superfermion as sf
from superfermion.algorithms.variational import VQE
from superfermion.chemistry import get_molecular_hamiltonian, uccsd_ansatz


def main() -> float:
    H      = get_molecular_hamiltonian("H2")
    ansatz = uccsd_ansatz(n_qubits=4, n_electrons=2)

    vqe    = VQE(ansatz, H, backend="statevector")
    result = vqe.minimize(iterations=200)

    print(f"H2 ground-state energy (VQE): {result.optimal_value:.6f} Ha")
    print(f"Optimiser: {vqe.optimizer}    iterations used: {len(result.history)}")
    return float(result.optimal_value)


if __name__ == "__main__":
    main()
