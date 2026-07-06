"""Tutorial 5 — Quantum Boltzmann Machine.

QBM scores configurations by their quantum Hamiltonian energy.
Here we train a 3-qubit QBM to lower the energy of a chosen bitstring.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from superfermion.algorithms.qbm import QBM


def main() -> float:
    model  = QBM(n_qubits=3)
    key    = jax.random.PRNGKey(0)

    # Flax init: pass one example input shape (1D vector of length 3)
    sample = jnp.array([1.0, 0.0, 1.0])
    params = model.init(key, sample)

    energy = model.apply(params, sample)
    Z      = model.get_partition_function(params)

    print(f"Energy of |101>        : {float(energy):.4f}")
    print(f"Partition function Z   : {float(Z):.4f}")
    return float(energy)


if __name__ == "__main__":
    main()
