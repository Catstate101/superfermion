"""Tutorial 2 — QAOA for MaxCut on a 4-node ring.

Uses the scipy-backed QAOA. ``edges`` is the graph; QAOA constructs the
cost Hamiltonian ZZ_{ij} for every edge internally.
"""
from __future__ import annotations

from superfermion.algorithms.variational import QAOA


def main() -> float:
    # 4-node ring: 0-1-2-3-0. Max-cut value is 4.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    qaoa   = QAOA(n_qubits=4, edges=edges, p_layers=2)
    result = qaoa.minimize()

    print(f"QAOA optimal value: {result.optimal_value:.4f}")
    print(f"Optimal angles   : {result.optimal_params}")
    return float(result.optimal_value)


if __name__ == "__main__":
    main()
