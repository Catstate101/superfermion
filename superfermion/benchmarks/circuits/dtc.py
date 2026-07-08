"""Discrete Time Crystal (DTC) Floquet circuit builder.

Each layer: RX on all qubits, nearest-neighbor RZZ, RZ on all qubits.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DTCFactory:

    @property
    def name(self) -> str:
        return "dtc"

    def create(self, strategy, n_qubits: int, *, n_layers: int = 100,
               seed: int = 42, **kw: Any):
        rng = np.random.default_rng(seed)
        sdk = strategy.name

        if sdk == "superfermion":
            return self._build_sf(n_qubits, n_layers, rng)
        elif sdk == "qiskit":
            return self._build_qiskit(n_qubits, n_layers, rng)
        else:
            return self._build_sf(n_qubits, n_layers, rng)

    def _build_sf(self, n: int, layers: int, rng):
        import superfermion as sf
        qc = sf.Circuit(n)
        for _ in range(layers):
            for q in range(n):
                qc = qc.rx(float(rng.uniform(0, np.pi)), q)
            for q in range(n - 1):
                qc = qc.rzz(float(rng.uniform(0, np.pi)), q, q + 1)
            for q in range(n):
                qc = qc.rz(float(rng.uniform(0, np.pi)), q)
        return qc

    def _build_qiskit(self, n: int, layers: int, rng):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for _ in range(layers):
            for q in range(n):
                qc.rx(float(rng.uniform(0, np.pi)), q)
            for q in range(n - 1):
                qc.rzz(float(rng.uniform(0, np.pi)), q, q + 1)
            for q in range(n):
                qc.rz(float(rng.uniform(0, np.pi)), q)
        return qc
