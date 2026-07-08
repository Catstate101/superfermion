"""Random Clifford circuit builder.

Generates random circuits from the Clifford gate set {H, S, CX}.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomCliffordFactory:

    @property
    def name(self) -> str:
        return "random_clifford"

    def create(self, strategy, n_qubits: int, *, depth: int = 0,
               seed: int = 42, **kw: Any):
        if depth == 0:
            depth = 2 * n_qubits
        rng = np.random.default_rng(seed)
        sdk = strategy.name

        if sdk == "superfermion":
            return self._build_sf(n_qubits, depth, rng)
        elif sdk == "qiskit":
            return self._build_qiskit(n_qubits, depth, rng)
        else:
            return self._build_sf(n_qubits, depth, rng)

    def _build_sf(self, n: int, depth: int, rng):
        import superfermion as sf
        qc = sf.Circuit(n)
        for _ in range(depth):
            for q in range(n):
                gate = rng.choice(["h", "s", "id"])
                if gate == "h":
                    qc = qc.h(q)
                elif gate == "s":
                    qc = qc.s(q)
            for q in range(0, n - 1, 2):
                if rng.random() > 0.3:
                    qc = qc.cx(q, q + 1)
            for q in range(1, n - 1, 2):
                if rng.random() > 0.3:
                    qc = qc.cx(q, q + 1)
        return qc

    def _build_qiskit(self, n: int, depth: int, rng):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for _ in range(depth):
            for q in range(n):
                gate = rng.choice(["h", "s", "id"])
                if gate == "h":
                    qc.h(q)
                elif gate == "s":
                    qc.s(q)
            for q in range(0, n - 1, 2):
                if rng.random() > 0.3:
                    qc.cx(q, q + 1)
            for q in range(1, n - 1, 2):
                if rng.random() > 0.3:
                    qc.cx(q, q + 1)
        return qc
