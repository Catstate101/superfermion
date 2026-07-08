"""Quantum Volume circuit builder.

Builds a QV circuit: depth layers of random SU(4) unitaries on random
qubit pairs, decomposed into {RY, RZ, CX}.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class QuantumVolumeFactory:

    @property
    def name(self) -> str:
        return "quantum_volume"

    def create(self, strategy, n_qubits: int, *, depth: int = 0, seed: int = 42, **kw: Any):
        if depth == 0:
            depth = n_qubits
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
            perm = rng.permutation(n).tolist()
            for j in range(0, n - 1, 2):
                q0, q1 = perm[j], perm[j + 1]
                angles = rng.uniform(0, 2 * np.pi, size=6)
                qc = (qc
                      .ry(float(angles[0]), q0).rz(float(angles[1]), q0)
                      .ry(float(angles[2]), q1).rz(float(angles[3]), q1)
                      .cx(q0, q1)
                      .ry(float(angles[4]), q0).rz(float(angles[5]), q1))
        return qc

    def _build_qiskit(self, n: int, depth: int, rng):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for _ in range(depth):
            perm = rng.permutation(n).tolist()
            for j in range(0, n - 1, 2):
                q0, q1 = perm[j], perm[j + 1]
                angles = rng.uniform(0, 2 * np.pi, size=6)
                qc.ry(float(angles[0]), q0)
                qc.rz(float(angles[1]), q0)
                qc.ry(float(angles[2]), q1)
                qc.rz(float(angles[3]), q1)
                qc.cx(q0, q1)
                qc.ry(float(angles[4]), q0)
                qc.rz(float(angles[5]), q1)
        return qc
