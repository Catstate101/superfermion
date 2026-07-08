"""EfficientSU2 parametric ansatz builder.

Pattern: circular entanglement with RY/RZ rotation layers + CX entanglement.
Matches Qiskit's ``EfficientSU2`` template.
"""

from __future__ import annotations

from typing import Any


class EfficientSU2Factory:

    @property
    def name(self) -> str:
        return "efficient_su2"

    def create(self, strategy, n_qubits: int, *, reps: int = 4, **kw: Any):
        sdk = strategy.name
        if sdk == "superfermion":
            return self._build_sf(n_qubits, reps)
        elif sdk == "qiskit":
            return self._build_qiskit(n_qubits, reps)
        else:
            return self._build_sf(n_qubits, reps)

    def _build_sf(self, n: int, reps: int):
        import superfermion as sf
        qc = sf.Circuit(n)
        idx = 0
        for rep in range(reps + 1):
            for q in range(n):
                qc = qc.ry(sf.param(f"ry_{idx}"), q)
                idx += 1
                qc = qc.rz(sf.param(f"rz_{idx}"), q)
                idx += 1
            if rep < reps:
                for q in range(n):
                    qc = qc.cx(q, (q + 1) % n)
        return qc

    def _build_qiskit(self, n: int, reps: int):
        from qiskit.circuit.library import EfficientSU2 as QiskitSU2
        return QiskitSU2(n, reps=reps, entanglement="circular").decompose()
