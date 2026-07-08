"""Multi-controlled X (MCX) circuit builder.

Builds an n-controlled X gate using recursive V-chain decomposition
into {H, T, TDG, CX} gates.
"""

from __future__ import annotations

from typing import Any

import math


class MultiControlXFactory:

    @property
    def name(self) -> str:
        return "multi_control_x"

    def create(self, strategy, n_qubits: int = 0, *, n_controls: int = 16,
               **kw: Any):
        total = n_controls + 1
        sdk = strategy.name

        if sdk == "superfermion":
            return self._build_sf(n_controls, total)
        elif sdk == "qiskit":
            return self._build_qiskit(n_controls, total)
        else:
            return self._build_sf(n_controls, total)

    def _build_sf(self, n_controls: int, total: int):
        import superfermion as sf
        qc = sf.Circuit(total + max(0, n_controls - 2))
        target = n_controls
        controls = list(range(n_controls))
        ancillas = list(range(n_controls + 1, total + max(0, n_controls - 2)))
        self._mcx_vchain_sf(qc, controls, target, ancillas)
        return qc

    def _mcx_vchain_sf(self, qc, controls, target, ancillas):
        n = len(controls)
        if n == 0:
            qc.x(target)
        elif n == 1:
            qc.cx(controls[0], target)
        elif n == 2:
            qc.ccx(controls[0], controls[1], target)
        else:
            qc.ccx(controls[0], controls[1], ancillas[0])
            for i in range(2, n):
                anc_in = ancillas[i - 2]
                anc_out = ancillas[i - 1] if i < n - 1 else target
                qc.ccx(controls[i], anc_in, anc_out)
            for i in range(n - 2, 1, -1):
                anc_in = ancillas[i - 2]
                anc_out = ancillas[i - 1] if i < n - 1 else target
                qc.ccx(controls[i], anc_in, anc_out)
            qc.ccx(controls[0], controls[1], ancillas[0])

    def _build_qiskit(self, n_controls: int, total: int):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import MCXGate
        qc = QuantumCircuit(total + max(0, n_controls - 2))
        controls = list(range(n_controls))
        target = n_controls
        qc.mcx(controls, target)
        return qc
