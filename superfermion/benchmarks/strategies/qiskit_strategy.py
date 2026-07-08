"""
QiskitStrategy — SDKStrategy implementation for Qiskit.

Import-guarded: if qiskit is not installed, this module raises ImportError
and the strategy is simply not registered.
"""

from __future__ import annotations

from typing import Any, Dict

from qiskit import QuantumCircuit, __version__ as _qiskit_version
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap as QiskitCouplingMap


class QiskitStrategy:

    @property
    def name(self) -> str:
        return "qiskit"

    @property
    def version(self) -> str:
        return _qiskit_version

    def build_circuit(self, factory, n_qubits: int, **kwargs: Any):
        return factory.create(self, n_qubits, **kwargs)

    def load_qasm(self, qasm_str: str):
        return QuantumCircuit.from_qasm_str(qasm_str)

    def compile(self, circuit, backend, level: int = 1):
        cmap = QiskitCouplingMap(couplinglist=list(backend.coupling_map))
        return transpile(
            circuit,
            coupling_map=cmap,
            basis_gates=list(backend.basis_gates),
            optimization_level=min(level, 3),
        )

    def count_ops(self, circuit) -> Dict[str, int]:
        return dict(circuit.count_ops())

    def gate_count(self, circuit) -> int:
        return circuit.size()

    def depth(self, circuit) -> int:
        return circuit.depth()

    def n_qubits(self, circuit) -> int:
        return circuit.num_qubits

    def bind_parameters(self, circuit, values: Dict[str, float]):
        from qiskit.circuit import Parameter
        param_map = {}
        for p in circuit.parameters:
            if p.name in values:
                param_map[p] = values[p.name]
        return circuit.assign_parameters(param_map)

    def n_parameters(self, circuit) -> int:
        return circuit.num_parameters
