"""
CirqStrategy — SDKStrategy implementation for Google Cirq.

Import-guarded: if cirq is not installed, this module raises ImportError
and the strategy is simply not registered.
"""

from __future__ import annotations

from typing import Any, Dict

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm


class CirqStrategy:

    @property
    def name(self) -> str:
        return "cirq"

    @property
    def version(self) -> str:
        return cirq.__version__

    def build_circuit(self, factory, n_qubits: int, **kwargs: Any):
        return factory.create(self, n_qubits, **kwargs)

    def load_qasm(self, qasm_str: str):
        return circuit_from_qasm(qasm_str)

    def compile(self, circuit, backend, level: int = 1):
        return cirq.optimize_for_target_gateset(
            circuit,
            gateset=cirq.CZTargetGateset(),
        )

    def count_ops(self, circuit) -> Dict[str, int]:
        ops: Dict[str, int] = {}
        for op in circuit.all_operations():
            name = type(op.gate).__name__ if op.gate else str(op)
            ops[name] = ops.get(name, 0) + 1
        return ops

    def gate_count(self, circuit) -> int:
        return len(list(circuit.all_operations()))

    def depth(self, circuit) -> int:
        return len(circuit)

    def n_qubits(self, circuit) -> int:
        return len(circuit.all_qubits())

    def bind_parameters(self, circuit, values: Dict[str, float]):
        import sympy
        resolver = {sympy.Symbol(k): v for k, v in values.items()}
        return cirq.resolve_parameters(circuit, resolver)

    def n_parameters(self, circuit) -> int:
        return len(circuit._parameter_names_() if hasattr(circuit, '_parameter_names_') else [])
