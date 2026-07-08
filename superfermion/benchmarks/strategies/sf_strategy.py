"""
SuperfermionStrategy — SDKStrategy implementation for Superfermion.

Pattern: Strategy
Problem: SF circuit/compiler API must be callable through the same interface
         as Qiskit/Cirq for fair benchmark comparison.
Solution: Thin adapter that maps protocol methods to sf.* calls.
"""

from __future__ import annotations

from typing import Any, Dict

import superfermion as sf
from superfermion.compiler.specs import HardwareSpec


class SuperfermionStrategy:

    @property
    def name(self) -> str:
        return "superfermion"

    @property
    def version(self) -> str:
        return sf.__version__

    def build_circuit(self, factory, n_qubits: int, **kwargs: Any):
        return factory.create(self, n_qubits, **kwargs)

    def load_qasm(self, qasm_str: str):
        from superfermion.bridge import from_qasm
        return from_qasm(qasm_str)

    def compile(self, circuit, backend, level: int = 1):
        spec = HardwareSpec(
            name="benchmark_target",
            n_qubits=backend.n_qubits,
            native_gates=list(backend.basis_gates),
            coupling_map=list(backend.coupling_map),
            basis_gates=list(backend.basis_gates),
        )
        return sf.compile(circuit, level=level, target=spec)

    def count_ops(self, circuit) -> Dict[str, int]:
        return circuit.count_ops()

    def gate_count(self, circuit) -> int:
        return int(circuit.gate_count)

    def depth(self, circuit) -> int:
        return int(circuit.depth)

    def n_qubits(self, circuit) -> int:
        return circuit.n_qubits

    def bind_parameters(self, circuit, values: Dict[str, float]):
        return circuit.bind(values)

    def n_parameters(self, circuit) -> int:
        return circuit.n_parameters
