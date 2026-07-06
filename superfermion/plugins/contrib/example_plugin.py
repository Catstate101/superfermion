"""
Minimal example plugin demonstrating the plugin registration system.

Usage:
    >>> sf.discover_plugins()
    >>> backend = sf.get_backend("example_debug_backend")
"""

from superfermion.plugins import register_backend, register_pass
from superfermion.backends.base import Backend
from superfermion.results import RunResult
from superfermion.compiler.passes import Pass
from superfermion.circuit import Circuit


@register_backend("example_debug_backend")
class ExampleDebugBackend(Backend):
    """Example backend that prints every gate during execution."""

    def __init__(self):
        super().__init__("example_debug_backend")

    @property
    def n_qubits(self) -> int:
        return 32

    @property
    def supported_gates(self):
        return ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ",
                "CX", "CNOT", "CZ", "SWAP", "MEASURE"]

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs):
        for gate in circuit._gates:
            print(f"[ExampleDebug] Gate: {gate.name} on qubits {gate.qubits}")
        return RunResult(
            counts={}, shots=shots, circuit=circuit,
            metadata={"backend": "example_debug"}
        )

    def expval(self, circuit, observable, **kwargs):
        return 0.0


@register_pass("example_debug_pass")
class ExampleDebugPass(Pass):
    """Example compiler pass that prints circuit stats."""

    def run(self, circuit: Circuit) -> Circuit:
        print(f"[ExampleDebugPass] Circuit: {circuit.n_qubits} qubits, "
              f"{len(circuit._gates)} gates, depth {circuit.depth}")
        return circuit
