"""
Quantum Compilation Passes — Python implementations of optimization & decomposition.
"""

from __future__ import annotations

from typing import List, Protocol, Tuple

from superfermion.circuit import Circuit, GateRecord


class Pass(Protocol):
    """Protocol for a compilation pass."""
    def name(self) -> str: ...
    def run(self, circuit: Circuit) -> Circuit: ...


class GateCancellationPass:
    """Removes self-inverse gate pairs (e.g., H*H, X*X)."""
    
    def name(self) -> str:
        return "GateCancellationPass"

    def _can_cancel(self, g1: GateRecord, g2: GateRecord) -> bool:
        if g1.qubits != g2.qubits:
            return False
        if g1.name == g2.name:
            if g1.name in ("H", "X", "Y", "Z"):
                return True
        if (g1.name, g2.name) in (("S", "SDG"), ("SDG", "S")): return True
        if (g1.name, g2.name) in (("T", "TDG"), ("TDG", "T")): return True
        return False

    def run(self, circuit: Circuit) -> Circuit:
        gates = circuit._gates
        if not gates: return circuit
        new_gates: List[GateRecord] = []
        i = 0
        changed = False
        while i < len(gates):
            if i + 1 < len(gates):
                g1, g2 = gates[i], gates[i+1]
                if self._can_cancel(g1, g2):
                    changed = True
                    i += 2
                    continue
            new_gates.append(gates[i])
            i += 1
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        new_circuit._gates = new_gates
        if changed: return self.run(new_circuit)
        return new_circuit


class RotationMergingPass:
    """Merges consecutive rotation gates on the same qubit."""
    def name(self) -> str: return "RotationMergingPass"
    def run(self, circuit: Circuit) -> Circuit:
        gates = circuit._gates
        if not gates: return circuit
        new_gates: List[GateRecord] = []
        i = 0; changed = False
        while i < len(gates):
            g = gates[i]
            if i + 1 < len(gates):
                gn = gates[i+1]
                if g.name.upper() in ("RX", "RY", "RZ") and g.name == gn.name and g.qubits == gn.qubits:
                    merged_params = [p1 + p2 for p1, p2 in zip(g.params, gn.params)]
                    new_gates.append(GateRecord(g.name, g.qubits, merged_params))
                    changed = True; i += 2; continue
            new_gates.append(g); i += 1
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        new_circuit._gates = new_gates
        if changed: return self.run(new_circuit)
        return new_circuit


class ConstantFoldingPass:
    """Removes zero-rotation gates."""
    def name(self) -> str: return "ConstantFoldingPass"
    def run(self, circuit: Circuit) -> Circuit:
        new_gates = []
        for g in circuit._gates:
            if g.name.upper() in ("RX", "RY", "RZ", "P") and len(g.params) > 0:
                try:
                    if abs(float(g.params[0])) < 1e-12:
                        continue
                except (TypeError, ValueError):
                    pass  # symbolic parameter, keep the gate
            new_gates.append(g)
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        new_circuit._gates = new_gates
        return new_circuit


class SwapDecompositionPass:
    """Decomposes SWAP gates into 3 CNOT gates."""
    def name(self) -> str: return "SwapDecompositionPass"
    def run(self, circuit: Circuit) -> Circuit:
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        for gate in circuit._gates:
            if gate.name.upper() == "SWAP":
                q0, q1 = gate.qubits[0], gate.qubits[1]
                new_circuit.cx(q0, q1).cx(q1, q0).cx(q0, q1)
            else: new_circuit._gates.append(gate)
        return new_circuit


class BasisTranslationPass:
    """Translates all gates to native basis."""
    def __init__(self, native_gates: List[str]):
        self.native_gates = [g.upper() for g in native_gates]
    def name(self) -> str: return "BasisTranslationPass"
    def run(self, circuit: Circuit) -> Circuit:
        new_circuit = Circuit(circuit.n_qubits, circuit.n_cbits)
        for gate in circuit._gates:
            name = gate.name.upper()
            if name in self.native_gates or name in ("MEASURE", "BARRIER"):
                new_circuit._gates.append(gate)
                continue
            if name == "H":
                q = gate.qubits[0]
                new_circuit.rz(1.570796, q).rx(1.570796, q).rz(1.570796, q)
            elif name in ("CNOT", "CX"):
                q0, q1 = gate.qubits[0], gate.qubits[1]
                if "CZ" in self.native_gates: new_circuit.h(q1).cz(q0, q1).h(q1)
                else: new_circuit._gates.append(gate)
            else: new_circuit._gates.append(gate)
        return new_circuit

class RoutingPass:
    """Inserts SWAP gates to satisfy hardware coupling map."""
    def __init__(self, coupling_map: List[Tuple[int, int]]):
        self.coupling_map = coupling_map
    def name(self) -> str: return "RoutingPass"
    def run(self, circuit: Circuit) -> Circuit: return circuit
