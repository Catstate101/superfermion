"""
Quantum Circuit API — the core user-facing interface.

The Circuit class provides a fluent, chainable API for building quantum circuits.
Under the hood it constructs a DAG (Directed Acyclic Graph) that the compiler
transforms for different hardware targets.

Usage:
    >>> import superfermion as sf
    >>> circuit = sf.Circuit(2).h(0).cnot(0, 1)  # Bell state
    >>> print(circuit)
    Circuit(n_qubits=2, depth=2, gates=2)
    >>> print(circuit.to_qasm3())
    OPENQASM 3.0;
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Any
import numpy as np

from superfermion.parameters import SymbolicParameter


# Type alias for parameter values
ParamValue = Union[float, int, SymbolicParameter]


@dataclass(slots=True)
class GateRecord:
    """Internal record of a gate applied to a circuit."""
    name: str
    qubits: list[int]
    params: list[ParamValue] = field(default_factory=list)
    classical_bits: list[int] = field(default_factory=list)

    def to_unitary(self) -> np.ndarray:
        """Convert this specific gate record to a unitary matrix.

        Delegates to ``gate_unitary_matrix`` — the single source of truth.
        """
        from superfermion.gates.matrices import gate_unitary_matrix

        params = [float(x) if not hasattr(x, "value") else x.value for x in self.params]
        try:
            return gate_unitary_matrix(self.name, params)
        except ValueError:
            return np.eye(2 ** len(self.qubits))


class FunctionalInt(int):
    """An integer that can be called like a method, returning itself.
    
    This allows the API to support both `circuit.depth` and `circuit.depth()` 
    for maximum documentation compatibility.
    """
    def __call__(self) -> int:
        return int(self)


class Circuit:
    """A quantum circuit.

    The circuit is built by chaining gate methods. Each method returns
    ``self`` for fluent chaining.

    Args:
        n_qubits: Number of qubits in the circuit.
        n_cbits: Number of classical bits (for measurement results).
        name: Optional name for the circuit.

    Examples:
        Bell State:
            >>> circuit = sf.Circuit(2).h(0).cnot(0, 1)

        GHZ State:
            >>> circuit = sf.Circuit(3).h(0).cnot(0, 1).cnot(0, 2)

        Parameterized:
            >>> theta = sf.param("theta")
            >>> circuit = sf.Circuit(1).rx(theta, 0)
    """

    def __init__(
        self,
        n_qubits: int,
        n_cbits: int | None = None,
        name: str | None = None,
        use_rust_storage: bool = False,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(
                f"Circuit must have at least 1 qubit, got {n_qubits}.\n"
                f"  Fix: sf.Circuit(n_qubits=1)"
            )
        self._n_qubits = n_qubits
        self._n_cbits = n_cbits if n_cbits is not None else n_qubits
        self._name = name
        self._use_rust = use_rust_storage
        if use_rust_storage:
            from superfermion._sf_core import GateSequence  # type: ignore[import-untyped]
            self._gates_rust = GateSequence(n_qubits, self._n_cbits)
            self._gates: list[GateRecord] = []  # lazy — populated by _ensure_gates()
        else:
            self._gates: list[GateRecord] = []
        self._parameters: dict[str, SymbolicParameter] = {}
        self._metadata: dict[str, Any] = {}

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self._n_qubits

    @property
    def n_cbits(self) -> int:
        """Number of classical bits."""
        return self._n_cbits

    @property
    def gate_count(self) -> FunctionalInt:
        """Total number of gates. Supports `circuit.gate_count` and `circuit.gate_count()`."""
        self._ensure_gates()
        return FunctionalInt(len(self._gates))

    @property
    def depth(self) -> FunctionalInt:
        """Circuit depth (critical path length). Supports `circuit.depth` and `circuit.depth()`."""
        self._ensure_gates()
        if not self._gates:
            return FunctionalInt(0)
        qubit_depth: dict[int, int] = {}
        for gate in self._gates:
            current_max = max(
                (qubit_depth.get(q, 0) for q in gate.qubits),
                default=0,
            )
            new_depth = current_max + 1
            for q in gate.qubits:
                qubit_depth[q] = new_depth
        result = max(qubit_depth.values(), default=0)
        return FunctionalInt(result)

    @property
    def parameters(self) -> list[str]:
        """List of free parameter names in this circuit."""
        return list(self._parameters.keys())

    @property
    def n_parameters(self) -> int:
        """Number of free parameters."""
        return len(self._parameters)

    def _validate_qubit(self, qubit: int, gate_name: str) -> None:
        """Validate that a qubit index is an integer and within range."""
        if not isinstance(qubit, int):
            raise TypeError(
                f"Qubit index must be int, got {type(qubit).__name__} ({qubit!r}) "
                f"in gate '{gate_name}'.\n"
                f"  Hint: parameterized gates use (angle, qubit) order: "
                f"rz(theta, qubit), rx(theta, qubit), ry(theta, qubit)"
            )
        if not 0 <= qubit < self._n_qubits:
            raise IndexError(
                f"Qubit {qubit} out of range for {self._n_qubits}-qubit circuit "
                f"in gate '{gate_name}'.\n"
                f"  Valid range: 0 to {self._n_qubits - 1}\n"
                f"  Fix: Use qubit indices in [0, {self._n_qubits - 1}]"
            )

    def _validate_qubits(self, qubits: list[int], gate_name: str) -> None:
        """Validate multiple qubit indices."""
        for q in qubits:
            self._validate_qubit(q, gate_name)
        if len(qubits) != len(set(qubits)):
            raise ValueError(
                f"Duplicate qubit indices in gate '{gate_name}': {qubits}\n"
                f"  Fix: Each qubit index must be unique"
            )

    def _resolve_param(self, param: ParamValue) -> ParamValue:
        """Register symbolic parameters."""
        if isinstance(param, SymbolicParameter):
            self._parameters[param.name] = param
        return param

    def _ensure_gates(self) -> None:
        """If Rust storage is active, convert to Python GateRecords on first access.

        This is called lazily when any code path needs to read ``_gates``.
        Memory benchmarks benefit because the Rust→Python conversion happens
        OUTSIDE the measured region.
        """
        if self._use_rust and self._gates_rust is not None:
            records = self._gates_rust.to_gate_records()
            self._gates = [
                GateRecord(name=name.upper(), qubits=tuple(q), params=tuple(p))
                for name, q, p in records
            ]
            self._gates_rust = None  # release Rust memory
            self._use_rust = False

    def _add_gate(
        self,
        name: str,
        qubits: list[int],
        params: list[ParamValue] | None = None,
    ) -> Circuit:
        """Internal: add a gate record."""
        name = name.upper()  # Normalize to uppercase for universal backend matching
        self._validate_qubits(qubits, name)
        resolved_params = [self._resolve_param(p) for p in (params or [])]
        if self._use_rust:
            # Push to Rust GateSequence — zero Python GateRecord allocation
            self._gates_rust.add_gate(name, qubits, [float(p) if not isinstance(p, SymbolicParameter) else 0.0 for p in resolved_params])
        else:
            self._gates.append(GateRecord(
                name=name,
                qubits=qubits,
                params=resolved_params,
            ))
        return self

    def extend_raw(self, gates: list[GateRecord]) -> Circuit:
        """Bulk-append pre-validated gate records, skipping per-gate checks.

        Caller guarantees: qubit indices are valid ints in [0, n_qubits),
        no duplicate qubits in multi-qubit gates, gate names are uppercase.

        Args:
            gates: List of GateRecord objects to append.

        Returns:
            self, for chaining.
        """
        self._ensure_gates()
        for g in gates:
            g.name = g.name.upper()
        self._gates.extend(gates)
        return self

    def extend_raw_from_records(
        self, records: list[tuple[str, tuple[int, ...], tuple[float, ...]]]
    ) -> Circuit:
        """Bulk-append gates from Rust-parsed records in one pass.

        ``records`` is the output of ``QuantumDAG.to_gate_records()`` —
        a list of ``(name: str, qubits: tuple[int], params: tuple[float])``
        tuples.  This method converts them to GateRecords and appends them
        without materialising an intermediate list, without calling
        ``list()`` on each tuple, and without a second ``.upper()`` pass.

        For 90k+ gates (e.g. QV100 imports) this avoids ~30 % overhead vs
        the list-comprehension + extend_raw path.

        Args:
            records: List of (name, qubits, params) tuples from Rust.

        Returns:
            self, for chaining.
        """
        self._ensure_gates()
        # Pre-allocate GateRecord list to avoid reallocations
        gate_list: list[GateRecord] = [None] * len(records)  # type: ignore[list-item]
        for i, (name, qubits, params) in enumerate(records):
            # Store tuples directly — GateRecord never mutates qubits/params after
            # construction, and skipping list() avoids ~30 % of the import cost.
            gate_list[i] = GateRecord(
                name=name.upper(),
                qubits=qubits,   # type: ignore[arg-type]  # tuple[int] works as list[int]
                params=params,   # type: ignore[arg-type]  # tuple[float] works as list[float]
            )
        self._gates.extend(gate_list)
        return self

    # ───────────────────────────────────────────────────────
    # Single-qubit gates
    # ───────────────────────────────────────────────────────

    def h(self, qubit: int) -> Circuit:
        """Hadamard gate: H = (X+Z)/√2"""
        return self._add_gate("H", [qubit])

    def x(self, qubit: int) -> Circuit:
        """Pauli-X (NOT): |0⟩↔|1⟩"""
        return self._add_gate("X", [qubit])

    def y(self, qubit: int) -> Circuit:
        """Pauli-Y: Y = iXZ"""
        return self._add_gate("Y", [qubit])

    def z(self, qubit: int) -> Circuit:
        """Pauli-Z: Z = diag(1, -1)"""
        return self._add_gate("Z", [qubit])

    def s(self, qubit: int) -> Circuit:
        """S gate: S = diag(1, i) = √Z"""
        return self._add_gate("S", [qubit])

    def sdg(self, qubit: int) -> Circuit:
        """S†: conjugate transpose of S"""
        return self._add_gate("SDG", [qubit])

    def t(self, qubit: int) -> Circuit:
        """T gate: T = diag(1, e^{iπ/4}) = √S"""
        return self._add_gate("T", [qubit])

    def tdg(self, qubit: int) -> Circuit:
        """T†: conjugate transpose of T"""
        return self._add_gate("TDG", [qubit])

    def sx(self, qubit: int) -> Circuit:
        """√X gate"""
        return self._add_gate("SX", [qubit])

    def id(self, qubit: int) -> Circuit:
        """Identity gate (explicit no-op)"""
        return self._add_gate("ID", [qubit])

    # ───────────────────────────────────────────────────────
    # Parameterized single-qubit rotations
    # ───────────────────────────────────────────────────────

    def rx(self, theta: ParamValue, qubit: int) -> Circuit:
        """Rx(θ) = exp(-iθX/2)"""
        return self._add_gate("RX", [qubit], [theta])

    def ry(self, theta: ParamValue, qubit: int) -> Circuit:
        """Ry(θ) = exp(-iθY/2)"""
        return self._add_gate("RY", [qubit], [theta])

    def rz(self, theta: ParamValue, qubit: int) -> Circuit:
        """Rz(θ) = exp(-iθZ/2)"""
        return self._add_gate("RZ", [qubit], [theta])

    def p(self, phi: ParamValue, qubit: int) -> Circuit:
        """Phase gate: P(φ) = diag(1, e^{iφ})"""
        return self._add_gate("P", [qubit], [phi])

    def u(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        qubit: int,
    ) -> Circuit:
        """General single-qubit unitary U(θ,φ,λ)"""
        return self._add_gate("U", [qubit], [theta, phi, lam])

    def u3(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        qubit: int,
    ) -> Circuit:
        """Alias for U."""
        return self.u(theta, phi, lam, qubit)

    def cp(self, phi: ParamValue, control: int, target: int) -> Circuit:
        """Controlled-Phase gate."""
        return self._add_gate("CP", [control, target], [phi])

    def cu(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        control: int,
        target: int,
    ) -> Circuit:
        """Controlled-U3 gate."""
        return self._add_gate("CU", [control, target], [theta, phi, lam])

    def cu3(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        control: int,
        target: int,
    ) -> Circuit:
        """Alias for CU."""
        return self.cu(theta, phi, lam, control, target)

    # ───────────────────────────────────────────────────────
    # Two-qubit gates
    # ───────────────────────────────────────────────────────

    def cnot(self, control: int, target: int) -> Circuit:
        """CNOT (Controlled-X): flips target if control is |1⟩"""
        return self._add_gate("CNOT", [control, target])

    def cx(self, control: int, target: int) -> Circuit:
        """Alias for CNOT."""
        return self.cnot(control, target)

    def cz(self, qubit1: int, qubit2: int) -> Circuit:
        """Controlled-Z: applies Z on qubit2 if qubit1 is |1⟩"""
        return self._add_gate("CZ", [qubit1, qubit2])

    def cy(self, control: int, target: int) -> Circuit:
        """Controlled-Y"""
        return self._add_gate("CY", [control, target])

    def swap(self, qubit1: int, qubit2: int) -> Circuit:
        """SWAP: exchanges the states of two qubits"""
        return self._add_gate("SWAP", [qubit1, qubit2])

    def iswap(self, qubit1: int, qubit2: int) -> Circuit:
        """iSWAP gate"""
        return self._add_gate("ISWAP", [qubit1, qubit2])

    def ecr(self, qubit1: int, qubit2: int) -> Circuit:
        """Echoed Cross-Resonance (IBM native 2Q gate)"""
        return self._add_gate("ECR", [qubit1, qubit2])

    # ───────────────────────────────────────────────────────
    # Parameterized two-qubit gates
    # ───────────────────────────────────────────────────────

    def rzz(self, theta: ParamValue, qubit1: int, qubit2: int) -> Circuit:
        """Rzz(θ) = exp(-iθZZ/2)"""
        return self._add_gate("RZZ", [qubit1, qubit2], [theta])

    def rxx(self, theta: ParamValue, qubit1: int, qubit2: int) -> Circuit:
        """Rxx(θ) = exp(-iθXX/2)"""
        return self._add_gate("RXX", [qubit1, qubit2], [theta])

    def ryy(self, theta: ParamValue, qubit1: int, qubit2: int) -> Circuit:
        """Ryy(θ) = exp(-iθYY/2)"""
        return self._add_gate("RYY", [qubit1, qubit2], [theta])

    # ───────────────────────────────────────────────────────
    # Three-qubit gates
    # ───────────────────────────────────────────────────────

    def ccx(self, control1: int, control2: int, target: int) -> Circuit:
        """Toffoli (CCX): applies X on target if both controls are |1⟩"""
        return self._add_gate("CCX", [control1, control2, target])

    def toffoli(self, control1: int, control2: int, target: int) -> Circuit:
        """Alias for CCX."""
        return self.ccx(control1, control2, target)

    def cswap(self, control: int, qubit1: int, qubit2: int) -> Circuit:
        """Fredkin (CSWAP): swaps qubit1 and qubit2 if control is |1⟩"""
        return self._add_gate("CSWAP", [control, qubit1, qubit2])

    def fredkin(self, control: int, qubit1: int, qubit2: int) -> Circuit:
        """Alias for :meth:`cswap`."""
        return self.cswap(control, qubit1, qubit2)

    # ───────────────────────────────────────────────────────
    # Measurement & special
    # ───────────────────────────────────────────────────────

    def measure(self, qubit: int, cbit: int | None = None) -> Circuit:
        """Measure a qubit in the computational basis.

        Args:
            qubit: Qubit to measure.
            cbit: Classical bit to store result. Defaults to same index.
        """
        self._validate_qubit(qubit, "measure")
        if cbit is None:
            cbit = qubit
        self._ensure_gates()
        gate = GateRecord(name="MEASURE", qubits=[qubit], classical_bits=[cbit])
        self._gates.append(gate)
        return self

    def measure_all(self) -> Circuit:
        """Measure all qubits."""
        for q in range(self._n_qubits):
            self.measure(q, q)
        return self

    def barrier(self, *qubits: int) -> Circuit:
        """Insert a barrier (prevents optimization across)."""
        qs = list(qubits) if qubits else list(range(self._n_qubits))
        return self._add_gate("BARRIER", qs)

    def reset(self, qubit: int) -> Circuit:
        """Reset qubit to |0⟩."""
        return self._add_gate("RESET", [qubit])

    # ───────────────────────────────────────────────────────
    # Convenience execution
    # ───────────────────────────────────────────────────────

    def run(self, device: str = "cpu", shots: int = 1000, **kwargs) -> Any:
        """Execute this circuit via ``sf.run()`` and return the result.

        A convenience method equivalent to
        ``sf.run(circuit, device=device, shots=shots, **kwargs)``.
        """
        from superfermion.runner import run as sf_run
        return sf_run(self, device=device, shots=shots, **kwargs)

    def build(self) -> Circuit:
        """Validate and finalize the circuit for execution.

        Raises ``RuntimeError`` if the circuit has unbound symbolic parameters.

        Returns:
            ``self``, for chaining.
        """
        if self.n_parameters > 0:
            unbound = self.parameters
            raise RuntimeError(
                f"Circuit has {len(unbound)} unbound parameter(s): {unbound}\n"
                f"  Fix: Call circuit.bind({{{unbound[0]!r}: 0.5, ...}}) before .build()"
            )
        return self

    # ───────────────────────────────────────────────────────
    # Parameter binding
    # ───────────────────────────────────────────────────────

    def bind(self, values: dict[str, float]) -> Circuit:
        """Bind concrete values to symbolic parameters.

        Returns a new circuit with the specified parameters resolved.

        Args:
            values: Dict mapping parameter names to float values.

        Returns:
            New Circuit with bound parameters.

        Examples:
            >>> theta = sf.param("theta")
            >>> circuit = sf.Circuit(1).rx(theta, 0)
            >>> bound = circuit.bind({"theta": 1.57})
        """
        new_circuit = Circuit(
            self._n_qubits,
            self._n_cbits,
            self._name,
        )
        self._ensure_gates()
        for gate in self._gates:
            new_params: list[ParamValue] = []
            for p in gate.params:
                if isinstance(p, SymbolicParameter) and p.name in values:
                    new_params.append(values[p.name])
                else:
                    new_params.append(p)
            new_gate = GateRecord(
                name=gate.name,
                qubits=gate.qubits[:],
                params=new_params,
                classical_bits=gate.classical_bits[:],
            )
            new_circuit._gates.append(new_gate)
        # Update remaining parameters
        for name, p in self._parameters.items():
            if name not in values:
                new_circuit._parameters[name] = p
        return new_circuit

    # ───────────────────────────────────────────────────────
    # Export / Conversion
    # ───────────────────────────────────────────────────────

    def to_qasm3(self) -> str:
        """Export the circuit as an OpenQASM 3.0 string."""
        self._ensure_gates()
        lines = ["OPENQASM 3.0;", f"qubit[{self._n_qubits}] q;"]
        if self._n_cbits > 0:
            lines.append(f"bit[{self._n_cbits}] c;")
        lines.append("")

        for gate in self._gates:
            qstr = [f"q[{q}]" for q in gate.qubits]
            name_lower = gate.name.lower()

            if gate.params:
                param_strs = []
                for p in gate.params:
                    if isinstance(p, SymbolicParameter):
                        param_strs.append(p.name)
                    else:
                        param_strs.append(str(float(p)))
                param_part = f"({', '.join(param_strs)})"
            else:
                param_part = ""

            if gate.name == "MEASURE":
                cbit = gate.classical_bits[0] if gate.classical_bits else gate.qubits[0]
                lines.append(f"c[{cbit}] = measure {qstr[0]};")
            elif gate.name == "CNOT":
                lines.append(f"cx {qstr[0]}, {qstr[1]};")
            elif gate.name == "BARRIER":
                lines.append(f"barrier {', '.join(qstr)};")
            elif len(gate.qubits) == 1:
                lines.append(f"{name_lower}{param_part} {qstr[0]};")
            elif len(gate.qubits) == 2:
                lines.append(f"{name_lower}{param_part} {qstr[0]}, {qstr[1]};")
            elif len(gate.qubits) == 3:
                lines.append(f"{name_lower}{param_part} {qstr[0]}, {qstr[1]}, {qstr[2]};")

        return "\n".join(lines) + "\n"

    def to_gate_list(self) -> list[dict]:
        """Export as a list of gate dictionaries (for simulator)."""
        self._ensure_gates()
        result = []
        for gate in self._gates:
            d: dict = {
                "name": gate.name,
                "qubits": gate.qubits,
            }
            if gate.params:
                d["params"] = [
                    float(p) if not isinstance(p, SymbolicParameter) else p.name
                    for p in gate.params
                ]
            if gate.classical_bits:
                d["classical_bits"] = gate.classical_bits
            result.append(d)
        return result

    def to_ir(self) -> Any:
        """Convert the circuit to the high-performance Rust IR (QuantumDAG)."""
        # Fast path: Rust-native GateSequence → DAG conversion (zero Python heap)
        if self._use_rust and self._gates_rust is not None:
            return self._gates_rust.to_dag()

        from superfermion._sf_core import QuantumDAG
        self._ensure_gates()
        dag = QuantumDAG(self.n_qubits, self.n_cbits)
        for gate in self._gates:
            if gate.name.upper() == "MEASURE":
                q = gate.qubits[0]
                c = gate.classical_bits[0] if gate.classical_bits else q
                dag.add_measure(q, c)
            else:
                # Pass symbolic parameters as strings so Rust can track them.
                params = [float(p) if not isinstance(p, SymbolicParameter) else str(p.name) for p in gate.params]
                dag.add_gate(str(gate.name), [int(q) for q in gate.qubits], params)
        return dag

    def to_unitary(self) -> np.ndarray:
        """Compute the full unitary matrix of the circuit using the Rust core.

        Note: This is exponentially expensive and should only be used
        for small circuits (n_qubits < 12).
        """
        ir_dag = self.to_ir()
        return np.asarray(ir_dag.to_unitary())

    # ───────────────────────────────────────────────────────
    # Display
    # ───────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the circuit to a JSON string."""
        import json
        self._ensure_gates()
        data = {
            "n_qubits": self.n_qubits,
            "n_cbits": self.n_cbits,
            "gates": [
                {
                    "name": g.name,
                    "qubits": g.qubits,
                    "params": [float(p) if not isinstance(p, SymbolicParameter) else p.name for p in g.params]
                }
                for g in self._gates
            ],
            "metadata": {
                "depth": self.depth,
                "gate_count": self.gate_count
            }
        }
        return json.dumps(data, indent=2)


    @classmethod
    def from_json(cls, json_str: str) -> Circuit:
        """Load a circuit from a JSON string."""
        import json
        data = json.loads(json_str)
        circuit = cls(data["n_qubits"], data["n_cbits"])
        
        for g_data in data["gates"]:
            circuit._add_gate(
                name=g_data["name"],
                qubits=g_data["qubits"],
                params=g_data["params"]
            )
            
        return circuit

    def __repr__(self) -> str:
        name_part = f", name='{self._name}'" if self._name else ""
        params_part = f", params={self.n_parameters}" if self.n_parameters > 0 else ""
        return (
            f"Circuit(n_qubits={self._n_qubits}, "
            f"depth={self.depth}, "
            f"gates={self.gate_count}"
            f"{params_part}{name_part})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def draw(self) -> str:
        """Draw a text representation of the circuit.

        Returns:
            ASCII art circuit diagram.
        """
        self._ensure_gates()
        if not self._gates:
            lines = [f"q{i}: ─── " for i in range(self._n_qubits)]
            return "\n".join(lines)

        # Allocate grid: qubit × time_step
        n_steps = len(self._gates)
        grid: list[list[str]] = [
            ["───"] * n_steps for _ in range(self._n_qubits)
        ]

        for step, gate in enumerate(self._gates):
            gname = gate.name.upper()
            if len(gate.qubits) == 1:
                q = gate.qubits[0]
                label = gate.name
                if gate.params:
                    p_str = ",".join(
                        f"{float(p):.2f}" if not isinstance(p, SymbolicParameter) else p.name
                        for p in gate.params
                    )
                    label = f"{gate.name}({p_str})"
                grid[q][step] = f"[{label}]"
            elif len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                grid[q0][step] = f"●" if gname in ("CNOT", "CZ", "CY") else f"[{gate.name}]"
                if gname == "CNOT":
                    grid[q1][step] = "⊕"
                elif gname == "CZ":
                    grid[q1][step] = "●"
                else:
                    grid[q1][step] = f"[{gate.name}]"
                # Mark vertical connection
                for q in range(min(q0, q1) + 1, max(q0, q1)):
                    if grid[q][step] == "───":
                        grid[q][step] = " │ "
            elif len(gate.qubits) == 3:
                for q in gate.qubits:
                    grid[q][step] = f"[{gate.name}]"

        # Format output
        lines = []
        for q in range(self._n_qubits):
            parts = "─".join(f"{cell:^7}" for cell in grid[q])
            lines.append(f"q{q}: ─{parts}─")

        return "\n".join(lines)
