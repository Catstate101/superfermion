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

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from superfermion.parameters import SymbolicParameter

# Type alias for parameter values
ParamValue = float | int | SymbolicParameter

# Glyphs used by :meth:`Circuit.draw` for the optional Unicode rendering.
_DIAGRAM_GLYPHS = "─●⊕│"


def _stdout_supports_unicode() -> bool:
    """Whether ``sys.stdout`` can encode :data:`_DIAGRAM_GLYPHS`.

    Legacy Windows code pages (cp1252) and ``PYTHONIOENCODING=ascii`` cannot,
    so ``print(circuit.draw())`` used to die with ``UnicodeEncodeError`` after
    the circuit had already been built. Returning ``False`` here makes
    :meth:`Circuit.draw` emit ASCII instead.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        _DIAGRAM_GLYPHS.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


@dataclass(slots=True)
class GateRecord:
    """Internal record of a gate applied to a circuit."""
    name: str
    qubits: list[int]
    params: list[ParamValue] = field(default_factory=list)
    classical_bits: list[int] = field(default_factory=list)
    condition: tuple[int, int] | None = field(default=None)
    matrix: np.ndarray | None = field(default=None, repr=False)

    def to_unitary(self) -> np.ndarray:
        """Convert this specific gate record to a unitary matrix.

        If this gate stores an opaque unitary matrix, return it directly.
        Otherwise delegates to ``gate_unitary_matrix``.

        Ordering: the returned matrix is indexed over the gate's own qubits in
        the order returned by :attr:`qubits`, with ``qubits[0]`` as the *most*
        significant bit — i.e. the textbook layout ``|q0 q1⟩``. This is the
        opposite of :meth:`Circuit.to_unitary`, which puts qubit 0 in the least
        significant bit (the Qiskit convention). For single-qubit gates the two
        agree; for a multi-qubit record they are bit-reversed with respect to
        each other, so embed a record matrix into a circuit by bit-reversing it
        first::

            local = record.to_unitary()
            k = len(record.qubits)
            perm = [int(format(i, f"0{k}b")[::-1], 2) for i in range(2**k)]
            circuit_local = local[np.ix_(perm, perm)]
        """
        if self.matrix is not None:
            return self.matrix

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


# OpenQASM 3's stdgates.inc has no entry for these names, so ``to_qasm3``
# emits a gate declaration for each one the circuit uses.  Every body is
# matrix-equivalent to the gate it defines (verified against Qiskit's Operator
# up to global phase), and uses only the builtin ``U`` and ``cx``, so a reader
# that ships a different stdgates.inc still accepts the program.
_QASM3_GATE_DEFS = {
    "iswap": (
        "gate iswap a, b {\n"
        "  U(0.0, 0.0, 1.5707963267948966) a;\n"
        "  U(1.5707963267948966, 0.0, 3.141592653589793) a;\n"
        "  U(0.0, 0.0, 1.5707963267948966) b;\n"
        "  cx a, b;\n"
        "  cx b, a;\n"
        "  U(1.5707963267948966, 0.0, 3.141592653589793) b;\n"
        "}"
    ),
    "ecr": (
        "gate ecr a, b {\n"
        "  U(0.0, 0.0, 1.5707963267948966) a;\n"
        "  U(1.5707963267948966, -1.5707963267948966, 1.5707963267948966) b;\n"
        "  cx a, b;\n"
        "  U(3.141592653589793, 0.0, 3.141592653589793) a;\n"
        "}"
    ),
    "rxx": (
        "gate rxx(theta) a, b {\n"
        "  U(1.5707963267948966, 0.0, 3.141592653589793) a;\n"
        "  U(1.5707963267948966, 0.0, 3.141592653589793) b;\n"
        "  cx a, b;\n"
        "  U(0.0, 0.0, theta) b;\n"
        "  cx a, b;\n"
        "  U(1.5707963267948966, 0.0, 3.141592653589793) a;\n"
        "  U(1.5707963267948966, 0.0, 3.141592653589793) b;\n"
        "}"
    ),
    "ryy": (
        "gate ryy(theta) a, b {\n"
        "  U(1.5707963267948966, -1.5707963267948966, 1.5707963267948966) a;\n"
        "  U(1.5707963267948966, -1.5707963267948966, 1.5707963267948966) b;\n"
        "  cx a, b;\n"
        "  U(0.0, 0.0, theta) b;\n"
        "  cx a, b;\n"
        "  U(-1.5707963267948966, -1.5707963267948966, 1.5707963267948966) a;\n"
        "  U(-1.5707963267948966, -1.5707963267948966, 1.5707963267948966) b;\n"
        "}"
    ),
    "rzz": (
        "gate rzz(theta) a, b {\n"
        "  cx a, b;\n"
        "  U(0.0, 0.0, theta) b;\n"
        "  cx a, b;\n"
        "}"
    ),
}

# QASM 3 builtin / stdgates spellings for circuit record names.
_QASM3_GATE_NAMES = {"u": "U", "u3": "U", "r1": "p"}


def _coerce_qubit_index(value):
    """Return an int for anything indexable (NumPy ints), else unchanged.

    ``isinstance(np.int64(0), int)`` is False, so the validator used to reject
    NumPy scalar indices even though they are unambiguous integers.
    """
    if isinstance(value, int):
        return value
    try:
        import operator

        return operator.index(value)
    except TypeError:
        return value


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
        # Classical condition (cbit, value) waiting for the next gate
        # (set by ``c_if``, consumed by the next ``_add_gate`` call).
        self._pending_condition: tuple[int, int] | None = None
        # Set when an opaque unitary spans more than three qubits.  The
        # 0.1.15 Rust statevector kernel applies only arities 1-3; to_ir()
        # returns a compatibility proxy for these circuits (see _fixes.py).
        self._has_large_unitary = False
        # Matrices for opaque unitaries held in the Rust ``GateSequence``.
        # ``to_gate_records()`` reports name/qubits/params only, so the
        # matrix is mirrored here (keyed by gate index) for serialization.
        self._rust_unitary_matrices: dict[int, Any] = {}
        # Compiled-engine artifact caches (e.g. the Rust stabilizer state
        # built by ``to_ir()`` → ``simulate_to_state``).  ``_engine_cache_ver``
        # increments on every gate mutation; entries record the revision they
        # were produced for, so stale entries are ignored after a change.
        self._engine_cache_ver = 0
        self._engine_caches: dict[tuple, tuple[int, Any]] = {}

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

    def count_ops(self) -> dict[str, int]:
        """Count gate operations by name.

        Returns:
            Dictionary mapping gate name to its occurrence count.
        """
        self._ensure_gates()
        ops: dict[str, int] = {}
        for g in self._gates:
            ops[g.name] = ops.get(g.name, 0) + 1
        return ops

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

    def _invalidate_engine_caches(self) -> None:
        """Drop compiled-engine artifacts (called by every gate mutation)."""
        self._engine_cache_ver += 1
        self._engine_caches.clear()

    def _engine_cache_get(self, key: tuple) -> Any:
        """Return the cached engine artifact for ``key``, or None if stale."""
        entry = self._engine_caches.get(key)
        if entry is None or entry[0] != self._engine_cache_ver:
            return None
        return entry[1]

    def _engine_cache_put(self, key: tuple, value: Any) -> None:
        """Cache an engine artifact for the circuit's current revision."""
        self._engine_caches[key] = (self._engine_cache_ver, value)

    def _add_gate(
        self,
        name: str,
        qubits: list[int],
        params: list[ParamValue] | None = None,
    ) -> Circuit:
        """Internal: add a gate record."""
        name = name.upper()  # Normalize to uppercase for universal backend matching
        qubits = [_coerce_qubit_index(q) for q in qubits]
        self._validate_qubits(qubits, name)
        resolved_params = [self._resolve_param(p) for p in (params or [])]
        condition = self._pending_condition
        self._pending_condition = None
        if self._use_rust and (condition is not None
                               or any(isinstance(p, SymbolicParameter) for p in resolved_params)):
            # Rust GateSequence can carry neither classical conditions nor
            # symbolic parameter names (it stores a placeholder 0.0). Convert to
            # Python GateRecords (lazy) so bind() can substitute the symbols.
            self._ensure_gates()
        if self._use_rust:
            # Push to Rust GateSequence — zero Python GateRecord allocation
            numeric = [
                float(p) if not isinstance(p, SymbolicParameter) else 0.0
                for p in resolved_params
            ]
            self._gates_rust.add_gate(name, qubits, numeric)
        else:
            self._gates.append(GateRecord(
                name=name,
                qubits=qubits,
                params=resolved_params,
                condition=condition,
            ))
        self._invalidate_engine_caches()
        return self

    def _restore_gate_fields(
        self,
        record: GateRecord,
        classical_bits: list[int] | None,
        condition: list[int] | tuple[int, int] | None,
    ) -> None:
        """Reattach the classical fields a serialized gate dict carries.

        ``to_json``/``.sfc`` store ``classical_bits`` and ``condition`` next
        to ``name``/``qubits``/``params``.  Rebuilding a record through
        ``_add_gate`` can express neither, so a ``measure(1, 0)`` used to come
        back as ``measure(1, 1)`` and every guarded gate lost its ``c_if``.
        Widths are validated the same way ``measure()``/``c_if()`` validate
        them: an out-of-range classical bit panics inside ``to_ir()``.
        """
        if classical_bits:
            bits = [int(c) for c in classical_bits]
            for cbit in bits:
                if not 0 <= cbit < self._n_cbits:
                    raise ValueError(
                        f"serialized classical bit {cbit} out of range "
                        f"(circuit has {self._n_cbits} cbits)"
                    )
            record.classical_bits = bits
        if condition is not None:
            cbit, value = int(condition[0]), int(condition[1])
            if not 0 <= cbit < self._n_cbits:
                raise ValueError(
                    f"serialized condition bit {cbit} out of range "
                    f"(circuit has {self._n_cbits} cbits)"
                )
            record.condition = (cbit, value)

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
        if any(g.matrix is not None and len(g.qubits) > 3 for g in gates):
            self._has_large_unitary = True
        for g in gates:
            g.name = g.name.upper()
            if self._pending_condition is not None:
                raise ValueError(
                    "extend_raw() cannot run while a c_if(...) condition is "
                    "pending; add the gated gate before bulk-extending"
                )
        self._gates.extend(gates)
        self._invalidate_engine_caches()
        return self

    def extend_raw_from_records(
        self, records: list[tuple]
    ) -> Circuit:
        """Bulk-append gates from Rust-parsed records in one pass.

        ``records`` is the output of ``QuantumDAG.to_gate_records()`` —
        a list of ``(name: str, qubits: tuple[int], params: tuple[float])``
        tuples.  This method converts them to GateRecords and appends them
        without materialising an intermediate list, without calling
        ``list()`` on each tuple, and without a second ``.upper()`` pass.

        A fourth element, ``condition: (cbit, value) | None``, is accepted
        for ``QuantumDAG.to_gate_records_with_conditions()`` so a gate
        guarded by OpenQASM 2 ``if (creg == value)`` keeps its condition.
        A fifth element, ``classical_bits: list[int]``, keeps explicit
        measurement targets (``measure q[1] -> c[0]``).

        For 90k+ gates (e.g. QV100 imports) this avoids ~30 % overhead vs
        the list-comprehension + extend_raw path.

        Args:
            records: List of (name, qubits, params[, condition[, classical_bits]])
                tuples.

        Returns:
            self, for chaining.
        """
        self._ensure_gates()
        # Pre-allocate GateRecord list to avoid reallocations
        gate_list: list[GateRecord] = [None] * len(records)  # type: ignore[list-item]
        if self._pending_condition is not None:
            raise ValueError(
                "extend_raw_from_records() cannot run while a c_if(...) "
                "condition is pending; add the gated gate before bulk-extending"
            )
        for i, record in enumerate(records):
            classical_bits = None
            if len(record) == 5:
                name, qubits, params, condition, classical_bits = record
            elif len(record) == 4:
                name, qubits, params, condition = record
            else:
                name, qubits, params = record
                condition = None
            # Store tuples directly — GateRecord never mutates qubits/params after
            # construction, and skipping list() avoids ~30 % of the import cost.
            gate_list[i] = GateRecord(
                name=name.upper(),
                qubits=qubits,   # type: ignore[arg-type]  # tuple[int] works as list[int]
                params=params,   # type: ignore[arg-type]  # tuple[float] works as list[float]
                condition=condition,
                classical_bits=classical_bits or [],
            )
        self._gates.extend(gate_list)
        self._invalidate_engine_caches()
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

    def sxdg(self, qubit: int) -> Circuit:
        """√X†: conjugate transpose of :meth:`sx` (the OpenQASM ``sxdg`` gate).

        The importer already produced ``SXDG`` records from ``sxdg`` in QASM,
        but there was no method to build one directly, so a parsed circuit
        could not be rebuilt through the public API. Matrices, export and
        compilation are unchanged; this only adds the missing constructor.
        """
        return self._add_gate("SXDG", [qubit])

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

    def r1(self, phi: ParamValue, qubit: int) -> Circuit:
        """R1(φ) = diag(1, e^{iφ}) — the OpenQASM ``r1`` gate.

        Numerically identical to :meth:`p` (up to nothing: the IR shares one
        matrix builder for both); it exists so circuits read out of QASM or
        written against the Qiskit ``r1`` name need no rewriting.
        """
        return self._add_gate("R1", [qubit], [phi])

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
        gamma: ParamValue = 0.0,
    ) -> Circuit:
        """Controlled-U3 gate with optional Qiskit-style gamma phase.

        Matrix (control first): ``diag(I, e^{iγ}·U3(θ,φ,λ))``. ``gamma=0``
        reproduces the plain controlled-U3 held by older Superfermion
        versions, so existing positional calls are unchanged.
        """
        return self._add_gate("CU", [control, target], [theta, phi, lam, gamma])

    def cu3(
        self,
        theta: ParamValue,
        phi: ParamValue,
        lam: ParamValue,
        control: int,
        target: int,
        gamma: ParamValue = 0.0,
    ) -> Circuit:
        """Alias for CU (accepts the same optional ``gamma``)."""
        return self.cu(theta, phi, lam, control, target, gamma)

    def crx(self, theta: ParamValue, control: int, target: int) -> Circuit:
        """Controlled-Rx: applies Rx(θ) on ``target`` if ``control`` is |1⟩."""
        return self._add_gate("CRX", [control, target], [theta])

    def crz(self, theta: ParamValue, control: int, target: int) -> Circuit:
        """Controlled-Rz: applies Rz(θ) on ``target`` if ``control`` is |1⟩.

        ``diag(1, 1, e^{-iθ/2}, e^{iθ/2})`` — a *phase* difference between the
        two control branches, which is why it is not the same as :meth:`cp`
        (``cp`` is ``diag(1, 1, 1, e^{iφ})``).
        """
        return self._add_gate("CRZ", [control, target], [theta])

    def cry(self, theta: ParamValue, control: int, target: int) -> Circuit:
        """Controlled-Ry: applies Ry(θ) on ``target`` if ``control`` is |1⟩.

        The IR has no dedicated ``CRy`` opcode, so this emits the exact
        equivalent :meth:`cu` with ``(θ, 0, 0)``: ``U3(θ,0,0) == Ry(θ)``, and
        a controlled block sees the phase, so the equality is exact — not up
        to a global phase. Adjoint and parameter-shift treat it as ``CU``,
        which is fully covered by both rule tables.
        """
        return self.cu(theta, 0.0, 0.0, control, target)

    def ch(self, control: int, target: int) -> Circuit:
        """Controlled-Hadamard: applies H on ``target`` if ``control`` is |1⟩.

        ``U3(π/2, 0, π)`` is exactly the Hadamard matrix — the U3 convention
        already carries the phase, so this is a true controlled-H and not a
        global-phase-shifted variant. Emitted as ``CU`` because the IR has no
        dedicated ``CH`` opcode; adjoint and parameter-shift both cover ``CU``.
        """
        return self.cu(np.pi / 2, 0.0, np.pi, control, target)

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

    def c_if(self, cbit: int, value: int = 1) -> Circuit:
        """Classical feed-forward: apply the NEXT operation only if classical
        bit ``cbit`` equals ``value``.

        Mid-circuit measurement analogue of PennyLane's
        ``qml.cond(m == value, op)`` (OpenQASM 3 ``if (c[cbit] == value)``
        semantics). The condition is attached to the next operation added
        after this call (gate, reset, or measurement), e.g.

        >>> circ = sf.Circuit(2).h(0).measure(0, 0).c_if(0, 1).x(1)

        Measures the state of qubit 0 into classical bit 0, then applies
        X on qubit 1 only when the outcome was 1.

        Args:
            cbit: Classical bit to test (must be < ``n_cbits``).
            value: Expected bit value (0 or 1).

        Returns:
            ``self``, for fluent chaining.
        """
        if value not in (0, 1):
            raise ValueError(
                f"c_if value must be 0 or 1, got {value!r}.\n"
                f"  Fix: circuit.c_if(cbit, 0) or circuit.c_if(cbit, 1)"
            )
        if cbit < 0 or cbit >= self._n_cbits:
            raise ValueError(
                f"c_if classical bit {cbit} out of range "
                f"(circuit has {self._n_cbits} cbits).\n"
                f"  Fix: sf.Circuit(n_qubits, n_cbits=...)"
            )
        if self._pending_condition is not None:
            raise ValueError(
                f"c_if({cbit}, {value}) called while a previous condition "
                f"{self._pending_condition} is still pending.\n"
                f"  Fix: add the gated gate before chaining another c_if"
            )
        self._pending_condition = (cbit, value)
        return self

    def measure(self, qubit: int, cbit: int | None = None) -> Circuit:
        """Measure a qubit in the computational basis.

        When the measured qubit is used again later (a gate, reset, or a
        second measurement), the measurement is *mid-circuit*: the qubit
        collapses onto the sampled outcome and the result is stored in the
        classical bit for classical feed-forward (see :meth:`c_if`).
        Measurements at the very end of a circuit remain terminal sampling
        instructions.

        A pending :meth:`c_if` guards the measurement itself: the qubit is
        only measured (and the classical bit only updated) when the condition
        holds.  This is Qiskit's ``if_test`` form and OpenQASM 3's
        ``if (c[0] == true) c[0] = measure q[0];``; :meth:`to_qasm` writes
        the equivalent OpenQASM 2 ``if(c==1) measure q[0] -> c[0];``.

        Args:
            qubit: Qubit to measure.
            cbit: Classical bit to store result. Defaults to same index.
        """
        condition = self._pending_condition
        self._pending_condition = None
        qubit = _coerce_qubit_index(qubit)
        if cbit is not None:
            cbit = _coerce_qubit_index(cbit)
        self._validate_qubit(qubit, "measure")
        if cbit is None:
            cbit = qubit
        # Same guard `c_if` already applies: without it `to_ir()` writes an
        # out-of-range classical bit and sf-ir *panics*, which surfaces as a
        # PyO3 PanicException (a BaseException, so `except Exception` misses
        # it).  Checked here so the failure is a catchable ValueError.
        if not 0 <= cbit < self._n_cbits:
            raise ValueError(
                f"measure() classical bit {cbit} out of range "
                f"(circuit has {self._n_cbits} cbits).\n"
                f"  Fix: sf.Circuit(n_qubits, n_cbits=...) or measure(q, c) "
                f"with 0 <= c < {self._n_cbits}"
            )
        self._ensure_gates()
        gate = GateRecord(
            name="MEASURE", qubits=[qubit], classical_bits=[cbit],
            condition=condition,
        )
        self._gates.append(gate)
        self._invalidate_engine_caches()
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

    def unitary(self, matrix: np.ndarray, qubits: list[int]) -> Circuit:
        """Apply an arbitrary unitary matrix as a single opaque gate.

        The matrix is stored without decomposition. During ``sf.run()``,
        it is applied directly to the statevector (one matrix multiply).
        During ``sf.compile(target=...)``, it is decomposed into basis
        gates by ``UnitaryDecompositionPass``.

        Bit order: ``matrix``'s index bit ``r`` drives ``qubits[k-1-r]``, so
        ``qubits[0]`` is the **most significant** bit of the matrix index.  The
        circuit's own state vector and :meth:`to_unitary` use the opposite
        (Qiskit) convention, qubit 0 as least significant, so this method
        applies ``B @ matrix @ B`` as an operator on the circuit, where ``B``
        is the ``k``-qubit bit-reversal permutation.  For ``k == 1`` the two
        agree exactly (max difference 0.000e+00 against Qiskit); for ``k >= 2``
        they are different operators, separated by up to 1.30 (``k == 2``) and
        1.06 (``k == 3``) in max-norm on random Haar unitaries, and by 1.137 /
        0.992 on the fixed matrices the engine harness uses -- the exact number
        is a property of the matrix, but it is O(1), not rounding.  To apply
        ``matrix`` in :meth:`to_unitary` order -- the same operator Qiskit
        applies -- pass the wires reversed::

            circuit.unitary(matrix, [q2, q1, q0])   # q2 is the MSB of matrix

        and to keep the wire order and reverse the operator instead, bit-reverse
        the matrix itself::

            perm = [int(format(i, f"0{k}b")[::-1], 2) for i in range(2 ** k)]
            circuit.unitary(matrix[np.ix_(perm, perm)], [q0, q1, q2])

        This is the relationship :meth:`GateRecord.to_unitary` documents from
        the other side; it is stated here as well because a caller reaching for
        ``unitary()`` has no reason to read the record's docstring first.  The
        algorithms in this package that build a matrix in ``qubits`` order (HHL,
        amplitude estimation) therefore pass ``[control] + reversed(targets)``.

        Args:
            matrix: A unitary matrix of shape ``(2^k, 2^k)`` where
                ``k = len(qubits)``.
            qubits: The qubit indices this unitary acts on, most significant
                matrix bit first.

        Returns:
            ``self``, for fluent chaining.

        Raises:
            ValueError: the shape does not match ``len(qubits)``, the matrix is
                not unitary, a qubit index is out of range, or a ``c_if(...)``
                condition is pending (classical conditions are only supported
                for named gates).
        """
        matrix = np.asarray(matrix, dtype=complex)
        k = len(qubits)
        expected = 2 ** k
        if matrix.shape != (expected, expected):
            raise ValueError(
                f"Unitary matrix shape {matrix.shape} doesn't match "
                f"{k} qubits (expected ({expected}, {expected}))"
            )
        check = matrix @ matrix.conj().T
        if not np.allclose(check, np.eye(expected), atol=1e-6):
            raise ValueError("Matrix is not unitary (U @ U^dagger != I)")

        self._validate_qubits(qubits, "UNITARY")
        if k > 3:
            self._has_large_unitary = True

        if self._pending_condition is not None:
            raise ValueError(
                "c_if(...) cannot precede unitary(): classical conditions are "
                "only supported for named gates.\n"
                "  Fix: decompose the matrix into basis gates and apply c_if "
                "to the first one"
            )

        if self._use_rust and self._gates_rust is not None:
            self._gates_rust.add_unitary(qubits, matrix)
            self._rust_unitary_matrices[
                int(self._gates_rust.gate_count()) - 1
            ] = np.asarray(matrix, dtype=complex)
        else:
            self._gates.append(GateRecord(
                name="UNITARY",
                qubits=qubits,
                matrix=matrix,
            ))
        self._invalidate_engine_caches()
        return self

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
                condition=gate.condition,
                matrix=gate.matrix,
            )
            new_circuit._gates.append(new_gate)
        # Update remaining parameters
        for name, p in self._parameters.items():
            if name not in values:
                new_circuit._parameters[name] = p
        return new_circuit

    def param_occurrences(self, name: str) -> list[tuple[str, int]]:
        """Every place one parameter appears, in gate order.

        A single symbol may drive several gates (``ry(t, 0); ry(t, 1)``) or
        several slots of one gate (``u(t, t, 0.1, 0)``).  Gradient rules
        must differentiate each occurrence separately, so they need the list
        rather than the name alone.

        Args:
            name: Parameter name (or ``SymbolicParameter``).

        Returns:
            ``[(gate_name, slot), ...]`` — ``slot`` is the index of the
            parameter inside that gate's parameter vector.  Empty when the
            name does not occur in the circuit.

        Examples:
            >>> t = sf.param("t")
            >>> sf.Circuit(1).ry(t, 0).rz(t, 0).param_occurrences("t")
            [('RY', 0), ('RZ', 0)]
        """
        target = name.name if isinstance(name, SymbolicParameter) else name
        self._ensure_gates()
        out: list[tuple[str, int]] = []
        for gate in self._gates:
            for slot, p in enumerate(gate.params):
                if isinstance(p, SymbolicParameter) and p.name == target:
                    out.append((gate.name, slot))
        return out

    def bind_occurrence(
        self,
        values: dict[str, float],
        name: str,
        k: int,
        value: float | None = None,
    ) -> Circuit:
        """Bind parameters like :meth:`bind`, but shift only one occurrence.

        ``bind()`` is name-global: binding ``t`` moves *every* gate that uses
        ``t`` at once, which silently halves the gradient of a circuit that
        reuses a name.  This method binds occurrence ``k`` of ``name`` to
        ``value`` while every other occurrence of ``name`` takes
        ``values[name]``, so a finite difference or a parameter shift sees
        one gate move at a time.

        Args:
            values: Base values (as passed to :meth:`bind`).  ``values[name]``
                is what every *other* occurrence of ``name`` receives.
            name: Parameter name whose occurrence is being moved.
            k: Occurrence index, as returned by :meth:`param_occurrences`.
            value: Value for occurrence ``k`` (default: ``values[name]``,
                i.e. identical to :meth:`bind`).

        Returns:
            New Circuit.

        Raises:
            IndexError: if occurrence ``k`` does not exist.

        Examples:
            >>> t = sf.param("t")
            >>> c = sf.Circuit(1).ry(t, 0).ry(t, 0)
            >>> c.bind_occurrence({"t": 0.35}, "t", 0, value=0.35 + 1.5)
        """
        target = name.name if isinstance(name, SymbolicParameter) else name
        n_occ = len(self.param_occurrences(target))
        if not 0 <= k < n_occ:
            raise IndexError(
                f"Occurrence {k} of parameter {target!r} does not exist "
                f"(the circuit has {n_occ} occurrence(s))."
            )
        if value is None:
            value = values.get(target)
        new_circuit = Circuit(
            self._n_qubits,
            self._n_cbits,
            self._name,
        )
        self._ensure_gates()
        occurrence = 0
        for gate in self._gates:
            new_params: list[ParamValue] = []
            for p in gate.params:
                if not isinstance(p, SymbolicParameter) or p.name != target:
                    if isinstance(p, SymbolicParameter) and p.name in values:
                        new_params.append(values[p.name])
                    else:
                        new_params.append(p)
                    continue
                if occurrence == k:
                    new_params.append(p if value is None else value)
                else:
                    new_params.append(values[p.name] if p.name in values else p)
                occurrence += 1
            new_circuit._gates.append(GateRecord(
                name=gate.name,
                qubits=gate.qubits[:],
                params=new_params,
                classical_bits=gate.classical_bits[:],
                condition=gate.condition,
                matrix=gate.matrix,
            ))
        # A name stays free only if some occurrence was left symbolic.
        for nm, p in self._parameters.items():
            if nm != target and nm not in values:
                new_circuit._parameters[nm] = p
        if target not in values:
            new_circuit._parameters[target] = self._parameters.get(
                target, SymbolicParameter(target)
            )
        return new_circuit

    # ───────────────────────────────────────────────────────
    # Export / Conversion
    # ───────────────────────────────────────────────────────

    def to_qasm3(self) -> str:
        """Export the circuit as an OpenQASM 3.0 string."""
        self._ensure_gates()
        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";',
                 f"qubit[{self._n_qubits}] q;"]
        used = {g.name.lower() for g in self._gates}
        for def_name in ("iswap", "ecr", "rxx", "ryy", "rzz"):
            if def_name in used:
                lines.append(_QASM3_GATE_DEFS[def_name])
        if self._n_cbits > 0:
            lines.append(f"bit[{self._n_cbits}] c;")
        lines.append("")

        for gate in self._gates:
            qstr = [f"q[{q}]" for q in gate.qubits]
            name_lower = gate.name.lower()
            display = _QASM3_GATE_NAMES.get(name_lower, name_lower)

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

            # Classical feed-forward: OpenQASM 3 compares a bit against a
            # boolean literal; `== 1` is rejected by Qiskit's importer
            # ("conditions must be 'bit == const bool'").
            prefix = ""
            if gate.condition is not None:
                value = "true" if gate.condition[1] else "false"
                prefix = f"if (c[{gate.condition[0]}] == {value}) "

            if gate.name == "MEASURE":
                cbit = gate.classical_bits[0] if gate.classical_bits else gate.qubits[0]
                lines.append(f"{prefix}c[{cbit}] = measure {qstr[0]};")
            elif gate.name == "CNOT":
                lines.append(f"{prefix}cx {qstr[0]}, {qstr[1]};")
            elif gate.name == "BARRIER":
                lines.append(f"{prefix}barrier {', '.join(qstr)};")
            elif gate.name == "SXDG":
                # Not present in every stdgates.inc: inline SXdg = H Sdg H
                # rather than emit a name the reader may not define.
                lines.append(f"{prefix}h {qstr[0]};")
                lines.append(f"{prefix}sdg {qstr[0]};")
                lines.append(f"{prefix}h {qstr[0]};")
            elif len(gate.qubits) == 1:
                lines.append(f"{prefix}{display}{param_part} {qstr[0]};")
            elif len(gate.qubits) == 2:
                lines.append(f"{prefix}{display}{param_part} {qstr[0]}, {qstr[1]};")
            elif len(gate.qubits) == 3:
                lines.append(f"{prefix}{display}{param_part} {qstr[0]}, {qstr[1]}, {qstr[2]};")

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
            if gate.condition is not None:
                d["condition"] = list(gate.condition)
            result.append(d)
        return result

    def to_ir(self) -> Any:
        """Convert the circuit to the high-performance Rust IR (QuantumDAG)."""
        # Fast path: Rust-native GateSequence → DAG conversion (zero Python heap)
        if self._use_rust and self._gates_rust is not None:
            dag = self._gates_rust.to_dag()
            if getattr(self, "_has_large_unitary", False):
                from superfermion._fixes import LargeUnitaryDAG
                return LargeUnitaryDAG(dag, int(self.n_qubits))
            return dag

        from superfermion._sf_core import QuantumDAG
        self._ensure_gates()
        dag = QuantumDAG(self.n_qubits, self.n_cbits)
        for gate in self._gates:
            if gate.name.upper() == "MEASURE":
                q = gate.qubits[0]
                c = gate.classical_bits[0] if gate.classical_bits else q
                if gate.condition is not None:
                    dag.add_measure_cond(
                        q, c, int(gate.condition[0]), int(gate.condition[1]),
                    )
                else:
                    dag.add_measure(q, c)
            elif gate.matrix is not None:
                dag.add_unitary([int(q) for q in gate.qubits], gate.matrix)
            else:
                params = [
                    float(p) if not isinstance(p, SymbolicParameter) else str(p.name)
                    for p in gate.params
                ]
                if gate.condition is not None:
                    dag.add_gate_cond(
                        str(gate.name), [int(q) for q in gate.qubits], params,
                        int(gate.condition[0]), int(gate.condition[1]),
                    )
                else:
                    dag.add_gate(str(gate.name), [int(q) for q in gate.qubits], params)
        if getattr(self, "_has_large_unitary", False):
            from superfermion._fixes import LargeUnitaryDAG
            return LargeUnitaryDAG(dag, int(self.n_qubits))
        return dag

    def to_unitary(self) -> np.ndarray:
        """Compute the full unitary matrix of the circuit using the Rust core.

        Note: This is exponentially expensive and should only be used
        for small circuits (n_qubits < 12).

        The matrix is little-endian — qubit 0 is the least significant bit of
        the row/column index, the same convention as Qiskit's ``Operator`` and
        the same order ``sf.run(...).statevector`` uses. Pauli strings are
        written the other way round (leftmost character = qubit 0), so
        ``"XI"`` and the basis index ``0b01`` refer to the same qubit.

        Raises:
            ValueError: the circuit contains a measurement, a reset or a
                ``c_if``-guarded gate.  Those are not unitary, and the Rust
                matrix builder silently skipped them — a guarded gate was
                even applied unconditionally.
        """
        dynamic: list[str] = []
        if getattr(self, "_use_rust", False) and getattr(self, "_gates_rust", None) is not None:
            # A Rust ``GateSequence`` cannot hold a condition or a measurement
            # (both force a conversion to Python records) but it does accept
            # RESET.  Reading it without ``_ensure_gates()`` keeps the mirrored
            # matrix of an opaque >3-qubit UNITARY alive for the
            # LargeUnitaryDAG proxy that ``to_ir()`` builds.
            for name, _qubits, _params in self._gates_rust.to_gate_records():
                if name.upper() in ("MEASURE", "RESET"):
                    dynamic.append(name.upper())
        else:
            for gate in self._gates:
                name = gate.name.upper()
                if gate.condition is not None:
                    dynamic.append(f"conditioned {gate.name}")
                elif name in ("MEASURE", "RESET"):
                    dynamic.append(name)
        if dynamic:
            shown = ", ".join(dict.fromkeys(dynamic))
            raise ValueError(
                f"to_unitary() cannot represent a dynamic circuit ({shown}): "
                "measurement, reset and classical feed-forward are not "
                "unitary operations.\n"
                "  Fix: simulate it with sf.run(circuit, shots=...), export it "
                "with to_qasm3()/to_qiskit(), or drop the dynamic gates first"
            )
        # MEM-1: the Rust allocator aborts the process on a failed allocation
        # (exit 0xC0000409, uncatchable), so a 4^n matrix that cannot fit is
        # refused here while it is still a Python exception.
        from superfermion._limits import guard_state_allocation

        guard_state_allocation(
            self.n_qubits, "unitary", what="Circuit.to_unitary()"
        )
        ir_dag = self.to_ir()
        return np.asarray(ir_dag.to_unitary())

    # ───────────────────────────────────────────────────────
    # Display
    # ───────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the circuit to a JSON string."""
        import json

        from superfermion._fixes import matrix_to_json
        self._ensure_gates()
        data = {
            "n_qubits": self.n_qubits,
            "n_cbits": self.n_cbits,
            "gates": [
                {
                    "name": g.name,
                    "qubits": g.qubits,
                    "params": [
                        float(p) if not isinstance(p, SymbolicParameter) else p.name
                        for p in g.params
                    ],
                    **({"matrix": matrix_to_json(g.matrix)} if g.matrix is not None else {}),
                    # Rebuilding a record from name/qubits/params cannot express
                    # either field: without them a ``measure(1, 0)`` reloads as
                    # ``measure(1, 1)`` and a guarded gate comes back unguarded.
                    **(
                        {"classical_bits": list(g.classical_bits)}
                        if g.classical_bits else {}
                    ),
                    **({"condition": list(g.condition)} if g.condition is not None else {}),
                }
                for g in self._gates
            ],
            "metadata": {
                "depth": self.depth,
                "gate_count": self.gate_count
            }
        }
        # ``_ensure_gates`` above rebuilds ``_gates`` in Rust order, so the
        # mirrored indices line up with the emitted gate list.
        for index, matrix in getattr(self, "_rust_unitary_matrices", {}).items():
            if 0 <= index < len(data["gates"]):
                data["gates"][index].setdefault(
                    "matrix", matrix_to_json(matrix)
                )
        return json.dumps(data, indent=2)


    @classmethod
    def from_json(cls, json_str: str) -> Circuit:
        """Load a circuit from a JSON string."""
        import json

        from superfermion._fixes import matrix_from_json
        data = json.loads(json_str)
        circuit = cls(data["n_qubits"], data["n_cbits"])

        for g_data in data["gates"]:
            gate_name = str(g_data["name"]).upper()
            if "matrix" in g_data:
                if gate_name != "UNITARY":
                    raise ValueError(
                        f"Gate {gate_name!r} carries a 'matrix' field; only "
                        "UNITARY records may store an opaque matrix"
                    )
                circuit.unitary(
                    matrix_from_json(g_data["matrix"]), g_data["qubits"]
                )
                circuit._restore_gate_fields(
                    circuit._gates[-1],
                    g_data.get("classical_bits"),
                    g_data.get("condition"),
                )
            elif gate_name == "UNITARY":
                raise ValueError(
                    "UNITARY gate record has no 'matrix' field; the serialized "
                    "circuit cannot be reconstructed. Re-export it with "
                    "Circuit.to_json() from a version that stores the matrix."
                )
            else:
                circuit._add_gate(
                    name=g_data["name"],
                    qubits=g_data["qubits"],
                    params=g_data["params"]
                )
                circuit._restore_gate_fields(
                    circuit._gates[-1],
                    g_data.get("classical_bits"),
                    g_data.get("condition"),
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

    def draw(self, ascii_only: bool | None = None) -> str:
        """Draw a text representation of the circuit.

        Args:
            ascii_only: Restrict the diagram to ASCII characters. ``None``
                (default) keeps the Unicode box drawing when the current
                ``sys.stdout`` can encode it and falls back to ASCII when it
                cannot, so ``print(circuit.draw())`` does not raise
                ``UnicodeEncodeError`` on a cp1252 console or with
                ``PYTHONIOENCODING=ascii``. Pass ``True``/``False`` to force it.

        Returns:
            Text circuit diagram.
        """
        if ascii_only is None:
            ascii_only = not _stdout_supports_unicode()
        wire = "---" if ascii_only else "───"
        link = "-" if ascii_only else "─"
        ctrl = "o" if ascii_only else "●"
        target = "+" if ascii_only else "⊕"
        vbar = " | " if ascii_only else " │ "

        self._ensure_gates()
        if not self._gates:
            lines = [f"q{i}: {wire} " for i in range(self._n_qubits)]
            return "\n".join(lines)

        # Allocate grid: qubit × time_step
        n_steps = len(self._gates)
        grid: list[list[str]] = [
            [wire] * n_steps for _ in range(self._n_qubits)
        ]

        def _box(text: str, gate: GateRecord) -> str:
            """Gate label; a c_if guard is part of the gate, so it is shown."""
            if gate.condition is not None:
                cbit, value = gate.condition
                return f"[{text} if c{cbit}={value}]"
            return f"[{text}]"

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
                if gname == "MEASURE" and gate.classical_bits:
                    label = f"{label}->c{gate.classical_bits[0]}"
                grid[q][step] = _box(label, gate)
            elif len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                if gate.condition is not None or gname not in ("CNOT", "CZ", "CY"):
                    grid[q0][step] = _box(gate.name, gate)
                    grid[q1][step] = _box(gate.name, gate)
                else:
                    grid[q0][step] = ctrl
                    if gname == "CNOT":
                        grid[q1][step] = target
                    else:
                        grid[q1][step] = ctrl
                # Mark vertical connection
                for q in range(min(q0, q1) + 1, max(q0, q1)):
                    if grid[q][step] == wire:
                        grid[q][step] = vbar
            elif len(gate.qubits) == 3:
                for q in gate.qubits:
                    grid[q][step] = _box(gate.name, gate)

        # Format output
        lines = []
        for q in range(self._n_qubits):
            parts = link.join(f"{cell:^7}" for cell in grid[q])
            lines.append(f"q{q}: {link}{parts}{link}")

        return "\n".join(lines)
