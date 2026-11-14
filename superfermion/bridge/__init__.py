"""
Framework Bridges — Import circuits from Qiskit, PennyLane, and Cirq.

Usage:
    from superfermion.bridge import from_qiskit, from_pennylane, from_cirq
    sf_circuit = from_qiskit(qiskit_circuit)
"""
from __future__ import annotations

import threading as _threading
from typing import Any

import superfermion as sf

# Diagnostics for :func:`from_qasm`. ``None`` after a call means the native Rust
# parser handled it; a string records why the pure-Python parser was used instead
# (``"bigint-parameter"``, or ``"<ExceptionType>: <message>"``). Exposed so tests
# can prove the native path actually ran rather than trusting that it did.
#
# The storage is thread-local: a module-level string would let two threads read
# each other's reason, so the value a caller inspects need not describe its own
# call. ``__getattr__`` below keeps the documented ``bridge._RUST_QASM_FALLBACK_REASON``
# spelling working without making the value shared.
_RUST_QASM_STATE = _threading.local()


def __getattr__(name: str):
    if name == "_RUST_QASM_FALLBACK_REASON":
        return getattr(_RUST_QASM_STATE, "reason", None)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _bit_reverse_matrix(matrix, n_qubits: int):
    """Bit-reverse a k-qubit unitary from Qiskit (little-endian) order.

    Needed ONLY for ``circuit.unitary()``, which takes big-endian matrices,
    while SF's numpy state order and named gates are little-endian
    (Qiskit-identical, verified by truth table). Qubit labels pass through
    unchanged; only the matrix is reordered.
    """
    import numpy as np

    m = np.array(matrix, dtype=complex)
    dim = 1 << n_qubits
    rev = [int(f"{i:0{n_qubits}b}"[::-1], 2) for i in range(dim)]
    return m[np.ix_(rev, rev)]


def from_qiskit(qiskit_circuit: Any) -> sf.Circuit:
    """Convert a Qiskit QuantumCircuit to a Superfermion Circuit.

    Args:
        qiskit_circuit: A qiskit.circuit.QuantumCircuit instance.

    Returns:
        Equivalent sf.Circuit.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> sf_circuit = from_qiskit(qc)
    """
    n_qubits = qiskit_circuit.num_qubits
    n_cbits = qiskit_circuit.num_clbits
    circuit = sf.Circuit(n_qubits, n_cbits=n_cbits, name=qiskit_circuit.name)

    # Gate mapping: Qiskit name -> Superfermion method
    gate_map = {
        'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z',
        's': 's', 'sdg': 'sdg', 't': 't', 'tdg': 'tdg',
        'sx': 'sx', 'id': 'id',
        'rx': 'rx', 'ry': 'ry', 'rz': 'rz',
        'p': 'p', 'u': 'u', 'u3': 'u3',
        # SF's cu/cu3 carry the Qiskit gamma phase natively (added after the
        # original 3-param definition; gamma defaults to 0).
        'cu': 'cu', 'cu3': 'cu3', 'cp': 'cp',
        'cx': 'cx', 'cnot': 'cx',
        'cz': 'cz', 'cy': 'cy',
        'swap': 'swap', 'iswap': 'iswap',
        'ccx': 'ccx', 'cswap': 'cswap',
        'rxx': 'rxx', 'ryy': 'ryy', 'rzz': 'rzz',
        'measure': 'measure',
        'barrier': 'barrier',
        'reset': 'reset',
    }

    def convert_instruction(operation, qubits, depth=0):
        """Convert one Qiskit instruction (recursive for composites)."""
        if depth > 8:
            raise ValueError(f"Definition recursion too deep for '{operation.name}'")
        gate_name = operation.name.lower()
        params = list(operation.params) if operation.params else []

        if gate_name == "unitary":
            import numpy as np
            matrix = np.array(operation.to_matrix())
            circuit.unitary(_bit_reverse_matrix(matrix, len(qubits)), qubits)
            return

        sf_name = gate_map.get(gate_name)
        if sf_name in ("cu", "cu3"):
            # Qiskit order is (theta, phi, lam, gamma) then control/target;
            # SF keeps gamma last (defaulted), so reorder explicitly.
            vals = list(params)
            theta, phi, lam = (vals + [0.0, 0.0, 0.0])[:3]
            gamma = vals[3] if len(vals) > 3 else 0.0
            getattr(circuit, sf_name)(theta, phi, lam, qubits[0], qubits[1], gamma)
            return
        if sf_name is not None:
            method = getattr(circuit, sf_name)
            if params:
                method(*params, *qubits)
            else:
                method(*qubits)
            return

        # Fallback 1: fixed matrix (ch, crx, rzz, c3sx, ...).
        try:
            import numpy as np
            matrix = np.array(operation.to_matrix())
            circuit.unitary(_bit_reverse_matrix(matrix, len(qubits)), qubits)
            return
        except Exception:
            pass

        # Fallback 2: definition expansion (IQFT, CDKM, custom composites).
        # Maps the definition's local qubits onto the physical ones.
        definition = getattr(operation, "definition", None)
        if definition is not None:
            local_map = {}
            for dq in definition.qubits:
                local_map[dq] = qubits[definition.find_bit(dq).index]
            for sub in definition.data:
                sub_q = [local_map[q] for q in sub.qubits]
                # bind no params here: parameterized composites carry
                # concrete values through their own definition closure
                convert_instruction(sub.operation, sub_q, depth + 1)
            return

        raise ValueError(
            f"Unsupported Qiskit gate: '{operation.name}'. "
            f"Supported: {list(gate_map.keys()) + ['unitary']}"
        )

    def import_if_else(operation) -> None:
        """Import a Qiskit ``if_test`` / ``if_else`` block as ``c_if`` guards.

        Every instruction in the body is guarded individually.  That is
        equivalent to Qiskit's block semantics as long as the body does not
        modify the condition bit itself (checked below); an ``else`` branch
        is imported with the inverted guard.
        """
        from qiskit.circuit import Clbit

        condition = getattr(operation, "condition", None)
        blocks = [
            block for block in getattr(operation, "blocks", ()) if len(block.data)
        ]
        if condition is None or not 1 <= len(blocks) <= 2:
            raise ValueError(
                "Only Qiskit if_test/if_else blocks with one or two non-empty "
                "branches can be converted to Circuit.c_if()"
            )
        clbit, value = condition
        if not isinstance(clbit, Clbit):
            raise ValueError(
                "Only single-Clbit if_test conditions can be converted to "
                "Circuit.c_if(); whole-register comparisons are not supported"
            )
        cond_index = qiskit_circuit.find_bit(clbit).index
        value = int(value)

        for branch_index, block in enumerate(blocks):
            # else-branch runs when the condition is false.
            branch_value = value if branch_index == 0 else 1 - value
            for body in block.data:
                body_name = body.operation.name.lower()
                if body_name == "if_else":
                    raise ValueError(
                        "Nested if_test/if_else blocks cannot be converted to "
                        "Circuit.c_if()"
                    )
                body_qubits = [
                    qiskit_circuit.find_bit(q).index for q in body.qubits
                ]
                if body_name == "measure":
                    cbits = [
                        qiskit_circuit.find_bit(c).index for c in body.clbits
                    ]
                    target = cbits[0] if cbits else body_qubits[0]
                    if target == cond_index:
                        # Later body instructions would then see a different
                        # condition value than Qiskit's block semantics.
                        raise ValueError(
                            "An if_test/if_else body that measures into its "
                            "own condition bit cannot be converted to "
                            "Circuit.c_if()"
                        )
                    circuit.c_if(cond_index, branch_value)
                    circuit.measure(body_qubits[0], target)
                    continue
                circuit.c_if(cond_index, branch_value)
                convert_instruction(body.operation, body_qubits)

    for instruction in qiskit_circuit.data:
        gate = instruction.operation
        name = gate.name.lower()
        qubits = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
        if name == "if_else":
            import_if_else(gate)
            continue
        if name == "measure":
            # Qiskit's clbit target is explicit; Circuit.measure(qubit) would
            # default it to the qubit index and silently remap c[i] -> c[q].
            cbits = [qiskit_circuit.find_bit(c).index for c in instruction.clbits]
            circuit.measure(qubits[0], cbits[0] if cbits else qubits[0])
            continue
        convert_instruction(gate, qubits)

    return circuit


def _reject_conditions(circuit: sf.Circuit, exporter: str) -> None:
    """Raise when a ``c_if`` gate cannot be represented by ``exporter``.

    Formats without classical feed-forward would otherwise emit the gate
    unconditionally and silently change the circuit's semantics.
    """
    circuit._ensure_gates()
    for gate in circuit._gates:
        condition = getattr(gate, "condition", None)
        if condition is not None:
            raise ValueError(
                f"{exporter} cannot represent classical conditions: gate "
                f"'{gate.name}' is guarded by c_if({condition[0]}, "
                f"{condition[1]}). Use to_qasm3(), which writes OpenQASM 3 "
                "'if' statements."
            )


def to_qiskit(circuit: sf.Circuit) -> Any:
    """Convert a Superfermion Circuit to a Qiskit QuantumCircuit.

    Requires qiskit to be installed.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit is required: pip install qiskit")

    qc = QuantumCircuit(circuit.n_qubits, circuit.n_cbits)

    # Materialize Rust-stored gates so ``_gates`` is populated.
    circuit._ensure_gates()

    gate_map = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'sdg', 'T': 't', 'TDG': 'tdg',
        'SX': 'sx', 'ID': 'id',
        'CRX': 'crx', 'CRZ': 'crz', 'R1': 'p', 'SXDG': 'sxdg',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'P': 'p', 'U': 'u', 'U3': 'u3',
        'CU': 'cu', 'CU3': 'cu3', 'CP': 'cp',
        'CX': 'cx', 'CNOT': 'cx',
        'CZ': 'cz', 'CY': 'cy',
        'SWAP': 'swap', 'ISWAP': 'iswap',
        'ECR': 'ecr',
        'CCX': 'ccx', 'CSWAP': 'cswap',
        'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
        'MEASURE': 'measure',
        'BARRIER': 'barrier',
        'RESET': 'reset',
    }

    from contextlib import nullcontext

    def _condition_scope(gate: Any) -> Any:
        """Wrap a gate in an ``if_test`` block when ``c_if`` is set."""
        condition = getattr(gate, "condition", None)
        if condition is None:
            return nullcontext()
        cbit, value = condition
        # Qiskit dropped ``QuantumCircuit.c_if``; dynamic circuits express
        # the same guard as ``with qc.if_test((clbit, value)):``.
        return qc.if_test((qc.clbits[cbit], value))

    def _emit(gate: Any, qiskit_name: str, mapped_qubits: list[int]) -> None:
        method = getattr(qc, qiskit_name)
        if gate.name == "MEASURE":
            cbit = gate.classical_bits[0] if gate.classical_bits else gate.qubits[0]
            method(mapped_qubits[0], cbit)
        elif gate.name in ("CU", "CU3"):
            # SF stores [theta, phi, lam, gamma]; Qiskit cu takes the same
            # four followed by control/target. Pad gamma for legacy records.
            p = (list(gate.params) + [0.0, 0.0, 0.0, 0.0])[:4]
            method(p[0], p[1], p[2], p[3], *mapped_qubits)
        elif gate.name == "U3":
            method(*gate.params, *mapped_qubits)
        elif gate.params:
            method(*gate.params, *mapped_qubits)
        else:
            method(*mapped_qubits)

    for gate in circuit._gates:
        if gate.name.upper() == "UNITARY" and gate.matrix is not None:
            from qiskit.circuit.library import UnitaryGate
            # ``circuit.unitary()`` stores big-endian matrices; Qiskit /
            # SF state order is little-endian, so bit-reverse on export.
            matrix = _bit_reverse_matrix(gate.matrix, len(gate.qubits))
            with _condition_scope(gate):
                qc.append(UnitaryGate(matrix), list(gate.qubits))
            continue

        qiskit_name = gate_map.get(gate.name.upper())
        if qiskit_name is None:
            raise ValueError(f"Cannot map gate '{gate.name}' to Qiskit")

        # Qubit labels pass through unchanged: SF and Qiskit share the
        # little-endian (q0=LSB) state order.
        mapped_qubits = list(gate.qubits)
        with _condition_scope(gate):
            _emit(gate, qiskit_name, mapped_qubits)

    return qc


def from_pennylane(pennylane_circuit: Any, n_qubits: int = None) -> sf.Circuit:
    """Convert a PennyLane tape/circuit to a Superfermion Circuit.

    Args:
        pennylane_circuit: A PennyLane QNode or tape.
        n_qubits: Number of qubits (auto-detected if possible).

    Returns:
        Equivalent sf.Circuit.
    """
    try:
        import pennylane as qml  # noqa: F401 - probe that PennyLane is importable
    except ImportError:
        raise ImportError("PennyLane is required: pip install pennylane")

    # Get the tape from the QNode.  A QNode only has a tape after it has been
    # constructed, so build it first: passing the QNode itself used to fail
    # with "'QNode' object has no attribute 'operations'".
    if hasattr(pennylane_circuit, 'construct'):
        try:
            pennylane_circuit.construct([], {})
        except TypeError:
            pass
        tape = (
            getattr(pennylane_circuit, 'tape', None)
            or getattr(pennylane_circuit, '_tape', None)
            or pennylane_circuit
        )
    elif hasattr(pennylane_circuit, 'tape'):
        tape = pennylane_circuit.tape
    else:
        tape = pennylane_circuit

    if n_qubits is None:
        n_qubits = len(tape.wires)

    circuit = sf.Circuit(n_qubits)

    # Gates absent from this map previously raised; the added entries all
    # flow through the identical generic method(*params, *wires) path the
    # existing entries already use, so they are pure additive coverage.
    pl_map = {
        'Hadamard': 'h', 'PauliX': 'x', 'PauliY': 'y', 'PauliZ': 'z',
        'S': 's', 'T': 't', 'SX': 'sx',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'PhaseShift': 'p',
        'CNOT': 'cx', 'CY': 'cy', 'CZ': 'cz',
        'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz',
        'ControlledPhaseShift': 'cp',
        'SWAP': 'swap', 'ISWAP': 'iswap', 'CSWAP': 'cswap',
        'Toffoli': 'ccx',
    }

    for op in tape.operations:
        sf_name = pl_map.get(op.name)
        if sf_name is None:
            raise ValueError(
                f"Unsupported PennyLane operation: '{op.name}'. "
                f"Supported: {list(pl_map.keys())}"
            )

        method = getattr(circuit, sf_name)
        wires = [int(w) for w in op.wires]
        params = list(op.parameters) if op.parameters else []

        if params:
            method(*params, *wires)
        else:
            method(*wires)

    return circuit


def _looks_like_qasm3(text: str) -> bool:
    """True when *text* uses OpenQASM 3 declarations the QASM 2 path cannot read.

    ``OPENQASM 3.0;`` headers and the ``qubit[n]``/``bit[n]`` register form are
    unambiguous: no OpenQASM 2 program contains them. The native parser accepts
    ``qubit[n]`` but not ``bit[n]``, and the pure-Python fallback is QASM 2 only,
    so without this check ``from_qasm(circuit.to_qasm3())`` raised on the
    ``bit[2] c;`` line even though this function documents QASM 3 support.
    """
    import re

    return bool(
        re.search(r"^\s*OPENQASM\s+3", text, re.MULTILINE)
        or re.search(r"^\s*(?:bit|qubit)\s*\[", text, re.MULTILINE)
    )


def _split_qasm2_statements(qasm_str: str) -> list[str]:
    """Return QASM 2 statements, however many share a physical line.

    QASM 2 does not require one statement per line, and minified programs put a
    whole circuit on one line.  Feeding such a line to the per-statement parser
    read only its first gate and dropped the rest (or, when that gate carried an
    angle, folded every ``q[i]`` on the line into its operand list).
    """
    statements: list[str] = []
    buf: list[str] = []
    depth = 0
    in_string = False
    for raw in qasm_str.split('\n'):
        # A ``//`` comment runs to end of line, unless it sits in a string.
        cut = raw.find('//')
        if cut != -1 and raw[:cut].count('"') % 2 == 0:
            raw = raw[:cut]
        for ch in raw:
            if ch == '"':
                in_string = not in_string
                buf.append(ch)
            elif ch == '(' and not in_string:
                depth += 1
                buf.append(ch)
            elif ch == ')' and not in_string:
                depth -= 1
                buf.append(ch)
            elif ch == ';' and not in_string and depth <= 0:
                buf.append(ch)
                statement = ''.join(buf).strip()
                if statement:
                    statements.append(statement)
                buf = []
            else:
                buf.append(ch)
        # A statement may continue on the next line, so only flush here when the
        # buffer already looks complete: a qubit operand ended it, or a ';' did.
        text = ''.join(buf)
        if not in_string and depth <= 0 and (
            text.rstrip().endswith(';') or ']' in text
        ):
            if text.strip():
                statements.append(text.strip())
            buf = []
        else:
            # Keep a separator so tokens from different lines never glue.
            buf.append(' ')
    tail = ''.join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _qasm2_fallback_error(message: str) -> ValueError:
    """Build a fallback error that reports the native rejection first."""
    reason = getattr(_RUST_QASM_STATE, "reason", None)
    if reason:
        return ValueError(f"{reason} (Python fallback: {message})")
    return ValueError(message)


def _parse_qasm2_python(qasm_str: str) -> sf.Circuit:
    """Pure-Python OpenQASM 2 importer, used when the native parser rejects.

    Mirrors the native parser's register model: declarations form a table
    ``name -> (offset, size)``, so ``qreg a[1]; qreg b[1]; x b[0];`` is a
    2-qubit circuit with X on qubit 1.  A whole register broadcasts over its
    qubits (``h q;`` is one H per qubit; ``cx a, b;`` pairs the registers
    element-wise), and ``if (c == N)`` lowers to ``Circuit.c_if``.
    """
    import re

    from superfermion.circuit import GateRecord as _GateRecord

    qasm_map = {
        'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z',
        's': 's', 'sdg': 'sdg', 't': 't', 'tdg': 'tdg',
        'sx': 'sx',
        'rx': 'rx', 'ry': 'ry', 'rz': 'rz',
        'u': 'u', 'u3': 'u', 'u2': 'u',
        'p': 'p', 'u1': 'p', 'r1': 'p',
        'cx': 'cx', 'cnot': 'cx',
        'cz': 'cz',
        'swap': 'swap',
        'ccx': 'ccx',
        'id': 'id', 'u0': 'id',
        'ch': 'ch', 'crx': 'crx', 'cry': 'cry', 'crz': 'crz',
        'cu': 'cu', 'cu3': 'cu3', 'cu1': 'cp', 'cp': 'cp',
        'cy': 'cy', 'iswap': 'iswap', 'cswap': 'cswap',
        'rxx': 'rxx', 'ryy': 'ryy', 'rzz': 'rzz', 'ecr': 'ecr',
        'sxdg': 'sxdg',
    }
    raw_gate_names = frozenset({'sxdg'})

    qregs: dict[str, tuple[int, int]] = {}
    cregs: dict[str, tuple[int, int]] = {}
    n_qubits = 0
    n_cbits = 0
    circuit = None

    def resolve(text: str, table: dict, width: int) -> list:
        """Resolve comma-separated operands to ('idx', i) / ('reg', off, size)."""
        ops = []
        for token in text.split(','):
            token = token.strip()
            if not token:
                continue
            m = re.fullmatch(r'([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?', token)
            if not m:
                raise _qasm2_fallback_error(f"cannot parse qubit operand {token!r}")
            name, idx = m.group(1), m.group(2)
            span = table.get(name)
            if span is None:
                raise _qasm2_fallback_error(f"unknown register {name!r}")
            offset, size = span
            if idx is None:
                ops.append(('reg', offset, size))
            else:
                i = int(idx)
                if i >= size:
                    raise _qasm2_fallback_error(
                        f"qubit index {i} out of range for register {name!r} "
                        f"(size {size})"
                    )
                absolute = offset + i
                if absolute >= width:
                    raise _qasm2_fallback_error(
                        f"qubit index {absolute} out of range for {width} qubits"
                    )
                ops.append(('idx', absolute))
        if not ops:
            raise _qasm2_fallback_error("statement names no qubit operand")
        return ops

    def expand(ops: list, gate_name: str) -> list:
        """One index list per gate application (whole registers broadcast)."""
        if all(op[0] == 'idx' for op in ops):
            return [[op[1] for op in ops]]
        if any(op[0] == 'idx' for op in ops):
            raise _qasm2_fallback_error(
                f"gate {gate_name!r} mixes whole-register and indexed operands"
            )
        sizes = {op[2] for op in ops}
        if len(sizes) != 1:
            raise _qasm2_fallback_error(
                f"gate {gate_name!r} operands have different register sizes"
            )
        size = sizes.pop()
        return [[op[1] + k for op in ops] for k in range(size)]

    def apply_gate(gate_name: str, params: list, ops: list, condition=None) -> None:
        instances = expand(ops, gate_name)
        sf_name = qasm_map.get(gate_name)
        if sf_name is None:
            raise _qasm2_fallback_error(f"cannot map gate {gate_name!r}")
        method = getattr(circuit, sf_name, None)
        if method is None and sf_name not in raw_gate_names:
            raise _qasm2_fallback_error(f"cannot map gate {gate_name!r}")
        for inst in instances:
            if len(set(inst)) != len(inst):
                raise _qasm2_fallback_error(
                    f"gate {gate_name!r} names a qubit more than once: {inst}"
                )
            if method is not None:
                if condition is not None:
                    circuit.c_if(*condition)
                try:
                    method(*params, *inst)
                except (TypeError, ValueError) as exc:
                    raise _qasm2_fallback_error(
                        f"gate {gate_name!r} cannot take qubit operands {inst}: {exc}"
                    ) from exc
            else:
                circuit._gates.append(_GateRecord(
                    name=sf_name.upper(),
                    qubits=list(inst),
                    params=list(params),
                    condition=condition,
                ))

    for line in _split_qasm2_statements(qasm_str):
        if (
            line.startswith('//')
            or line.startswith('OPENQASM')
            or line.startswith('include')
        ):
            continue

        m = re.fullmatch(r'qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;?', line)
        if m:
            if circuit is not None:
                raise _qasm2_fallback_error(
                    f"qreg declaration after the first instruction: {line!r}"
                )
            name, size = m.group(1), int(m.group(2))
            qregs[name] = (n_qubits, size)
            n_qubits += size
            continue

        m = re.fullmatch(r'qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)\s*;?', line)
        if m:
            if circuit is not None:
                raise _qasm2_fallback_error(
                    f"qubit declaration after the first instruction: {line!r}"
                )
            size, name = int(m.group(1)), m.group(2)
            qregs[name] = (n_qubits, size)
            n_qubits += size
            continue

        m = re.fullmatch(r'creg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;?', line)
        if m:
            if circuit is not None:
                raise _qasm2_fallback_error(
                    f"creg declaration after the first instruction: {line!r}"
                )
            name, size = m.group(1), int(m.group(2))
            cregs[name] = (n_cbits, size)
            n_cbits += size
            continue

        # A bare barrier has no semantics (to_qasm drops BARRIER too).
        if line.rstrip(';').strip() == 'barrier':
            continue

        if circuit is None:
            if n_qubits == 0:
                raise _qasm2_fallback_error(
                    f"instruction before any qreg declaration: {line!r}"
                )
            circuit = sf.Circuit(n_qubits, n_cbits=n_cbits or n_qubits)

        # Classical feed-forward.  OpenQASM 2 writes ``if (c == value)`` where
        # ``c`` names a whole creg whose integer value is compared; the indexed
        # form ``if (c[i] == value)`` is accepted too.  ``Circuit.c_if`` tests a
        # single classical bit, so only one-bit conditions can be lowered.
        cond = re.match(
            r'if\s*\(\s*([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?\s*==\s*(\d+)\s*\)\s*(.+)',
            line,
        )
        condition = None
        if cond:
            cname, cidx, cval = cond.group(1), cond.group(2), int(cond.group(3))
            span = cregs.get(cname)
            if span is None:
                raise _qasm2_fallback_error(
                    f"if statement names unknown classical register {cname!r}"
                )
            offset, size = span
            if cidx is None:
                if size != 1:
                    raise _qasm2_fallback_error(
                        f"if statement compares the whole {size}-bit register "
                        f"{cname!r}, but Circuit.c_if only tests one bit"
                    )
                cidx = 0
            else:
                cidx = int(cidx)
                if cidx >= size:
                    raise _qasm2_fallback_error(
                        f"if statement names classical bit {cidx} outside "
                        f"register {cname!r} (size {size})"
                    )
            if cval not in (0, 1):
                raise _qasm2_fallback_error(
                    f"if statement compares {cname}[{cidx}] with {cval}, "
                    f"but Circuit.c_if only tests 0 or 1"
                )
            condition = (offset + cidx, cval)
            line = cond.group(4).strip()

        if line.startswith('measure'):
            mm = re.match(r'measure\s+(.+?)\s*(?:->\s*(.+?))?\s*;?$', line)
            if not mm:
                raise _qasm2_fallback_error(f"cannot parse measure: {line!r}")
            q_rows = expand(resolve(mm.group(1), qregs, n_qubits), 'measure')
            c_text = (mm.group(2) or '').strip()
            c_rows = None
            if c_text:
                width = n_cbits or n_qubits
                c_rows = expand(resolve(c_text, cregs, width), 'measure')
            if c_rows is None:
                pairs = [(q, q) for (q,) in q_rows]
            elif len(c_rows) == len(q_rows):
                pairs = [(q, c) for (q,), (c,) in zip(q_rows, c_rows)]
            else:
                raise _qasm2_fallback_error(
                    f"measure qubit and classical operands have different sizes: {line!r}"
                )
            for q, c in pairs:
                # `if(c==1) measure q[0] -> c[0];` guards the measurement
                # itself (QASM3 / Qiskit if_test agree on these semantics).
                if condition is not None:
                    circuit.c_if(*condition)
                circuit.measure(q, c)
            continue

        if line.startswith('barrier'):
            if condition is not None:
                # The native parser and Qiskit's QASM2 importer both reject a
                # guarded barrier: OpenQASM 2 allows ``if`` only on gates,
                # measurements and resets.
                raise _qasm2_fallback_error("a barrier cannot be conditioned")
            rest = line[len('barrier'):].strip().rstrip(';').strip()
            if rest:
                indices = []
                for op in resolve(rest, qregs, n_qubits):
                    if op[0] == 'idx':
                        indices.append(op[1])
                    else:
                        indices.extend(range(op[1], op[1] + op[2]))
                circuit.barrier(*indices)
            continue

        if line.startswith('reset'):
            rest = line[len('reset'):].strip().rstrip(';').strip()
            for inst in expand(resolve(rest, qregs, n_qubits), 'reset'):
                # ``reset`` is a quantum operation, so ``if`` applies to it
                # (Qiskit imports it as an IfElseOp).  This used to bypass the
                # condition and silently make the reset unconditional.
                if condition is not None:
                    circuit.c_if(*condition)
                circuit.reset(inst[0])
            continue

        m = re.match(r'([A-Za-z_]\w*)(?:\(([^)]*)\))?\s+(.+?)\s*;?$', line)
        if not m:
            raise _qasm2_fallback_error(f"cannot parse statement: {line!r}")
        gate_name = m.group(1).lower()
        params_str = m.group(2)
        qubits_str = m.group(3)

        params = []
        if params_str:
            for p in params_str.split(','):
                p = p.strip()
                try:
                    as_int = int(p)
                    if abs(as_int) > 2 ** 53:
                        params.append(as_int)
                    else:
                        params.append(float(p))
                except ValueError:
                    try:
                        params.append(float(p))
                    except ValueError:
                        if 'pi' in p:
                            import math

                            try:
                                params.append(
                                    eval(p.replace('pi', str(math.pi)),
                                         {'__builtins__': {}}, {})
                                )
                            except Exception as exc:
                                raise _qasm2_fallback_error(
                                    f"cannot evaluate parameter {p!r} in {line!r}"
                                ) from exc
                        else:
                            params.append(sf.param(p))

        if gate_name == 'u2':
            import math

            params = [math.pi / 2] + params
        if gate_name == 'u0':
            # u0(gamma) is the idle gate; the native parser and the QASM
            # definition both reduce it to U(0, 0, 0).
            gate_name = 'u'
            params = [0.0, 0.0, 0.0]

        apply_gate(
            gate_name, params, resolve(qubits_str, qregs, n_qubits), condition
        )

    if circuit is None:
        if n_qubits == 0:
            raise _qasm2_fallback_error("no qubit declaration found")
        # A program may contain declarations only; return the empty circuit.
        circuit = sf.Circuit(n_qubits, n_cbits=n_cbits or n_qubits)
    return circuit


def from_qasm(qasm_str: str) -> sf.Circuit:
    """Parse an OpenQASM 2.0/3.0 string into a Superfermion Circuit.

    Uses the native Rust parser (6× faster than Qiskit) when available,
    falling back to the pure-Python parser for unsupported syntax. OpenQASM 3
    input (``OPENQASM 3.0;``, ``qubit[n] q;``, ``bit[n] c;``) that the native
    parser rejects is handed to :func:`superfermion.serialization.from_qasm3`,
    so ``from_qasm(circuit.to_qasm3())`` round-trips.

    Both parsers are strict: a gate or statement that cannot be represented
    raises ``ValueError`` instead of being dropped, so a returned circuit is
    never silently missing operations.
    """
    # Reset before anything else can raise. The pre-scan below throws on a
    # non-str argument, and without this a later read of the attribute would
    # still report the previous call's reason.
    _RUST_QASM_STATE.reason = None

    # ── Pre-scan: skip Rust path if QASM contains big-integer parameters (> 2^53)
    #   that would lose precision when parsed as IEEE-754 f64.
    import re as _re
    _has_bigint = any(
        int(m) >= 2**53
        for m in _re.findall(r'(?<![\w.])(\d+)(?![.\d])', qasm_str)
        if len(m) >= 16
    ) if _re.search(r'\b\d{16,}\b', qasm_str) else False

    # ── Fast path: Rust-native QASM2 parser ─────────────────
    # QuantumDAG.from_qasm2 uses little-endian pass-through (an OpenQASM qubit
    # q[i] maps to SF qubit i), matching from_qiskit/to_qiskit and to_qasm, so
    # SF -> to_qasm -> from_qasm is an involution for asymmetric circuits.
    # (The parser previously reversed q[i] -> n-1-i (big-endian); that defect
    # was fixed in sf-ir/src/qasm.rs and is locked by test_bridges.
    # TestQASMEndianness.test_qasm_rust_path_roundtrip_preserves_statevector.)
    #
    # Only the failures that mean "this build or this string cannot use the
    # native parser" are caught:
    #   ImportError -- the extension is not built (pure-Python install)
    #   ValueError  -- from_qasm2 rejected the string. sf-ir/src/qasm.rs now
    #                  reports unsupported gates, out-of-range qubits, missing
    #                  declarations and unparseable statements as errors rather
    #                  than skipping them, so this is a real rejection
    #   TypeError   -- native API shape did not match (defensive)
    # Anything else propagates. This used to be ``except BaseException``, which
    # also swallowed ``pyo3_runtime.PanicException``: a panicking native parser
    # then produced a correct circuit via the Python fallback, so the involution
    # test passed while the Rust parser stayed broken.
    if _has_bigint:
        # Documented skip: integer literals above 2^53 cannot round-trip through
        # the f64 parameters the native parser produces.
        _RUST_QASM_STATE.reason = "bigint-parameter"
    else:
        # Only the import and the parse call are guarded. Everything after a
        # successful parse -- to_gate_records, Circuit(), extend_raw_from_records
        # -- is Rust-backed conversion, and a failure there is a defect, not an
        # "unsupported syntax" case, so it must propagate.
        dag = None
        try:
            from superfermion._sf_core import QuantumDAG
            dag = QuantumDAG.from_qasm2(qasm_str)
        except (ImportError, ValueError, TypeError) as exc:
            _RUST_QASM_STATE.reason = f"{type(exc).__name__}: {exc}"
        except BaseException as exc:
            # Recorded, then re-raised. Letting it propagate is the whole point
            # (a panicking parser is a defect); leaving the reason at ``None``
            # would report the opposite of what happened.
            _RUST_QASM_STATE.reason = f"{type(exc).__name__}: {exc} (propagated)"
            raise
        if dag is not None:
            try:
                # Conditioned records keep the ``if (creg == value)`` guard
                # the native parser now attaches; the plain export drops it.
                records = dag.to_gate_records_with_conditions()
                circ = sf.Circuit(dag.n_qubits(), n_cbits=dag.n_cbits())
                # Single-pass bulk append: avoids intermediate list + double .upper()
                circ.extend_raw_from_records(records)
            except BaseException as exc:
                _RUST_QASM_STATE.reason = (
                    f"conversion-failed: {type(exc).__name__}: {exc}"
                )
                raise
            return circ

    # ── QASM 3 path ───────────────────────────────────────────
    # Only reached when the native parser could not read the string. If the
    # string is unmistakably QASM 3, the QASM 3 importer is the right fallback:
    # the pure-Python parser below understands QASM 2 syntax only and would
    # reject a ``bit[n] c;`` declaration even though it carries no operation.
    if _looks_like_qasm3(qasm_str):
        from superfermion.serialization.qasm_roundtrip import from_qasm3

        return from_qasm3(qasm_str)

    return _parse_qasm2_python(qasm_str)


def to_ionq(circuit: sf.Circuit) -> list[dict[str, Any]]:
    """Convert a Superfermion Circuit to IonQ JSON gate format.

    Args:
        circuit: Superfermion Circuit.

    Returns:
        List of gate dictionaries as required by IonQ API.
    """
    ionq_gates = []
    _reject_conditions(circuit, "to_ionq")

    # Mapper for Superfermion gate names to IonQ gate names
    # Note: IonQ gate names are lowercase (e.g., 'h', 'x', 'cnot')
    gate_map = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'si', 'T': 't', 'TDG': 'ti',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CX': 'cnot', 'CNOT': 'cnot',
        'CZ': 'cz',
        'SWAP': 'swap',
        # Additive 1:1 mappings for names that used to raise.
        # 'SX' is 'v' in IonQ's QIS gateset (the sqrt-of-not gate); 'sx' is
        # not an IonQ API name.
        'SX': 'v', 'CY': 'cy', 'ISWAP': 'iswap',
    }

    circuit._ensure_gates()  # no-op for default Python storage

    def _emit_cnot(control: int, target: int) -> None:
        """Emit CNOT(control, target), reusing the reversed-CNOT workaround."""
        if control > target:
            # H(c) H(t) . CNOT(t,c) . H(c) H(t) == CNOT(c,t)
            ionq_gates.append({"gate": "h", "target": control})
            ionq_gates.append({"gate": "h", "target": target})
            ionq_gates.append({"gate": "cnot", "control": target, "target": control})
            ionq_gates.append({"gate": "h", "target": control})
            ionq_gates.append({"gate": "h", "target": target})
        else:
            ionq_gates.append({"gate": "cnot", "control": control, "target": target})

    for gate in circuit._gates:
        name = gate.name.upper()
        ionq_name = gate_map.get(name)

        # Gates IonQ has no single instruction for, emitted as short exact
        # decompositions (global phase is unobservable on measurement).
        if name == "U" and len(gate.params) == 3:
            theta, phi, lam = (float(p) for p in gate.params)
            target = gate.qubits[0]
            # U3 = e^{-i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam)
            ionq_gates.append({"gate": "rz", "target": target, "rotation": lam})
            ionq_gates.append({"gate": "ry", "target": target, "rotation": theta})
            ionq_gates.append({"gate": "rz", "target": target, "rotation": phi})
            continue

        if name == "CP" and len(gate.params) == 1:
            angle = float(gate.params[0])
            control, target = gate.qubits[0], gate.qubits[1]
            # CP(phi) = e^{-i phi/4} RZ(phi/2) RZ(phi/2) RZZ(-phi/2)
            ionq_gates.append({"gate": "rz", "target": control, "rotation": angle / 2})
            ionq_gates.append({"gate": "rz", "target": target, "rotation": angle / 2})
            ionq_gates.append({
                "gate": "zz",
                "targets": [control, target],
                "rotation": -angle / 2,
            })
            continue

        if name in ("RXX", "RYY", "RZZ") and len(gate.params) == 1:
            ionq_gates.append({
                "gate": {"RXX": "xx", "RYY": "yy", "RZZ": "zz"}[name],
                "targets": [gate.qubits[0], gate.qubits[1]],
                "rotation": float(gate.params[0]),
            })
            continue

        if name == "P" and len(gate.params) == 1:
            # IonQ has no phase gate; RZ differs by a global phase only.
            ionq_gates.append({
                "gate": "rz",
                "target": gate.qubits[0],
                "rotation": float(gate.params[0]),
            })
            continue

        # IonQ's documented QIS gateset (v0.3 "Supported Gates") has no
        # 'ccnot' / 'cswap' / 'iswap' instruction, so emit the documented
        # multi-controlled form or a short exact decomposition instead.
        if name == "CCX":
            c1, c2, target = (int(q) for q in gate.qubits)
            ionq_gates.append({
                "gate": "not",
                "target": target,
                "controls": [c1, c2],
            })
            continue

        if name == "CSWAP":
            a, b, c = (int(q) for q in gate.qubits)
            # CSWAP(a;b,c) = CX(b->c) . CCX(a,c,b) . CX(b->c), exact.
            _emit_cnot(b, c)
            ionq_gates.append({"gate": "not", "target": b, "controls": [a, c]})
            _emit_cnot(b, c)
            continue

        if name == "ISWAP":
            q0, q1 = (int(q) for q in gate.qubits)
            # iSWAP = CZ . (S (x) S) . SWAP, exact.
            ionq_gates.append({"gate": "swap", "targets": [q0, q1]})
            ionq_gates.append({"gate": "s", "target": q0})
            ionq_gates.append({"gate": "s", "target": q1})
            ionq_gates.append({"gate": "cz", "control": q0, "target": q1})
            continue

        # Superseded by the documented-form branches above; kept in place
        # per the additive-only policy (unreachable for CCX/CSWAP now).
        if name in ("CCX", "CSWAP"):
            ionq_gates.append({
                "gate": "ccnot" if name == "CCX" else "cswap",
                "targets": list(gate.qubits),
            })
            continue

        if ionq_name is None:
            # Skip gates IonQ might not support natively in this bridge
            if gate.name.upper() in ["MEASURE", "BARRIER"]:
                continue
            raise ValueError(f"Cannot map gate '{gate.name}' to IonQ API")

        gate_dict = {
            "gate": ionq_name
        }

        # IonQ API v0.3 formatting
        if len(gate.qubits) == 1:
            gate_dict["target"] = gate.qubits[0]
        elif len(gate.qubits) == 2:
            if ionq_name in ['cnot', 'cz', 'cy']:
                # Control is index 0, target is index 1
                control = gate.qubits[0]
                target = gate.qubits[1]

                # IonQ simulator has a bug with cnot where control=1, target=0
                # in 6-qubit circuits (states collapse to 16/32, q0=q1 correlation).
                # The root cause is in the IonQ simulator, not our bridge.
                # Workaround: decompose CNOT(c,t) where c > t as:
                #   H(c) * H(t) * CNOT(t,c) * H(c) * H(t)
                # This uses only forward CNOT (control < target), avoiding the bug.
                if ionq_name == 'cnot' and control > target:
                    # H on control
                    ionq_gates.append({"gate": "h", "target": control})
                    # H on target
                    ionq_gates.append({"gate": "h", "target": target})
                    # CNOT with swapped (now control < target)
                    ionq_gates.append({"gate": "cnot", "control": target, "target": control})
                    # H on control
                    ionq_gates.append({"gate": "h", "target": control})
                    # H on target
                    ionq_gates.append({"gate": "h", "target": target})
                    continue  # skip normal gate_dict append below

                gate_dict["control"] = control
                gate_dict["target"] = target
            elif ionq_name == 'swap':
                gate_dict["targets"] = [gate.qubits[0], gate.qubits[1]]
            else:
                gate_dict["targets"] = [gate.qubits[0], gate.qubits[1]]

        # Add parameters if they exist
        if gate.params:
            # IonQ expects 'rotation' in radians for rx, ry, rz
            if ionq_name in ['rx', 'ry', 'rz']:
                gate_dict["rotation"] = gate.params[0]
            else:
                gate_dict["params"] = gate.params

        ionq_gates.append(gate_dict)

    return ionq_gates


def to_braket(circuit: sf.Circuit) -> Any:
    """Convert a Superfermion Circuit to an Amazon Braket Circuit.

    Requires: pip install amazon-braket-sdk
    """
    from superfermion.devices.braket import _to_braket

    _reject_conditions(circuit, "to_braket")
    return _to_braket(circuit)


def to_qasm(circuit: sf.Circuit) -> str:
    """Convert a Superfermion Circuit to OpenQASM 2.0 string.

    Returns:
        String in OpenQASM 2.0 format.
    """
    qasm = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    qasm.append(f"qreg q[{circuit.n_qubits}];")
    if circuit.n_cbits > 0:
        qasm.append(f"creg c[{circuit.n_cbits}];")

    gate_map = {
        'H': 'h', 'X': 'u3', 'Y': 'u3', 'Z': 'u1',
        'S': 'u1', 'SDG': 'u1', 'T': 'u1', 'TDG': 'u1',
        'RX': 'u3', 'RY': 'u3', 'RZ': 'u1',
        'CX': 'cx', 'CNOT': 'cx', 'CZ': 'cz',
        'SWAP': 'swap', 'MEASURE': 'measure',
        # Additive coverage: names sf.Circuit stores that used to raise
        # "Cannot map gate".  P/CP/U map onto qelib1 gates directly, RESET
        # is valid OpenQASM 2.0, and the four entries at the bottom are
        # expanded into gate sequences below before this name is emitted.
        'CY': 'cy', 'CCX': 'ccx', 'CSWAP': 'cswap',
        'P': 'u1', 'CP': 'cu1', 'U': 'u3', 'SX': 'u3', 'RESET': 'reset',
        'ISWAP': 'iswap', 'RXX': 'rxx', 'RYY': 'ryy', 'RZZ': 'rzz',
        # Additive coverage for the names ``sf.Circuit`` stores that qelib1
        # already defines: ID, R1, the controlled rotations and CU.  ECR has
        # no qelib1 name and is expanded below instead.
        'ID': 'id', 'R1': 'u1', 'SXDG': 'sxdg',
        'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz', 'CH': 'ch',
        'CU': 'cu', 'CU3': 'cu3', 'ECR': 'ecr',
        # `barrier` is a native OpenQASM 2.0 statement; it used to be dropped
        # with the (incorrect) claim that the format has no equivalent.
        'BARRIER': 'barrier',
    }

    circuit._ensure_gates()  # no-op for default Python storage

    for gate in circuit._gates:
        qasm_name = gate_map.get(gate.name.upper())
        if not qasm_name:
            # An unmapped gate must not be silently dropped -- that would
            # emit a circuit that is missing operations. Raise instead,
            # matching to_qiskit / to_cirq / to_ionq.
            raise ValueError(
                f"Cannot map gate '{gate.name}' to OpenQASM 2.0 "
                f"(supported names: {sorted(set(gate_map.values()))})"
            )

        # OpenQASM 2 applies `if(c==value)` to the *whole* classical
        # register, so a single-bit c_if is only expressible when the
        # circuit declares exactly one classical bit (which is how Qiskit
        # exports `if(c==1) measure q[0] -> c[0];`).  Wider registers have
        # no OpenQASM 2 spelling; to_qasm3() writes the per-bit form.
        guard = ""
        if gate.condition is not None:
            cbit, value = gate.condition
            if circuit.n_cbits != 1:
                raise ValueError(
                    "Cannot map a conditional gate to OpenQASM 2.0: "
                    "`if(c==value)` compares the whole classical register, "
                    f"but Circuit.c_if tests c[{cbit}] and this circuit has "
                    f"{circuit.n_cbits} classical bits. Use to_qasm3(), "
                    f"which writes 'if (c[{cbit}] == {value})'."
                )
            guard = f"if(c=={value}) "

        def emit(statements: list[str], _guard: str = guard) -> None:
            qasm.extend(_guard + stmt for stmt in statements)

        if gate.name == "MEASURE":
            cbit = gate.classical_bits[0] if gate.classical_bits else gate.qubits[0]
            emit([f"measure q[{gate.qubits[0]}] -> c[{cbit}];"])
            continue

        if gate.name.upper() == "BARRIER":
            if gate.condition is not None:
                # OpenQASM 2 allows `if` only on gates, measurements and
                # resets; Qiskit's importer rejects `if(c==1) barrier ...`.
                # Both SF parsers reject a guarded barrier for the same
                # reason.  QASM3 has no such restriction.
                raise ValueError(
                    "Cannot map a conditional barrier to OpenQASM 2.0: "
                    "`if` may only guard gates, measurements and resets. "
                    "Use to_qasm3(), which writes 'if (...) barrier ...;'."
                )
            qubits_str = ", ".join(f"q[{i}]" for i in gate.qubits)
            emit([f"barrier {qubits_str};"])
            continue

        # Handle Parameters for U Gates
        name_upper = gate.name.upper()

        # Two-qubit gates with no qelib1 name: expand to exact u3/cx/s/u1
        # sequences.  Every sequence below was round-tripped through Qiskit's
        # OpenQASM 2.0 parser and compared against the gate unitary (max
        # deviation 6e-16, global phase ignored).
        # IBM's echoed cross-resonance has no qelib1 name, so it is emitted as
        # the exact u3/cx sequence it is (verified against Qiskit's ECRGate to
        # 2.4e-16, global phase ignored):
        #   U3(0,0,pi/2)(0) . U3(pi/2,-pi/2,pi/2)(1) . CX(0,1) . U3(pi,0,pi)(0)
        if name_upper == "ECR":
            q0, q1 = gate.qubits[0], gate.qubits[1]
            emit([
                f"u3(0, 0, pi/2) q[{q0}];",
                f"u3(pi/2, -pi/2, pi/2) q[{q1}];",
                f"cx q[{q0}], q[{q1}];",
                f"u3(pi, 0, pi) q[{q0}];",
            ])
            continue

        if name_upper in ("RXX", "RYY", "RZZ", "ISWAP"):
            q0, q1 = gate.qubits[0], gate.qubits[1]
            if name_upper == "ISWAP":
                emit([
                    f"u3(0, 0, 1.57079632679) q[{q0}];",
                    f"u3(1.57079632679, 0, pi) q[{q0}];",
                    f"u3(0, 0, 1.57079632679) q[{q1}];",
                    f"cx q[{q0}], q[{q1}];",
                    f"cx q[{q1}], q[{q0}];",
                    f"u3(1.57079632679, 0, pi) q[{q1}];",
                ])
                continue
            theta = gate.params[0]
            if name_upper == "RXX":
                pre = [f"h q[{q0}];", f"h q[{q1}];"]
                suf = [f"h q[{q0}];", f"h q[{q1}];"]
            elif name_upper == "RYY":
                # Y = S X S^dagger, so conjugating RXX gives RYY exactly.
                pre = [f"s q[{q0}];", f"s q[{q1}];",
                       f"h q[{q0}];", f"h q[{q1}];"]
                suf = [f"h q[{q0}];", f"h q[{q1}];",
                       f"sdg q[{q0}];", f"sdg q[{q1}];"]
            else:
                pre, suf = [], []
            emit(pre)
            emit([
                f"cx q[{q0}], q[{q1}];",
                f"u1({theta}) q[{q1}];",
                f"cx q[{q0}], q[{q1}];",
            ])
            emit(suf)
            continue

        if qasm_name == 'u1':
            if name_upper == 'RZ':
                phi = gate.params[0]
            elif name_upper == 'Z':
                phi = "pi"
            elif name_upper == 'S':
                phi = "pi/2"
            elif name_upper == 'SDG':
                phi = "-pi/2"
            elif name_upper == 'T':
                phi = "pi/4"
            elif name_upper == 'TDG':
                phi = "-pi/4"
            else:
                phi = gate.params[0] if gate.params else 0
            params_str = f"({phi})"
        elif qasm_name == 'u3':
            if name_upper == 'RX':
                params_str = f"({gate.params[0]}, -1.57079632679, 1.57079632679)"
            elif name_upper == 'RY':
                params_str = f"({gate.params[0]}, 0, 0)"
            elif name_upper == 'X':
                params_str = "(pi, 0, pi)"
            elif name_upper == 'Y':
                params_str = "(pi, pi/2, pi/2)"
            elif name_upper == 'SX':
                # sx == u3(pi/2, -pi/2, pi/2), verified to 6e-17
                params_str = "(1.57079632679, -1.57079632679, 1.57079632679)"
            else:
                params_str = f"({', '.join(map(str, gate.params))})"
        else:
            params_str = f"({', '.join(map(str, gate.params))})" if gate.params else ""
            if name_upper in ("CU", "CU3") and len(gate.params) == 3:
                # qelib1's ``cu``/``cu3`` take three angles; a three-angle CU
                # record is the pre-``gamma`` convention, i.e. gamma = 0.
                params_str = (
                    f"({', '.join(map(str, list(gate.params) + [0.0]))})"
                )

        qubits_str = ", ".join([f"q[{i}]" for i in gate.qubits])
        emit([f"{qasm_name}{params_str} {qubits_str};"])

    return "\n".join(qasm)


# ═════════════════════════════════════════════════════════════════════════
# Cirq Bridge
# ═════════════════════════════════════════════════════════════════════════

def from_cirq(cirq_circuit: Any) -> sf.Circuit:
    """Convert a Cirq Circuit to a Superfermion Circuit.

    Args:
        cirq_circuit: A cirq.Circuit instance.

    Returns:
        Equivalent sf.Circuit.

    Example:
        >>> import cirq
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> cirq_circ = cirq.Circuit(
        ...     cirq.H(q0),
        ...     cirq.CNOT(q0, q1)
        ... )
        >>> sf_circuit = from_cirq(cirq_circ)
    """
    try:
        import cirq
    except ImportError:
        raise ImportError("Cirq is required: pip install cirq")

    # Get qubits from circuit
    qubits = sorted(cirq_circuit.all_qubits(), key=lambda q: q.x if hasattr(q, 'x') else str(q))
    n_qubits = len(qubits)

    # Map qubits to indices
    qubit_map = {q: i for i, q in enumerate(qubits)}

    circuit = sf.Circuit(n_qubits)

    # Gate mapping: Cirq gate -> Superfermion method
    gate_map = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'T': 't',
        'HPowGate': 'h',  # H with exponent
        'XPowGate': 'rx',
        'YPowGate': 'ry',
        'ZPowGate': 'rz',
        'CNOT': 'cx', 'CNOTPowGate': 'cx', 'CXPowGate': 'cx',
        'CZ': 'cz', 'CZPowGate': 'cz',
        'SWAP': 'swap',
        'MeasurementGate': 'measure',
        'Rx': 'rx', 'Ry': 'ry', 'Rz': 'rz',  # Aliases
    }

    for moment in cirq_circuit:
        for op in moment.operations:
            gate = op.gate
            gate_type = type(gate).__name__

            # Get qubit indices
            qubit_indices = [qubit_map[q] for q in op.qubits]

            # Map gate by exact type first
            sf_name = gate_map.get(gate_type)

            if sf_name is None:
                if hasattr(gate, '_unitary_') or gate_type == 'MatrixGate':
                    import numpy as np
                    matrix = np.array(cirq.unitary(gate))
                    circuit.unitary(matrix, qubit_indices)
                    continue
                raise ValueError(f"Unsupported Cirq gate: {gate_type} ({gate})")

            method = getattr(circuit, sf_name)

            # Handle parameterized gates
            if gate_type in ('XPowGate', 'YPowGate', 'ZPowGate', 'Rx', 'Ry', 'Rz'):
                # Cirq uses exponent, need to multiply by pi for rotations
                # For Rx/Ry/Rz, the angle is directly accessible via _rads or param
                if gate_type in ('Rx', 'Ry', 'Rz'):
                    # Try to get the angle directly from the gate
                    if hasattr(gate, '_rads'):
                        angle = gate._rads
                    elif hasattr(gate, 'exponent'):
                        angle = gate.exponent * 3.14159265359
                    else:
                        # Parse from string representation
                        gate_str = str(gate)
                        # Handle formats like "Rx(0.15915494309189535π)" or "Rx(0.5)"
                        import re
                        match = re.search(r'\(([0-9.]+)(π|pi)?\)', gate_str)
                        if match:
                            val = float(match.group(1))
                            if match.group(2):  # Has pi
                                angle = val * 3.14159265359
                            else:
                                angle = val
                        else:
                            angle = 0.0
                else:
                    exponent = gate.exponent if hasattr(gate, 'exponent') else 1.0
                    angle = exponent * 3.14159265359  # pi
                method(angle, qubit_indices[0])
            elif gate_type == 'HPowGate':
                # H gate with possible exponent
                if len(qubit_indices) == 1:
                    method(qubit_indices[0])
            elif gate_type in ('CNOT', 'CNOTPowGate', 'CXPowGate', 'CZ', 'CZPowGate', 'SWAP'):
                # Two-qubit gates
                method(qubit_indices[0], qubit_indices[1])
            elif gate_type == 'MeasurementGate':
                method(*qubit_indices)
            elif len(qubit_indices) == 1:
                # Single-qubit gates (H, X, Y, Z, S, T)
                method(qubit_indices[0])
            elif len(qubit_indices) == 2:
                method(qubit_indices[0], qubit_indices[1])
            else:
                method(*qubit_indices)

    return circuit


def to_cirq(circuit: sf.Circuit) -> Any:
    """Convert a Superfermion Circuit to a Cirq Circuit.

    Requires: pip install cirq
    """
    try:
        import cirq
    except ImportError:
        raise ImportError("Cirq is required: pip install cirq")

    # Create qubits
    qubits = [cirq.LineQubit(i) for i in range(circuit.n_qubits)]

    # Gate mapping
    gate_map = {
        'H': cirq.H,
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z,
        'S': cirq.S,
        'SDG': cirq.S ** -1,
        'T': cirq.T,
        'TDG': cirq.T ** -1,
        'RX': cirq.rx,
        'RY': cirq.ry,
        'RZ': cirq.rz,
        'CX': cirq.CNOT,
        'CNOT': cirq.CNOT,
        'CZ': cirq.CZ,
        'SWAP': cirq.SWAP,
    }

    operations = []
    _reject_conditions(circuit, "to_cirq")

    circuit._ensure_gates()  # no-op for default Python storage

    for gate in circuit._gates:
        gate_name = gate.name.upper()

        if gate_name == "UNITARY" and gate.matrix is not None:
            target_qubits = [qubits[i] for i in gate.qubits]
            operations.append(cirq.MatrixGate(gate.matrix).on(*target_qubits))
            continue

        cirq_gate = gate_map.get(gate_name)

        if cirq_gate is None:
            raise ValueError(f"Unsupported gate for Cirq conversion: {gate_name}")

        target_qubits = [qubits[i] for i in gate.qubits]

        if gate_name in ('RX', 'RY', 'RZ'):
            angle = gate.params[0] if gate.params else 0
            operations.append(cirq_gate(angle).on(*target_qubits))
        else:
            operations.append(cirq_gate.on(*target_qubits))

    return cirq.Circuit(operations)


# ═════════════════════════════════════════════════════════════════════════
# PennyLane Bridge
# ═════════════════════════════════════════════════════════════════════════

def to_pennylane(circuit: sf.Circuit) -> Any:
    """Convert a Superfermion Circuit to a PennyLane quantum function.

    Returns:
        A callable that can be used as a PennyLane QNode.

    Example:
        >>> import pennylane as qml
        >>> sf_circuit = sf.Circuit(2)
        >>> sf_circuit.h(0).cx(0, 1)
        >>> qfunc = to_pennylane(sf_circuit)
        >>> dev = qml.device('default.qubit', wires=2)
        >>> @qml.qnode(dev)
        ... def circuit():
        ...     qfunc()
        ...     return qml.state()
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required: pip install pennylane")

    # Gate mapping
    gate_map = {
        'H': qml.Hadamard,
        'X': qml.PauliX,
        'Y': qml.PauliY,
        'Z': qml.PauliZ,
        'S': qml.S,
        'SDG': qml.adjoint(qml.S),
        'T': qml.T,
        'TDG': qml.adjoint(qml.T),
        'RX': qml.RX,
        'RY': qml.RY,
        'RZ': qml.RZ,
        'CX': qml.CNOT,
        'CNOT': qml.CNOT,
        'CZ': qml.CZ,
        'SWAP': qml.SWAP,
        'MEASURE': None,  # Measurement handled separately
    }

    # Store circuit data
    _reject_conditions(circuit, "to_pennylane")
    circuit._ensure_gates()  # no-op for default Python storage
    n_qubits = circuit.n_qubits
    gates = list(circuit._gates)

    def qfunc():
        """Generated PennyLane quantum function."""
        for gate in gates:
            gate_name = gate.name.upper()
            pl_gate = gate_map.get(gate_name)

            if pl_gate is None:
                if gate_name == 'MEASURE':
                    # PennyLane measurements are returns
                    continue
                raise ValueError(f"Unsupported gate for PennyLane: {gate_name}")

            wires = gate.qubits

            if gate_name in ('RX', 'RY', 'RZ'):
                angle = gate.params[0] if gate.params else 0
                pl_gate(angle, wires=wires[0])
            elif len(wires) == 1:
                pl_gate(wires=wires[0])
            elif len(wires) == 2:
                pl_gate(wires=wires)
            else:
                pl_gate(wires=wires)

    # Attach metadata
    qfunc.n_qubits = n_qubits
    qfunc.gates = gates

    return qfunc


# ═════════════════════════════════════════════════════════════════════════
# Stim Bridge
# ═════════════════════════════════════════════════════════════════════════

def to_stim(circuit: sf.Circuit):
    """Convert a Superfermion Clifford circuit to a Stim circuit.

    Gives SF users access to Stim's fast tableau sampler, detector
    machinery, and sinter/pymatching ecosystem for circuits built in SF.

    Only Clifford gates are supported (H, X, Y, Z, S, SDG, SX, CX/CNOT,
    CZ, CY, SWAP, ID, RESET, MEASURE, BARRIER→TICK); anything else raises
    ``ValueError`` — use ``method="statevector"`` for universal circuits.

    Args:
        circuit: sf.Circuit built from Clifford gates (and measure/reset).

    Returns:
        ``(stim_circuit, record_cbits)`` where ``record_cbits[k]`` is the
        SF classical-bit index of the k-th measurement in Stim's record
        order (``None`` for measurements without an explicit cbit).
        Convert a Stim sample row to an SF-style counts key via::

            key = "".join(
                str(row[k]) if (c := record_cbits[k]) is not None else "0"
                for k in range(len(row))
            )

        Note: SF counts keys list classical bits cbit-0-first
        (leftmost char = cbit 0).

    Requires: pip install stim

    Example:
        >>> code = RepetitionCode(n=3).build()
        >>> stim_circuit, order = to_stim(code)
        >>> samples = stim_circuit.compile_sampler().sample(1000)
    """
    try:
        import stim
    except ImportError:
        raise ImportError("Stim is required: pip install stim")

    gate_map = {
        'H': 'H', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
        'S': 'S', 'SDG': 'S_DAG', 'SX': 'SX',
        'CX': 'CX', 'CNOT': 'CX',
        'CZ': 'CZ', 'CY': 'CY',
        'SWAP': 'SWAP', 'ID': 'I',
        'RESET': 'R',
    }

    lines = []
    record_cbits = []
    _reject_conditions(circuit, "to_stim")
    for gate in circuit.to_gate_list():
        name = str(gate["name"]).upper()
        qubits = [int(q) for q in gate["qubits"]]
        if name == "BARRIER":
            lines.append("TICK")
            continue
        if name == "MEASURE":
            cbits = gate.get("classical_bits", []) or []
            cbit = int(cbits[0]) if cbits else None
            lines.append(f"M {qubits[0]}")
            record_cbits.append(cbit)
            continue
        stim_name = gate_map.get(name)
        if stim_name is None:
            raise ValueError(
                f"Cannot map gate '{gate['name']}' to Stim "
                "(to_stim supports Clifford circuits only)"
            )
        lines.append(f"{stim_name} {' '.join(map(str, qubits))}")
    return stim.Circuit("\n".join(lines)), record_cbits


# ═════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════

__all__ = [
    "from_qiskit",
    "to_qiskit",
    "from_pennylane",
    "to_pennylane",
    "from_cirq",
    "to_cirq",
    "to_ionq",
    "to_braket",
    "to_qasm",
    "from_qasm",
    "to_stim",
]
