"""
Framework Bridges — Import circuits from Qiskit, PennyLane, and Cirq.

Usage:
    from superfermion.bridge import from_qiskit, from_pennylane, from_cirq
    sf_circuit = from_qiskit(qiskit_circuit)
"""

from __future__ import annotations

from typing import Any
import superfermion as sf


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
    GATE_MAP = {
        'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z',
        's': 's', 'sdg': 'sdg', 't': 't', 'tdg': 'tdg',
        'sx': 'sx', 'id': 'id',
        'rx': 'rx', 'ry': 'ry', 'rz': 'rz',
        'p': 'p', 'u': 'u', 'u3': 'u3',
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
    
    for instruction in qiskit_circuit.data:
        gate = instruction.operation
        gate_name = gate.name.lower()
        qubits = [n_qubits - 1 - qiskit_circuit.find_bit(q).index for q in instruction.qubits]
        params = list(gate.params) if gate.params else []
        
        if gate_name == "unitary":
            import numpy as np
            matrix = np.array(gate.to_matrix())
            circuit.unitary(matrix, qubits)
            continue
        
        sf_name = GATE_MAP.get(gate_name)
        if sf_name is None:
            raise ValueError(
                f"Unsupported Qiskit gate: '{gate.name}'. "
                f"Supported: {list(GATE_MAP.keys()) + ['unitary']}"
            )
        
        method = getattr(circuit, sf_name)
        
        if params:
            method(*params, *qubits)
        else:
            method(*qubits)
    
    return circuit


def to_qiskit(circuit: sf.Circuit) -> Any:
    """Convert a Superfermion Circuit to a Qiskit QuantumCircuit.
    
    Requires qiskit to be installed.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit is required: pip install qiskit")
    
    qc = QuantumCircuit(circuit.n_qubits, circuit.n_cbits)
    
    GATE_MAP = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'sdg', 'T': 't', 'TDG': 'tdg',
        'SX': 'sx', 'ID': 'id',
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
    
    for gate in circuit._gates:
        if gate.name.upper() == "UNITARY" and gate.matrix is not None:
            from qiskit.circuit.library import UnitaryGate
            mapped_qubits = [circuit.n_qubits - 1 - q for q in gate.qubits]
            qc.append(UnitaryGate(gate.matrix), mapped_qubits)
            continue

        qiskit_name = GATE_MAP.get(gate.name.upper())
        if qiskit_name is None:
            raise ValueError(f"Cannot map gate '{gate.name}' to Qiskit")
        
        method = getattr(qc, qiskit_name)
        # Reverse endianness: SF MSB (0) -> Qiskit LSB (n-1)
        mapped_qubits = [circuit.n_qubits - 1 - q for q in gate.qubits]
        
        if gate.name == "MEASURE":
            cbit = gate.classical_bits[0] if gate.classical_bits else gate.qubits[0]
            # Classical bits usually follow the same order or are left as is? 
            # In Qiskit, cbit 0 is also rightmost. So we reverse them too.
            mapped_cbit = circuit.n_cbits - 1 - cbit
            method(mapped_qubits[0], mapped_cbit)
        elif gate.name in ("CU", "CU3"):
            theta, phi, lam = gate.params
            method(theta, phi, lam, 0, *mapped_qubits)
        elif gate.name == "U3":
            method(*gate.params, *mapped_qubits)
        elif gate.params:
            method(*gate.params, *mapped_qubits)
        else:
            method(*mapped_qubits)
    
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
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required: pip install pennylane")
    
    # Get the tape from the QNode
    if hasattr(pennylane_circuit, 'tape'):
        tape = pennylane_circuit.tape
    else:
        tape = pennylane_circuit
    
    if n_qubits is None:
        n_qubits = len(tape.wires)
    
    circuit = sf.Circuit(n_qubits)
    
    PL_MAP = {
        'Hadamard': 'h', 'PauliX': 'x', 'PauliY': 'y', 'PauliZ': 'z',
        'S': 's', 'T': 't',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CNOT': 'cx', 'CZ': 'cz',
        'SWAP': 'swap',
        'Toffoli': 'ccx',
    }
    
    for op in tape.operations:
        sf_name = PL_MAP.get(op.name)
        if sf_name is None:
            raise ValueError(
                f"Unsupported PennyLane operation: '{op.name}'. "
                f"Supported: {list(PL_MAP.keys())}"
            )
        
        method = getattr(circuit, sf_name)
        wires = [int(w) for w in op.wires]
        params = list(op.parameters) if op.parameters else []
        
        if params:
            method(*params, *wires)
        else:
            method(*wires)
    
    return circuit


def from_qasm(qasm_str: str) -> sf.Circuit:
    """Parse an OpenQASM 2.0/3.0 string into a Superfermion Circuit.

    Uses the native Rust parser (6× faster than Qiskit) when available,
    falling back to the pure-Python parser for unsupported syntax.
    """
    # ── Pre-scan: skip Rust path if QASM contains big-integer parameters (> 2^53)
    #   that would lose precision when parsed as IEEE-754 f64.
    import re as _re
    _has_bigint = any(
        int(m) >= 2**53
        for m in _re.findall(r'(?<![\w.])(\d+)(?![.\d])', qasm_str)
        if len(m) >= 16
    ) if _re.search(r'\b\d{16,}\b', qasm_str) else False

    # ── Fast path: Rust-native QASM2 parser ─────────────────
    if not _has_bigint:
        try:
            from superfermion._sf_core import QuantumDAG
            from superfermion.circuit import GateRecord as _GateRecord
            dag = QuantumDAG.from_qasm2(qasm_str)
            records = dag.to_gate_records()
            circ = sf.Circuit(dag.n_qubits(), n_cbits=dag.n_cbits())
            # Single-pass bulk append: avoids intermediate list + double .upper()
            circ.extend_raw_from_records(records)
            return circ
        except BaseException:
            pass  # Fall back to Python parser (catches PanicException too)

    from superfermion.circuit import GateRecord as _GateRecord
    import re
    lines = [l.strip() for l in qasm_str.strip().split('\n') if l.strip()]

    # Gate-name → SF method mapping (hoisted outside loop for speed)
    QASM_MAP = {
        'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z',
        's': 's', 'sdg': 'sdg', 't': 't', 'tdg': 'tdg',
        'sx': 'sx',
        'rx': 'rx', 'ry': 'ry', 'rz': 'rz',
        'u': 'u', 'u3': 'u',
        'p': 'p', 'u1': 'p',
        'cx': 'cx', 'cnot': 'cx',
        'cz': 'cz',
        'swap': 'swap',
        'ccx': 'ccx',
    }
    
    n_qubits = None
    circuit = None
    
    for line in lines:
        # Skip comments and headers
        if line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include'):
            continue
        
        # qubit declaration
        if line.startswith('qubit[') or line.startswith('qreg'):
            # qubit[N] q; OR qreg q[N];
            m = re.search(r'\[(\d+)\]', line)
            if m:
                n_qubits = int(m.group(1))
                circuit = sf.Circuit(n_qubits)
            continue
        
        if circuit is None:
            continue
        
        # Gate instructions
        # Format: gate_name q[i], q[j]; OR gate_name(param) q[i];
        m = re.match(r'(\w+)(?:\(([^)]*)\))?\s+(.+);', line)
        if m:
            gate_name = m.group(1).lower()
            params_str = m.group(2)
            qubits_str = m.group(3)
            
            # Parse qubits
            qubit_indices = [int(x) for x in re.findall(r'\[(\d+)\]', qubits_str)]
            
            # Parse params
            params = []
            if params_str:
                for p in params_str.split(','):
                    p = p.strip()
                    # Handle big integers (> 2^53) that float() would lose precision on.
                    # Try int first, then float, then symbolic.
                    try:
                        as_int = int(p)
                        if abs(as_int) > 2**53:
                            params.append(as_int)  # preserve as Python int
                        else:
                            params.append(float(p))  # small int → float
                    except ValueError:
                        try:
                            params.append(float(p))
                        except ValueError:
                            if 'pi' in p:
                                import math
                                params.append(eval(p.replace('pi', str(math.pi))))
                            else:
                                params.append(sf.param(p))
            
            # Map gate (QASM_MAP hoisted outside loop)
            
            # CRITICAL: Reverse qubit indices to match SF's MSB-first internal representation
            # This ensures SF bitstrings match Qiskit ones 1:1.
            n = circuit.n_qubits
            mapped_indices = [n - 1 - i for i in qubit_indices]
            
            sf_name = QASM_MAP.get(gate_name)
            if sf_name and hasattr(circuit, sf_name):
                # Fast path: append GateRecord directly, skipping
                # per-gate validation (qubits already verified by parser).
                circuit._gates.append(_GateRecord(
                    name=sf_name.upper(),
                    qubits=mapped_indices,
                    params=params,
                ))
    
    if circuit is None:
        raise ValueError("Could not parse QASM: no qubit declaration found")
    
    return circuit


def to_ionq(circuit: sf.Circuit) -> List[Dict[str, Any]]:
    """Convert a Superfermion Circuit to IonQ JSON gate format.
    
    Args:
        circuit: Superfermion Circuit.
        
    Returns:
        List of gate dictionaries as required by IonQ API.
    """
    ionq_gates = []
    
    # Mapper for Superfermion gate names to IonQ gate names
    # Note: IonQ gate names are lowercase (e.g., 'h', 'x', 'cnot')
    GATE_MAP = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SDG': 'si', 'T': 't', 'TDG': 'ti',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CX': 'cnot', 'CNOT': 'cnot',
        'CZ': 'cz',
        'SWAP': 'swap'
    }
    
    for gate in circuit._gates:
        ionq_name = GATE_MAP.get(gate.name.upper())
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
            if ionq_name in ['cnot', 'cz']:
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
        
    GATE_MAP = {
        'H': 'h', 'X': 'u3', 'Y': 'u3', 'Z': 'u1',
        'S': 'u1', 'SDG': 'u1', 'T': 'u1', 'TDG': 'u1',
        'RX': 'u3', 'RY': 'u3', 'RZ': 'u1',
        'CX': 'cx', 'CNOT': 'cx', 'CZ': 'cz',
        'SWAP': 'swap', 'MEASURE': 'measure'
    }
    
    for gate in circuit._gates:
        qasm_name = GATE_MAP.get(gate.name.upper())
        if not qasm_name:
            continue
            
        if gate.name == "MEASURE":
            cbit = gate.classical_bits[0] if gate.classical_bits else gate.qubits[0]
            qasm.append(f"measure q[{gate.qubits[0]}] -> c[{cbit}];")
            continue
            
        # Handle Parameters for U Gates
        name_upper = gate.name.upper()
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
            else:
                params_str = f"({', '.join(map(str, gate.params))})"
        else:
            params_str = f"({', '.join(map(str, gate.params))})" if gate.params else ""
            
        qubits_str = ", ".join([f"q[{i}]" for i in gate.qubits])
        qasm.append(f"{qasm_name}{params_str} {qubits_str};")
        
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
    GATE_MAP = {
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
            sf_name = GATE_MAP.get(gate_type)
            
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
    GATE_MAP = {
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
    
    for gate in circuit._gates:
        gate_name = gate.name.upper()

        if gate_name == "UNITARY" and gate.matrix is not None:
            target_qubits = [qubits[i] for i in gate.qubits]
            operations.append(cirq.MatrixGate(gate.matrix).on(*target_qubits))
            continue

        cirq_gate = GATE_MAP.get(gate_name)
        
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
    GATE_MAP = {
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
    n_qubits = circuit.n_qubits
    gates = list(circuit._gates)
    
    def qfunc():
        """Generated PennyLane quantum function."""
        for gate in gates:
            gate_name = gate.name.upper()
            pl_gate = GATE_MAP.get(gate_name)
            
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
]
