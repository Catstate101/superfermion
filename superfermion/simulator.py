"""
Statevector quantum simulator — pure Python implementation.

Simulates quantum circuits by direct statevector evolution.
Each gate is applied as a unitary matrix multiplication on the full state.

This is the default simulator backend. For larger circuits, GPU/MPS
backends will be used (Sessions 13-14).

Performance:
    - Up to ~20 qubits: fast enough for interactive use
    - Up to ~25 qubits: feasible on modern machines (8GB RAM)
    - Beyond: use GPU (CuQuantum) or MPS backends
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from superfermion.circuit import Circuit, GateRecord
from superfermion.parameters import SymbolicParameter


# Type alias for complex statevector
StateVector = NDArray[np.complex128]


def _get_gate_matrix(gate: GateRecord) -> NDArray[np.complex128]:
    """Get the unitary matrix for a gate.

    Phase conventions match the Rust IR (ops.rs):
    Rx(θ) = exp(-iθX/2) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    """
    name = gate.name
    params = [
        float(p) if not isinstance(p, SymbolicParameter) else 0.0
        for p in gate.params
    ]

    if name == "ID":
        return np.eye(2, dtype=np.complex128)

    elif name == "H":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)

    elif name == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)

    elif name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    elif name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)

    elif name == "S":
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)

    elif name == "SDG":
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)

    elif name == "T":
        return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)

    elif name == "TDG":
        return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)

    elif name == "SX":
        return np.array([
            [0.5 + 0.5j, 0.5 - 0.5j],
            [0.5 - 0.5j, 0.5 + 0.5j],
        ], dtype=np.complex128)

    elif name == "RX":
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([
            [c, -1j * s],
            [-1j * s, c],
        ], dtype=np.complex128)

    elif name == "RY":
        theta = params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([
            [c, -s],
            [s, c],
        ], dtype=np.complex128)

    elif name == "RZ":
        theta = params[0]
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ], dtype=np.complex128)

    elif name in ("R1", "P"):
        phi = params[0]
        return np.array([
            [1, 0],
            [0, np.exp(1j * phi)],
        ], dtype=np.complex128)

    elif name == "U":
        theta, phi, lam = params[0], params[1], params[2]
        ct = math.cos(theta / 2)
        st = math.sin(theta / 2)
        return np.array([
            [ct, -np.exp(1j * lam) * st],
            [np.exp(1j * phi) * st, np.exp(1j * (phi + lam)) * ct],
        ], dtype=np.complex128)

    elif name == "CNOT":
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.complex128)

    elif name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)

    elif name == "SWAP":
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)

    elif name == "ISWAP":
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)

    elif name == "RZZ":
        theta = params[0]
        em = np.exp(-1j * theta / 2)
        ep = np.exp(1j * theta / 2)
        return np.diag([em, ep, ep, em]).astype(np.complex128)

    elif name == "CCX":
        m = np.eye(8, dtype=np.complex128)
        m[6, 6] = 0
        m[7, 7] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        return m

    else:
        raise ValueError(
            f"Gate '{name}' not implemented in simulator.\n"
            f"  Supported gates: H, X, Y, Z, S, Sdg, T, Tdg, SX, Id,\n"
            f"  Rx, Ry, Rz, R1, P, U, CNOT, CZ, SWAP, Rzz, CCX"
        )


def _apply_single_qubit_gate(
    state: StateVector,
    gate_matrix: NDArray[np.complex128],
    qubit: int,
    n_qubits: int,
) -> StateVector:
    """Apply a single-qubit gate to the statevector.

    Uses tensor reshaping for efficiency:
    state is reshaped to (2, 2, ..., 2), gate is applied on the qubit axis,
    then reshaped back to the flat vector.
    """
    # Reshape state into tensor of shape (2, 2, ..., 2)
    state_tensor = state.reshape([2] * n_qubits)

    # Apply gate by contracting along the qubit axis
    # np.tensordot(gate, state, axes=([1], [qubit]))
    # moves the qubit axis to position 0, so we need to move it back
    result = np.tensordot(gate_matrix, state_tensor, axes=([1], [qubit]))

    # Move the qubit axis back to position `qubit`
    result = np.moveaxis(result, 0, qubit)

    return result.reshape(-1)


def _apply_two_qubit_gate(
    state: StateVector,
    gate_matrix: NDArray[np.complex128],
    qubit1: int,
    qubit2: int,
    n_qubits: int,
) -> StateVector:
    """Apply a two-qubit gate to the statevector."""
    state_tensor = state.reshape([2] * n_qubits)
    gate_tensor = gate_matrix.reshape(2, 2, 2, 2)

    # Contract gate with state on both qubit axes
    result = np.tensordot(
        gate_tensor, state_tensor,
        axes=([2, 3], [qubit1, qubit2]),
    )
    # Move axes back
    # After tensordot, the first two axes are the output of the gate
    # and need to go to positions qubit1 and qubit2
    # result has shape (2, 2, other_dims...)
    # We move axis 0 (output 1) to qubit1 and axis 1 (output 2) to qubit2
    result = np.moveaxis(result, [0, 1], [qubit1, qubit2])

    return result.reshape(-1)


def _apply_three_qubit_gate(
    state: StateVector,
    gate_matrix: NDArray[np.complex128],
    qubit1: int,
    qubit2: int,
    qubit3: int,
    n_qubits: int,
) -> StateVector:
    """Apply a three-qubit gate to the statevector."""
    state_tensor = state.reshape([2] * n_qubits)
    gate_tensor = gate_matrix.reshape(2, 2, 2, 2, 2, 2)

    result = np.tensordot(
        gate_tensor, state_tensor,
        axes=([3, 4, 5], [qubit1, qubit2, qubit3]),
    )
    source = [0, 1, 2]
    dest = sorted([qubit1, qubit2, qubit3])
    result = np.moveaxis(result, source, dest)

    return result.reshape(-1)


def simulate_statevector(
    circuit: Circuit,
    initial_state: Optional[StateVector] = None,
) -> StateVector:
    """Simulate the circuit and return the final statevector.

    Args:
        circuit: The quantum circuit to simulate.
        initial_state: Optional initial statevector. Defaults to |0...0⟩.

    Returns:
        Final statevector as a complex numpy array.
    """
    n = circuit.n_qubits
    dim = 2 ** n

    if initial_state is not None:
        state = initial_state.copy().astype(np.complex128)
        assert len(state) == dim, f"State dimension {len(state)} != 2^{n}"
    else:
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0  # |0...0⟩

    # 2. Gate execution
    # Matrix Cache: Massive speedup for repetitive rotations
    _mat_cache = {}
    
    for gate in circuit._gates:
        name = gate.name.upper()
        if name == "BARRIER":
            continue
        
        # Cache key
        g_key = (name, tuple(gate.params) if gate.params else None)
        if g_key not in _mat_cache:
            _mat_cache[g_key] = _get_gate_matrix(gate)
        matrix = _mat_cache[g_key]
        
        n_gate_qubits = len(gate.qubits)

        if n_gate_qubits == 1:
            state = _apply_single_qubit_gate(state, matrix, gate.qubits[0], n)
        elif n_gate_qubits == 2:
            state = _apply_two_qubit_gate(state, matrix, gate.qubits[0], gate.qubits[1], n)
        elif n_gate_qubits == 3:
            state = _apply_three_qubit_gate(
                state, matrix, gate.qubits[0], gate.qubits[1], gate.qubits[2], n,
            )

    return state


def sample_counts(
    statevector: StateVector,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> dict[str, int]:
    """Sample measurement outcomes from a statevector.

    Args:
        statevector: The state to sample from.
        shots: Number of measurement shots.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping bitstrings to counts. e.g. {"00": 503, "11": 497}
    """
    rng = random.Random(seed)
    probs = np.abs(statevector) ** 2
    # Normalize to handle floating point drift
    probs = probs / probs.sum()

    n_qubits = int(math.log2(len(statevector)))
    dim = len(statevector)

    # Use numpy for efficient sampling
    np_rng = np.random.default_rng(seed)
    indices = np_rng.choice(dim, size=shots, p=probs)

    counts: dict[str, int] = {}
    for idx in indices:
        bitstring = format(idx, f"0{n_qubits}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


def expectation_value(
    statevector: StateVector,
    observable_matrix: NDArray[np.complex128],
) -> float:
    """Compute ⟨ψ|O|ψ⟩ for observable O.

    Args:
        statevector: The quantum state |ψ⟩.
        observable_matrix: Hermitian operator O.

    Returns:
        Real expectation value ⟨ψ|O|ψ⟩.
    """
    return float(np.real(statevector.conj() @ observable_matrix @ statevector))
