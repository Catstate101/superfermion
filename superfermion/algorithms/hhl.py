"""
HHL Algorithm — Quantum linear system solver (Harrow-Hassidim-Lloyd).

Solves A|x⟩ = |b⟩ for a Hermitian matrix A in O(log N · κ²/ε) time,
exponentially faster than classical methods for well-conditioned sparse systems.

The implementation uses:
  1. State preparation of |b⟩
  2. QPE to extract eigenvalues of A (via Hamiltonian simulation e^{iAt})
  3. Eigenvalue inversion (1/λ_j rotation)
  4. Inverse QPE to uncompute eigenvalues
  5. Measurement + post-selection on ancilla

Usage:
    >>> from superfermion.algorithms.hhl import hhl_solve
    >>>
    >>> A = np.array([[1.5, 0.5], [0.5, 1.5]])  # Hermitian 2×2
    >>> b = np.array([1.0, 0.0])     # |b⟩ = |0⟩
    >>> result = hhl_solve(A, b, precision_bits=3, t_scale=2.0)
    >>> print(result["solution"])     # ≈ [0.75, −0.25] / |...|
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import superfermion as sf


def _controlled_unitary_matrix(
    circuit: sf.Circuit,
    U: np.ndarray,
    control: int,
    target_qubits: List[int],
):
    """Apply a controlled-U gate where U is an arbitrary 2^n × 2^n unitary.

    For single-target (2×2 U), uses the ZYZ decomposition:
        U = e^{iα} Rz(β) Ry(γ) Rz(δ)
    then constructs CU from single-qubit gates + CNOT.
    """
    n = len(target_qubits)
    if n == 1:
        _controlled_1q_unitary(circuit, U, control, target_qubits[0])
    else:
        _controlled_nq_unitary(circuit, U, control, target_qubits)


def _controlled_1q_unitary(
    circuit: sf.Circuit,
    U: np.ndarray,
    control: int,
    target: int,
):
    """Controlled 2×2 unitary via ZYZ decomposition."""
    alpha, beta, gamma, delta = _zyz_decompose(U)

    circuit.rz((delta - beta) / 2, target)
    circuit.cx(control, target)
    circuit.rz(-(delta + beta) / 2, target)
    circuit.ry(-gamma / 2, target)
    circuit.cx(control, target)
    circuit.ry(gamma / 2, target)
    circuit.rz(beta, target)
    circuit.p(alpha, control)


def _zyz_decompose(U: np.ndarray) -> Tuple[float, float, float, float]:
    """Decompose U = e^{iα} Rz(β) Ry(γ) Rz(δ)."""
    det = np.linalg.det(U)
    alpha = float(np.angle(det) / 2)
    V = U / np.exp(1j * alpha)

    gamma = 2 * np.arccos(np.clip(np.abs(V[0, 0]), -1, 1))
    if abs(np.sin(gamma / 2)) < 1e-12:
        beta = float(np.angle(V[0, 0]))
        delta = 0.0
    elif abs(np.cos(gamma / 2)) < 1e-12:
        beta = float(np.angle(V[1, 0]))
        delta = 0.0
    else:
        beta = float(np.angle(V[1, 1]) + np.angle(V[1, 0]))
        delta = float(np.angle(V[1, 1]) - np.angle(V[1, 0]))

    return alpha, beta, gamma, delta


def _controlled_nq_unitary(
    circuit: sf.Circuit,
    U: np.ndarray,
    control: int,
    target_qubits: List[int],
):
    """Controlled multi-qubit unitary via eigendecomposition.

    CU = I ⊗ |0><0| + U ⊗ |1><1|

    For small matrices, we decompose U = V D V† and apply controlled rotations.
    """
    N = U.shape[0]
    eigenvalues, V = np.linalg.eigh(
        -1j * _log_unitary(U)
    )

    for q_idx in range(len(target_qubits)):
        q = target_qubits[q_idx]
        _controlled_1q_unitary(circuit, V[:2, :2] if N == 2 else np.eye(2), control, q)

    for i in range(N):
        phase = float(eigenvalues[i])
        if abs(phase) > 1e-12:
            circuit.cp(phase, control, target_qubits[0])


def _log_unitary(U: np.ndarray) -> np.ndarray:
    """Matrix logarithm of a unitary: U = e^{iH}, return H."""
    eigenvalues, V = np.linalg.eig(U)
    log_eigenvalues = np.log(eigenvalues)
    return V @ np.diag(log_eigenvalues) @ np.linalg.inv(V)


def _iqft(circuit: sf.Circuit, qubits: List[int]):
    """Inverse QFT matching the QPE phase convention."""
    n = len(qubits)
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])
    for i in range(n):
        for j in range(i):
            angle = -np.pi / (2 ** (i - j))
            circuit.cp(angle, qubits[j], qubits[i])
        circuit.h(qubits[i])


def _qft(circuit: sf.Circuit, qubits: List[int]):
    """Forward QFT (inverse of _iqft)."""
    n = len(qubits)
    for i in range(n - 1, -1, -1):
        circuit.h(qubits[i])
        for j in range(i - 1, -1, -1):
            angle = np.pi / (2 ** (i - j))
            circuit.cp(angle, qubits[j], qubits[i])
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])


def hhl_solve(
    A: np.ndarray,
    b: np.ndarray,
    precision_bits: int = 4,
    t_scale: float = 1.0,
    device: Any = "cpu",
    method: str = "statevector",
) -> Dict[str, Any]:
    r"""Solve the linear system A|x⟩ = |b⟩ using the HHL algorithm.

    Args:
        A: Hermitian N × N matrix (N must be a power of 2).
        b: Right-hand side vector of length N.
        precision_bits: Number of qubits in the eigenvalue estimation register.
        t_scale: Time scaling factor for Hamiltonian simulation e^{iAt}.
        device: Execution target — ``"cpu"``, ``"gpu"``, or ``DeviceExecutor``.
        method: Simulation method — ``"statevector"``, ``"mps"``, etc.

    Returns:
        Dict with keys:
          - ``"solution"``: complex array of the normalized solution vector.
          - ``"success_probability"``: probability of measuring ancilla |1⟩.
          - ``"eigenvalues"``: estimated eigenvalues of A.
          - ``"precision_bits"``: number of QPE estimation qubits.
    """
    N = A.shape[0]
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"Matrix dimension {N} is not a power of 2")

    b_vec = np.asarray(b, dtype=complex).flatten()
    b_norm = np.linalg.norm(b_vec)
    if b_norm < 1e-15:
        raise ValueError("|b⟩ has zero norm")
    b_vec = b_vec / b_norm

    t = precision_bits
    ancilla_idx = t + n
    total = t + n + 1
    circuit = sf.Circuit(total, name=f"HHL(t={t}, n={n})")

    data_qubits = list(range(t, t + n))
    counting_qubits = list(range(t))

    # 1. Encode |b⟩ on data register
    _encode_state(circuit, b_vec, data_qubits)

    # 2. QPE: estimate eigenvalues of A
    for i in range(t):
        circuit.h(i)

    # Controlled-U^{2^k} where U = e^{iA*t_scale}
    for k in range(t):
        power = 2 ** k
        U_k = _matrix_exp(1j * A * t_scale * power)
        _controlled_unitary_matrix(circuit, U_k, k, data_qubits)

    _iqft(circuit, counting_qubits)

    # 3. Eigenvalue inversion via controlled rotations on ancilla
    # For each basis state |j⟩ of the counting register, the eigenvalue is
    # λ_j = 2π·j / (2^t · t_scale). We apply RY(2·arcsin(C/λ_j)) on
    # the ancilla conditioned on qubit j being 1.
    # C is chosen so that C/λ_max ≤ 1.
    eigenvalues_A = np.linalg.eigvalsh(A)
    lambda_min = np.min(np.abs(eigenvalues_A[np.abs(eigenvalues_A) > 1e-12]))
    C = 0.5 * lambda_min

    for j in range(t):
        bit_val = 2 ** j
        lambda_j = 2 * np.pi * bit_val / (2 ** t * t_scale)
        if lambda_j < 1e-12:
            continue
        ratio = min(C / lambda_j, 1.0)
        angle = 2 * np.arcsin(ratio)
        # Controlled-RY(angle) on ancilla, controlled by counting qubit j
        circuit.ry(angle / 2, ancilla_idx)
        circuit.cx(j, ancilla_idx)
        circuit.ry(-angle / 2, ancilla_idx)
        circuit.cx(j, ancilla_idx)

    # 4. Inverse QPE (uncompute)
    _qft(circuit, counting_qubits)

    for k in range(t - 1, -1, -1):
        power = 2 ** k
        U_k_dag = _matrix_exp(-1j * A * t_scale * power)
        _controlled_unitary_matrix(circuit, U_k_dag, k, data_qubits)

    for i in range(t):
        circuit.h(i)

    result = sf.run(circuit, device=device, method=method, shots=0)

    if result.statevector is None:
        return {
            "solution": np.zeros(N, dtype=complex),
            "success_probability": 0.0,
            "eigenvalues": [],
            "precision_bits": t,
        }

    sv = np.asarray(result.statevector).flatten()

    # Post-select on ancilla = |1⟩ and counting register = |0...0⟩
    # Extract the data-register amplitudes
    solution = np.zeros(N, dtype=complex)
    ancilla_bit = 1 << ancilla_idx
    for i in range(len(sv)):
        if (i & ancilla_bit) == 0:
            continue
        counting_val = i & ((1 << t) - 1)
        if counting_val != 0:
            continue
        data_val = (i >> t) & ((1 << n) - 1)
        solution[data_val] = sv[i]

    norm = np.linalg.norm(solution)
    success_prob = norm ** 2
    if norm > 1e-12:
        solution = solution / norm

    return {
        "solution": solution,
        "success_probability": float(success_prob),
        "eigenvalues": eigenvalues_A.tolist(),
        "precision_bits": t,
        "result": result,
    }


def _matrix_exp(M: np.ndarray) -> np.ndarray:
    """Matrix exponential via eigendecomposition."""
    eigenvalues, V = np.linalg.eig(M)
    return V @ np.diag(np.exp(eigenvalues)) @ np.linalg.inv(V)


def _encode_state(circuit: sf.Circuit, state: np.ndarray, qubits: List[int]):
    """Encode a normalized state vector into qubits via angle decomposition."""
    N = len(state)
    n = len(qubits)
    if N == 2:
        theta = 2 * np.arctan2(np.abs(state[1]), np.abs(state[0]) + 1e-15)
        circuit.ry(float(theta), qubits[0])
        if np.abs(state[1]) > 1e-12:
            phase = np.angle(state[1]) - np.angle(state[0])
            if abs(phase) > 1e-12:
                circuit.rz(float(phase), qubits[0])
    else:
        for i in range(N):
            if np.abs(state[i]) > 1e-12 and i > 0:
                bits = format(i, f"0{n}b")
                for j, b in enumerate(reversed(bits)):
                    if b == "1":
                        circuit.x(qubits[j])
