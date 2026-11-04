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
    >>> print(result["solution"])     # ≈ [0.952, −0.305]

The exact answer is ``A^-1 b / |A^-1 b| = [0.9487, -0.3162]``; the docstring
values above are the measured output of that exact call on this build, whose
``|<x_exact|x_hhl>|`` is **0.999930** (ALG-8 measured it at 0.9999298308667551).
(Before the ALG-8 fixes the same call returned ``[0.989, +0.146]`` -- the sign
of the second component was wrong, and its overlap with the exact answer is
0.8924.)

Accuracy is set by the eigenvalue register: raising ``precision_bits`` (at
``t_scale=1.0``, the default) tightens it further --

    ==============  =====================  ============================
    precision_bits  solution               |<x_exact|x_hhl>|
    ==============  =====================  ============================
    3               [0.940656, -0.339361]  0.999700
    4               [0.954225, -0.299091]  0.999838
    5               [0.949090, -0.315004]  0.999999
    6               [0.948678, -0.316245]  1.000000
    ==============  =====================  ============================

so the answer converges to the exact one rather than merely pointing at it.
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
        # gamma = 0  =>  V = Rz(beta + delta), so V[0, 0] = e^{-i(beta+delta)/2}
        # and therefore (beta + delta) = -2 * arg(V[0, 0]). The previous
        # `beta = arg(V[0, 0])` was off by a factor of -2, which silently broke
        # every *diagonal* unitary -- exactly the shape HHL feeds in as
        # exp(i*A*t) for a diagonal A. (ALG-8)
        beta = -2.0 * float(np.angle(V[0, 0]))
        delta = 0.0
    elif abs(np.cos(gamma / 2)) < 1e-12:
        # gamma = pi  =>  V[1, 0] = e^{i(beta - delta)/2}, so
        # (beta - delta) = 2 * arg(V[1, 0]); `beta = arg(V[1, 0])` was off by 2.
        beta = 2.0 * float(np.angle(V[1, 0]))
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
    """Exact controlled multi-qubit unitary: ``blockdiag(I, U)``.

    This used to loop over the target qubits applying
    ``_controlled_1q_unitary(V[:2, :2])`` and then ``cp(eigenvalues[i])`` on
    the *first* target — which never acts on the other targets and does not
    compose into ``U`` for any matrix. It is now a single embedded ``UNITARY``,
    so ``CU = I ⊕ U`` exactly.

    ``Circuit.unitary(M, qs)`` maps ``qs[0]`` to the **most significant** index
    bit of ``M`` (measured against ``ccx``/``cx``), so the control is listed
    first and the targets follow in reverse.
    """
    k = len(target_qubits)
    u = np.asarray(U, dtype=complex)
    if u.shape != (2 ** k, 2 ** k):
        raise ValueError(
            f"controlled unitary shape {u.shape} does not match {k} target "
            f"qubit(s) (expected ({2 ** k}, {2 ** k}))"
        )
    embedded = np.eye(2 ** (k + 1), dtype=complex)
    embedded[2 ** k:, 2 ** k:] = u
    circuit.unitary(embedded, [control] + list(reversed(target_qubits)))


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

    The eigenvalue-estimation register only sees the phase
    ``lambda * t_scale / (2*pi)`` modulo 1, so every eigenvalue must lie in
    ``(0, 2*pi / t_scale)`` for the bin index to identify it. Choosing
    ``t_scale`` so that ``lambda_max * t_scale < 2*pi`` guarantees that;
    otherwise eigenvalues alias to a different bin and are inverted with the
    wrong magnitude (or, for negative eigenvalues, the wrong sign). Small
    ``t_scale`` also widens the window but coarsens the bin spacing.

    Args:
        A: Hermitian N × N matrix (N must be a power of 2).
        b: Right-hand side vector of length N.
        precision_bits: Number of qubits in the eigenvalue estimation register.
            The eigenvalue-inversion step is a single uniformly controlled
            rotation, applied here as one opaque ``unitary`` over
            ``precision_bits + 1`` qubits. The engine's cost for an opaque
            unitary grows steeply with that width, so keep this small: the
            default of 4 runs in milliseconds, 6 in about a second, and 7 or
            more becomes noticeably slow (this is a demonstration algorithm,
            not a production solver).
        t_scale: Time scaling factor for Hamiltonian simulation e^{iAt}.
        device: Execution target — ``"cpu"``, ``"gpu"``, or ``DeviceExecutor``.
        method: Simulation method — ``"statevector"``, ``"mps"``, etc.

    Returns:
        Dict with keys:
          - ``"solution"``: complex array, the normalised ``A^-1 b`` (up to a
            global phase).
          - ``"success_probability"``: probability of measuring the ancilla in
            |1⟩ *and* the counting register in |0…0⟩ -- the post-selected
            branch, whose norm this is.
          - ``"eigenvalues"``: ``np.linalg.eigvalsh(A).tolist()`` -- the
            **exact** spectrum of ``A``, computed classically. It is the list
            the inversion rotation is built from (``C = 0.5 * lambda_min``),
            not a value read back out of the eigenvalue register: the QPE
            estimate never leaves the circuit, and at ``t`` bits the best it
            can say is the bin ``2*pi*j / (2**t * t_scale)``. Judge the
            register by ``"solution"`` against ``A^-1 b`` instead.
          - ``"precision_bits"``: number of QPE estimation qubits (``t``).
          - ``"result"``: the ``RunResult`` of the HHL circuit, so the raw
            statevector stays reachable as ``result.statevector``. Only the
            successful path returns it -- see below.

        When the run produces no statevector (``result.statevector is None``,
        e.g. a method that does not expose one) the dict is
        ``{"solution": zeros, "success_probability": 0.0,
        "eigenvalues": [], "precision_bits": t}``: no ``"result"`` key, and the
        eigenvalues are dropped even though they are known.

    Raises:
        ValueError: ``A`` is not square, ``N`` is not a power of 2, ``A`` is not
            Hermitian, or ``b`` has zero norm.
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square 2-D matrix, got shape {A.shape}")
    N = A.shape[0]
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"Matrix dimension {N} is not a power of 2")

    # HHL simulates e^{iAt}, which is unitary only for Hermitian A. Until now the
    # check happened by accident and only on part of the input space: at N == 2
    # the ZYZ decomposition inside ``_controlled_1q_unitary`` can absorb a
    # non-Hermitian 2x2 (A = [[1,2],[0,3]] returned success_probability 0.238066
    # with |<A^-1 b|x>| = 1.000000), while the same matrix at N == 4 raised
    # "Matrix is not unitary (U @ U^dagger != I)" from ``Circuit.unitary``.
    # Enforce it once, up front, for every N. (xq5/V-204)
    if not np.allclose(A, A.conj().T):
        skew = float(np.max(np.abs(A - A.conj().T)))
        raise ValueError(
            f"A is not Hermitian: max|A - A^dagger| = {skew:.6g}, so exp(iAt) is "
            "not unitary and the eigenvalue read-out is not well defined. Embed "
            "the system in the Hermitian dilation [[0, A], [A^dagger, 0]] "
            "(2N x 2N) and solve that instead."
        )

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

    # The rotation has to be conditioned on the *whole* counting register.
    # The old loop fired RY(angle/2) / CX(j) / RY(-angle/2) / CX(j) once per
    # counting qubit, which only ever fires on single-bit register values
    # (j = 1, 2, 4) and does nothing on |3>, |5>, |6>, |7> -- yet those bins carry
    # essentially all of the weight once the phase is not a dyadic fraction.
    # (ALG-8)
    n_states = 1 << t
    inversion = np.eye(2 * n_states, dtype=complex)
    for m in range(n_states):
        j = _bit_reverse(m, t)
        if j == 0:
            continue  # lambda_0 = 0: leave the ancilla untouched
        lambda_j = 2 * np.pi * j / (n_states * t_scale)
        ratio = 1.0 if lambda_j <= 0 else min(C / lambda_j, 1.0)
        angle = 2 * np.arcsin(ratio)
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        block = np.array(
            [[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex
        )
        idx = [m, n_states + m]
        inversion[np.ix_(idx, idx)] = block
    # ``unitary(M, qs)`` maps M-index bit r onto qs[k-1-r], so listing the
    # ancilla first makes it the most significant bit and bit r = t-1-i of the
    # rest carries counting qubit i -- hence the bit reversal above.
    circuit.unitary(inversion, [ancilla_idx] + counting_qubits)

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


def _bit_reverse(value: int, width: int) -> int:
    """Reverse the low ``width`` bits of ``value`` (e.g. 0b011 -> 0b110)."""
    if width <= 0:
        return 0
    return int(format(value, f"0{width}b")[::-1], 2)


def _encode_state(circuit: sf.Circuit, state: np.ndarray, qubits: List[int]):
    """Encode a normalized state vector into qubits.

    ``N == 2`` keeps the exact ``Ry``/``Rz`` angle decomposition. Any other
    ``N`` uses a single opaque ``unitary`` built by completing ``state`` to a
    full unitary, which is exact for every amplitude distribution.
    """
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
        # General N: complete |state> into a unitary whose first column *is*
        # |state>, then let ``unitary`` prepare it in one shot. The previous
        # body flipped X gates for every non-zero amplitude, which prepares an
        # equal superposition of those indices and cannot encode amplitudes or
        # phases at all -- so hhl_solve() could never be tested on N > 2. (ALG-8)
        v = np.asarray(state, dtype=complex).flatten()
        nrm_v = np.linalg.norm(v)
        if nrm_v < 1e-15:
            raise ValueError("cannot encode the zero state vector")
        v = v / nrm_v
        if N != 2 ** n:
            raise ValueError(
                f"state length {N} does not match {n} target qubit(s)"
            )
        # Pick the largest-magnitude component p and fill the remaining columns
        # with e_k for k != p. {v} + {e_k : k != p} is then always linearly
        # independent (v_p != 0 forbids the only possible dependence), so the
        # QR below is full rank and its first column is parallel to v.
        p = int(np.argmax(np.abs(v)))
        eye = np.eye(N, dtype=complex)
        cols = [v] + [eye[:, k] for k in range(N) if k != p]
        prep, _ = np.linalg.qr(np.column_stack(cols))
        phase = np.vdot(prep[:, 0], v)  # |phase| == 1 for a normalised v
        prep = prep * phase
        # ``unitary(M, qs)`` maps M-index bit r onto qs[k-1-r], so passing the
        # data qubits in reverse makes the M basis index equal to ``data_val``.
        if qubits:
            circuit.unitary(prep, list(reversed(qubits)))
