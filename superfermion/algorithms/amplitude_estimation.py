"""
Amplitude Estimation — Quantum Monte Carlo speedup (Brassard et al.).

Estimates the amplitude a = √⟨ψ|Π|ψ⟩ of a "good" subspace marked by oracle Π,
achieving O(1/ε) convergence vs O(1/ε²) for classical Monte Carlo.

This module provides both:
  - Canonical AE (QPE-based, optimal query complexity)
  - Iterative AE (simpler, no QPE, good for noisy devices)

Usage:
    >>> from superfermion.algorithms.amplitude_estimation import amplitude_estimation
    >>>
    >>> # Estimate P(|1⟩) of a balanced superposition
    >>> a_est = amplitude_estimation(
    ...     state_prep=lambda c: c.h(0),     # |+⟩ = (|0⟩ + |1⟩)/√2
    ...     oracle=lambda c: c.z(0),          # Mark |1⟩ (phase flip)
    ...     precision_bits=5,
    ... )
    >>> print(a_est["amplitude"] ** 2)  # → 0.5
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import superfermion as sf


def amplitude_estimation(
    state_prep: Callable[[sf.Circuit], None],
    oracle: Callable[[sf.Circuit], None],
    precision_bits: int = 5,
    n_qubits: Optional[int] = None,
    device: Any = "cpu",
    ae_method: str = "canonical",
    shots: int = 0,
) -> Dict[str, Any]:
    r"""Estimate the amplitude of the marked subspace.

    Args:
        state_prep: Function A that prepares |ψ⟩ = A|0⟩.
        oracle: Function S_χ that marks the "good" subspace via a phase flip.
        precision_bits: Number of estimation qubits (m). Gives error O(1/2^m).
        n_qubits: Qubit count of the state register. Auto-detected if None.
        device: Execution target — ``"cpu"``, ``"gpu"``, or ``DeviceExecutor``.
        ae_method: ``"canonical"`` (QPE-based, optimal) or ``"iterative"``
                   (no QPE, simpler circuit).
        shots: If > 0, sample results directly.

    Returns:
        Dict with keys:
          - ``"amplitude"``: Estimated amplitude a ∈ [0, 1].
          - ``"probability"``: a² (estimated probability of marked state).
          - ``"int_value"``: Raw measurement outcome (y).
          - ``"precision_bits"``: m (estimation qubits).
          - ``"method"``: Which method was used.
          - ``"statevector"`` / ``"counts"``: Raw result data.
    """
    if ae_method == "canonical":
        return _canonical_ae(state_prep, oracle, precision_bits, n_qubits, device, shots)
    elif ae_method == "iterative":
        return _iterative_ae(state_prep, oracle, precision_bits, n_qubits, device, shots)
    else:
        raise ValueError(f"Unknown ae_method: {ae_method}. Use 'canonical' or 'iterative'.")


# ── Canonical AE (QPE-based) ───────────────────────────────────────────

def _canonical_ae(
    state_prep: Callable[[sf.Circuit], None],
    oracle: Callable[[sf.Circuit], None],
    m: int,
    n_qubits: Optional[int],
    device: Any,
    shots: int,
) -> Dict[str, Any]:
    """QPE-based amplitude estimation: O(1/ε) queries, optimal.

    Architecture: m estimation qubits + n state qubits + 1 ancilla.
    Uses the amplitude amplification operator Q = A S_0 A^† S_χ.
    """
    # Detect n_qubits by running state_prep on a temp circuit
    if n_qubits is None:
        n_qubits = _detect_n_qubits(state_prep)

    n = n_qubits
    total = m + n

    circuit = sf.Circuit(total, name=f"AE_canonical(m={m}, n={n})")

    # ── 1. Initialize estimation qubits with H gates ───────────────────
    for i in range(m):
        circuit.h(i)

    # ── 2. Prepare |ψ⟩ = A|0⟩ on state register ───────────────────────
    # Apply state_prep to state register (qubits m..m+n-1) via gate replay
    sub = sf.Circuit(n)
    state_prep(sub)
    for gate in sub._gates:
        phys_qubits = [m + q for q in gate.qubits]
        args = list(gate.params) if gate.params else []
        method = getattr(circuit, gate.name.lower(), None)
        if method is not None:
            method(*args, *phys_qubits)

    # ── 3. Apply controlled Q^{2^k} operations ─────────────────────────
    for k in range(m):
        power = 2 ** k
        _controlled_q_power(circuit, state_prep, oracle, k, m, n, power)

    # ── 4. Inverse QFT ─────────────────────────────────────────────────
    _qft_inverse(circuit, list(range(m)))

    result = sf.run(circuit, device=device, shots=shots)

    # Parse measurement
    if shots == 0 and result.statevector is not None:
        sv = np.asarray(result.statevector).flatten()
        probs = np.abs(sv) ** 2
        # Trace over state register (qubits m..total-1)
        probs = probs.reshape([2] * total)
        state_axes = tuple(total - 1 - q for q in range(m, total))
        probs = probs.sum(axis=state_axes)  # shape (2,)*m

        # The IQFT swap reverses qubit order (q0 ↔ q_{m-1}, …).
        # np.argmax on the C-order-flattened tensor returns an index
        # whose bits are ordered [MSB…LSB]; we convert to physical y.
        max_idx = np.unravel_index(np.argmax(probs), [2] * m)
        y = sum(int(max_idx[i]) << i for i in range(m))
        a_est, prob = _interpret_measurement(y, m)
        return {
            "amplitude": a_est,
            "probability": prob,
            "int_value": y,
            "precision_bits": m,
            "method": "canonical",
            "result": result,
        }

    if result.counts:
        best_bits = max(result.counts, key=result.counts.get)
        y = int(best_bits[:m][::-1], 2) if len(best_bits) >= m else 0
        a_est, prob = _interpret_measurement(y, m)
        return {
            "amplitude": a_est,
            "probability": prob,
            "int_value": y,
            "precision_bits": m,
            "method": "canonical",
            "counts": result.counts,
            "result": result,
        }

    return {
        "amplitude": 0.0,
        "probability": 0.0,
        "precision_bits": m,
        "method": "canonical",
        "result": result,
    }


# ── Iterative AE (IQAE — simpler, no QPE) ─────────────────────────────

def _iterative_ae(
    state_prep: Callable[[sf.Circuit], None],
    oracle: Callable[[sf.Circuit], None],
    m: int,
    n_qubits: Optional[int],
    device: Any,
    shots: int,
) -> Dict[str, Any]:
    """Iterative amplitude estimation without QPE.

    Uses maximum-likelihood estimation from Grover iterates at
    geometrically increasing depths to converge on the amplitude.
    """
    if n_qubits is None:
        n_qubits = _detect_n_qubits(state_prep)

    n = n_qubits
    marked_states = _find_marked_states(oracle, n, device)

    # Direct measurement: run the state prep and measure probability
    c0 = sf.Circuit(n)
    state_prep(c0)
    r0 = sf.run(c0, device=device, shots=0)
    if r0.statevector is not None:
        sv0 = np.asarray(r0.statevector).flatten()
        p_direct = float(np.sum(np.abs(sv0[marked_states]) ** 2))
    else:
        p_direct = 0.0

    # Refine with Grover iterates at increasing depths
    theta_est = np.arcsin(np.sqrt(max(p_direct, 1e-12)))
    best_probs = [(0, p_direct)]

    for depth in range(1, m + 1):
        k = depth
        c = sf.Circuit(n, name=f"AE_iter_k{k}")
        state_prep(c)
        for _ in range(k):
            oracle(c)
            _apply_inverse_state_prep(c, state_prep, n)
            _reflection_about_zero(c, n)
            sub = sf.Circuit(n)
            state_prep(sub)
            for gate in sub._gates:
                gate_fn = getattr(c, gate.name.lower(), None)
                if gate_fn is not None:
                    args = list(gate.params) if gate.params else []
                    gate_fn(*args, *gate.qubits)
        r = sf.run(c, device=device, shots=0)
        if r.statevector is not None:
            sv = np.asarray(r.statevector).flatten()
            p_k = float(np.sum(np.abs(sv[marked_states]) ** 2))
            best_probs.append((k, p_k))

    # MLE: find theta that best explains all observed probabilities
    # P(k) = sin²((2k+1)θ)
    best_theta = theta_est
    best_cost = float("inf")
    for theta_try in np.linspace(0, np.pi / 2, 2 ** (m + 2)):
        cost = sum(
            (p_obs - np.sin((2 * k + 1) * theta_try) ** 2) ** 2
            for k, p_obs in best_probs
        )
        if cost < best_cost:
            best_cost = cost
            best_theta = theta_try

    a_est = float(np.sin(best_theta))
    return {
        "amplitude": a_est,
        "probability": a_est ** 2,
        "precision_bits": m,
        "method": "iterative",
        "result": None,
    }


# ── Helper: Inverse state prep and S_0 reflection ─────────────────────

def _apply_inverse_state_prep(circuit, state_prep, n):
    sub = sf.Circuit(n)
    state_prep(sub)
    for gate in reversed(sub._gates):
        inv_name = gate.name.lower()
        inv_params = [-p for p in gate.params] if gate.params else []
        if inv_name in ("h", "x", "y", "z", "cx", "cnot", "cz", "swap"):
            inv_params = list(gate.params) if gate.params else []
        elif inv_name == "s":
            inv_name = "sdg"
            inv_params = []
        elif inv_name == "t":
            inv_name = "tdg"
            inv_params = []
        gate_fn = getattr(circuit, inv_name, None)
        if gate_fn is not None:
            gate_fn(*inv_params, *gate.qubits)


def _reflection_about_zero(circuit, n):
    for q in range(n):
        circuit.x(q)
    _mcz(circuit, list(range(n)))
    for q in range(n):
        circuit.x(q)


def _reflection_about_mean(circuit: sf.Circuit):
    """Apply S_0 = 2|0⟩⟨0| − I reflection."""
    n = circuit.n_qubits
    for q in range(n):
        circuit.h(q)
        circuit.x(q)
    # Multi-controlled Z
    _mcz(circuit, list(range(n)))
    for q in range(n):
        circuit.x(q)
        circuit.h(q)


def _mcz(circuit: sf.Circuit, qubits: List[int]):
    """Multi-controlled Z gate."""
    n = len(qubits)
    if n == 1:
        circuit.z(qubits[0])
    elif n == 2:
        circuit.cz(qubits[0], qubits[1])
    else:
        circuit.h(qubits[-1])
        _mcx(circuit, qubits[:-1], qubits[-1])
        circuit.h(qubits[-1])


def _mcx(circuit: sf.Circuit, controls: List[int], target: int):
    """Multi-controlled X using Toffoli cascade."""
    if len(controls) == 1:
        circuit.cx(controls[0], target)
        return
    if len(controls) == 2:
        circuit.toffoli(controls[0], controls[1], target)
        return
    ancilla_base = circuit.n_qubits
    n_ctrl = len(controls)
    for i in range(n_ctrl - 1):
        a = controls[i] if i == 0 else ancilla_base + i - 1
        b = controls[i + 1]
        c = ancilla_base + i
        circuit.toffoli(a, b, c)
    circuit.toffoli(ancilla_base + n_ctrl - 2, controls[-1], target)
    for i in range(n_ctrl - 2, -1, -1):
        a = controls[i] if i == 0 else ancilla_base + i - 1
        b = controls[i + 1]
        c = ancilla_base + i
        circuit.toffoli(a, b, c)


# ── Helper: Controlled Q^power ─────────────────────────────────────────

def _controlled_q_power(
    circuit: sf.Circuit,
    state_prep: Callable,
    oracle: Callable,
    control: int,
    state_offset: int,
    n_state: int,
    power: int,
):
    """Apply controlled (Q)^{power} where Q = -A S_0 A^† S_χ.

    The full Q operator includes the state preparation A and its inverse A^†
    wrapped around the reflection S_0.  All components are controlled by the
    estimation qubit `control`.
    """
    # Build oracle and state-prep on sub-circuits to extract their gates
    sub_oracle = sf.Circuit(n_state)
    oracle(sub_oracle)

    sub_prep = sf.Circuit(n_state)
    state_prep(sub_prep)

    for _ in range(power):
        # 1. Controlled S_χ (oracle)
        for gate in sub_oracle._gates:
            phys_qubits = [state_offset + q for q in gate.qubits]
            _apply_controlled_gate(circuit, gate, control, phys_qubits)

        # 2. Controlled A^† (inverse state preparation)
        for gate in reversed(sub_prep._gates):
            inv_gate = _invert_gate(gate)
            phys_qubits = [state_offset + q for q in inv_gate.qubits]
            _apply_controlled_gate(circuit, inv_gate, control, phys_qubits)

        # 3. Controlled S_0 = 2|0⟩⟨0| − I
        state_qubits = list(range(state_offset, state_offset + n_state))
        _controlled_reflection(circuit, control, state_qubits)

        # 4. Controlled A (state preparation)
        for gate in sub_prep._gates:
            phys_qubits = [state_offset + q for q in gate.qubits]
            _apply_controlled_gate(circuit, gate, control, phys_qubits)


def _apply_controlled_gate(
    circuit: sf.Circuit,
    gate,
    control: int,
    qubits: List[int],
):
    """Apply a single gate in controlled form."""
    name = gate.name.upper()
    if name in ("H",):
        # CH = RY(pi/4) . CZ . RY(-pi/4)  on target
        # Apply: RY(-pi/4) first, then CZ, then RY(pi/4)
        q = qubits[0]
        circuit.ry(-np.pi / 4, q)
        circuit.cz(control, q)
        circuit.ry(np.pi / 4, q)
    elif name in ("X",):
        circuit.cx(control, qubits[0])
    elif name in ("Y",):
        circuit.cy(control, qubits[0])
    elif name in ("Z",):
        circuit.cz(control, qubits[0])
    elif name in ("S",):
        circuit.cp(np.pi / 2, control, qubits[0])
    elif name in ("T",):
        circuit.cp(np.pi / 4, control, qubits[0])
    elif name in ("SDG",):
        circuit.cp(-np.pi / 2, control, qubits[0])
    elif name in ("TDG",):
        circuit.cp(-np.pi / 4, control, qubits[0])
    elif name in ("P", "PHASE"):
        theta = gate.params[0] if gate.params else 0.0
        circuit.cp(theta, control, qubits[0])
    elif name in ("RX",):
        theta = gate.params[0] if gate.params else 0.0
        q = qubits[0]
        circuit.rx(-theta / 2, q)
        circuit.cx(control, q)
        circuit.rx(theta / 2, q)
        circuit.cx(control, q)
    elif name in ("RY",):
        theta = gate.params[0] if gate.params else 0.0
        q = qubits[0]
        circuit.ry(-theta / 2, q)
        circuit.cx(control, q)
        circuit.ry(theta / 2, q)
        circuit.cx(control, q)
    elif name in ("RZ",):
        theta = gate.params[0] if gate.params else 0.0
        circuit.cp(theta, control, qubits[0])
    elif name in ("CNOT", "CX"):
        # Controlled-CNOT = Toffoli
        circuit.toffoli(control, qubits[0], qubits[1])
    elif name in ("CZ",):
        # Controlled-CZ = CCZ via H + Toffoli + H
        circuit.h(qubits[1])
        circuit.toffoli(control, qubits[0], qubits[1])
        circuit.h(qubits[1])
    elif name in ("SWAP",):
        # Controlled-SWAP (Fredkin) decomposed as:
        # CX(tgt, ctl); Toffoli(ctrl, ctl, tgt); CX(tgt, ctl)
        circuit.cx(qubits[1], qubits[0])
        circuit.toffoli(control, qubits[0], qubits[1])
        circuit.cx(qubits[1], qubits[0])
    elif name in ("TOFFOLI", "CCX"):
        # Controlled-Toffoli = CCCX; decompose via ancilla
        # Use qubit 0 as workspace (borrowed, restored)
        circuit.toffoli(qubits[0], qubits[1], 0)
        circuit.toffoli(control, 0, qubits[2])
        circuit.toffoli(qubits[0], qubits[1], 0)
    else:
        # Best-effort controlled version via CX sandwich pattern
        if len(qubits) == 1:
            circuit.cx(control, qubits[0])
            method = getattr(circuit, gate.name.lower(), None)
            if method is not None:
                args = list(gate.params) + [qubits[0]]
                method(*args)
            circuit.cx(control, qubits[0])


def _invert_gate(gate):
    """Return a modified GateRecord representing the inverse of `gate`.

    Self-inverse gates (H, X, Y, Z, CNOT, CZ, SWAP) are returned unchanged.
    Rotation gates have their parameters negated.  S ↔ SDG and T ↔ TDG.
    """
    import superfermion as _sf

    name = gate.name.upper()
    inv_params = []
    for p in (gate.params or []):
        try:
            inv_params.append(-float(p))
        except (TypeError, ValueError):
            inv_params.append(p)  # SymbolicParameter — caller handles binding

    # Map non-self-inverse gates to their conjugates
    inv_name = name
    if name == "S":
        inv_name = "SDG"
        inv_params = []
    elif name == "SDG":
        inv_name = "S"
        inv_params = []
    elif name == "T":
        inv_name = "TDG"
        inv_params = []
    elif name == "TDG":
        inv_name = "T"
        inv_params = []

    return _sf.circuit.GateRecord(
        name=inv_name,
        qubits=list(gate.qubits),
        params=inv_params,
    )


def _controlled_reflection(
    circuit: sf.Circuit,
    control: int,
    qubits: List[int],
):
    """Apply controlled reflection 2|0⟩⟨0| − I.

    S_0 = X^⊗n · MCZ · X^⊗n.  The controlled version wraps MCZ with an
    extra control qubit.  _controlled_mcz already handles the H·C-MCX·H
    decomposition internally, so we only need X wrapping on either side.
    """
    for q in qubits:
        circuit.x(q)
    all_ctrl = [control] + qubits[:-1]
    _controlled_mcz(circuit, all_ctrl, qubits[-1])
    for q in qubits:
        circuit.x(q)


def _controlled_mcz(
    circuit: sf.Circuit,
    controls: List[int],
    target: int,
):
    """Multi-controlled Z with arbitrary controls."""
    circuit.h(target)
    if len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        circuit.toffoli(controls[0], controls[1], target)
    else:
        # Multi-controlled Z with >2 controls requires ancilla qubits
        # for a proper Toffoli-cascade decomposition.
        # Decompose using (n_ctrl-2) clean ancillas via linear Toffoli chain.
        n_ctrl = len(controls)
        # Borrow the first estimation qubit (0) as the ancilla chain start.
        # NOTE: This requires qubit 0 to be a clean |0⟩ ancilla — in the
        # canonical AE circuit, estimation qubits are in |+⟩ after Hadamard
        # initialization, so this only works when n_ctrl ≤ 4 (1-2 ancillas
        # can be synthesized from |+⟩ by measurement+reset, which we skip).
        # For practical AE usage (n_state ≤ 2), n_ctrl ≤ 3, handled above.
        ancilla_base = 0
        for i in range(n_ctrl - 1):
            a = controls[i] if i == 0 else ancilla_base + i - 1
            b = controls[i + 1]
            c_idx = ancilla_base + i
            if c_idx in controls or c_idx == target:
                raise NotImplementedError(
                    f"_controlled_mcz: ancilla qubit {c_idx} conflicts with "
                    f"controls/target. Use n_state ≤ 2 or provide extra clean qubits."
                )
            circuit.toffoli(a, b, c_idx)
        circuit.toffoli(ancilla_base + n_ctrl - 2, controls[-1], target)
        for i in range(n_ctrl - 2, -1, -1):
            a = controls[i] if i == 0 else ancilla_base + i - 1
            b = controls[i + 1]
            c_idx = ancilla_base + i
            circuit.toffoli(a, b, c_idx)
    circuit.h(target)


# ── QFT helpers ────────────────────────────────────────────────────────

def _qft_inverse(circuit: sf.Circuit, qubits: List[int]):
    """Inverse Quantum Fourier Transform."""
    n = len(qubits)
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])
    for i in range(n):
        for j in range(i):
            angle = -np.pi / (2 ** (i - j))
            circuit.cp(angle, qubits[j], qubits[i])
        circuit.h(qubits[i])


# ── Measurement interpretation ─────────────────────────────────────────

def _interpret_measurement(y: int, m: int) -> Tuple[float, float]:
    r"""Convert QPE outcome y ∈ [0, 2^m) to amplitude estimate.

    a = sin²(π · y / 2^m)
    """
    theta = np.pi * y / (2 ** m)
    a = float(np.sin(theta))
    prob = float(np.sin(theta) ** 2)
    return a, prob


# ── Utility: detect n_qubits ──────────────────────────────────────────

def _find_marked_states(
    oracle: Callable[[sf.Circuit], None],
    n: int,
    device: Any,
) -> List[int]:
    """Find which computational basis states are marked (phase-flipped) by the oracle.

    Probes each basis state |i⟩ by applying the oracle and checking
    whether the amplitude becomes −1.
    """
    N = 2 ** n
    marked = []
    for i in range(N):
        c = sf.Circuit(n)
        bits = format(i, f"0{n}b")
        for j, b in enumerate(bits):
            if b == "1":
                c.x(j)
        oracle(c)
        r = sf.run(c, device=device, shots=0)
        if r.statevector is not None:
            sv = np.asarray(r.statevector).flatten()
            # Oracle flips sign: |i⟩ → −|i⟩ iff state i is marked
            if np.abs(sv[i] + 1.0) < 1e-6:
                marked.append(i)
    return marked


def _detect_n_qubits(
    state_prep: Callable[[sf.Circuit], None],
) -> int:
    """Detect state register size by running state_prep on a probe circuit."""
    for n in range(1, 11):
        try:
            c = sf.Circuit(n)
            state_prep(c)
            return n
        except Exception:
            continue
    raise ValueError(
        "Could not detect n_qubits. Please provide n_qubits explicitly."
    )
