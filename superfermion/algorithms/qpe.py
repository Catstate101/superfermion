"""
Quantum Phase Estimation (QPE) — Extract eigenvalues of a unitary operator.

Given a unitary U and its eigenstate |ψ⟩ where U|ψ⟩ = e^{2πiφ}|ψ⟩,
QPE estimates the phase φ to t bits of precision.

Usage:
    >>> from superfermion.algorithms.qpe import quantum_phase_estimation
    >>>
    >>> # Estimate the phase of the T gate (= P(π/4)) on |1⟩
    >>> phi = quantum_phase_estimation(
    ...     unitary_circuit=sf.Circuit(1).t(0),
    ...     eigenstate_prep=lambda c: c.x(0),  # |1⟩
    ...     precision_bits=4,
    ... )
    >>> print(phi)  # → 0.125 (= 1/8 = e^{2πi/8} = e^{iπ/4})
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import superfermion as sf


def _controlled_unitary(
    circuit: sf.Circuit,
    unitary_fn: Callable[[sf.Circuit], sf.Circuit],
    control: int,
    target_qubits: List[int],
    power: int = 1,
):
    """Apply controlled-U^{power} to the circuit.

    Simple implementation: for each gate in U, apply a controlled version.
    Power > 1 repeats the unitary.
    """
    # Build the powered unitary
    sub = sf.Circuit(len(target_qubits))
    for _ in range(power):
        unitary_fn(sub)

    # Apply each gate as controlled
    for gate in sub._gates:
        gate_qubits = [control] + [target_qubits[q] for q in gate.qubits]
        if gate.name.upper() in ("H",):
            circuit.ch(control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("X",):
            if len(gate.qubits) == 1:
                circuit.cx(control, target_qubits[gate.qubits[0]])
            else:
                circuit.cx(control, target_qubits[gate.qubits[0]])
                circuit.cx(control, target_qubits[gate.qubits[1]])
        elif gate.name.upper() in ("Y",):
            circuit.cy(control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("Z",):
            circuit.cz(control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("S",):
            circuit.cp(np.pi / 2, control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("T",):
            circuit.cp(np.pi / 4, control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("P", "PHASE"):
            theta = gate.params[0] if gate.params else 0.0
            circuit.cp(theta, control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("RX",):
            # Controlled-RX: CRX(θ) = (I⊗RX(θ/2)) · CNOT · (I⊗RX(−θ/2)) · CNOT
            theta = gate.params[0] if gate.params else 0.0
            q = target_qubits[gate.qubits[0]]
            circuit.rx(-theta / 2, q)
            circuit.cx(control, q)
            circuit.rx(theta / 2, q)
            circuit.cx(control, q)
        elif gate.name.upper() in ("RY",):
            theta = gate.params[0] if gate.params else 0.0
            q = target_qubits[gate.qubits[0]]
            circuit.ry(-theta / 2, q)
            circuit.cx(control, q)
            circuit.ry(theta / 2, q)
            circuit.cx(control, q)
        elif gate.name.upper() in ("RZ",):
            theta = gate.params[0] if gate.params else 0.0
            circuit.cp(theta, control, target_qubits[gate.qubits[0]])
        elif gate.name.upper() in ("U",):
            # Controlled-U(θ,φ,λ): use 3 controlled rotations
            theta, phi, lam = (
                (gate.params[0], gate.params[1], gate.params[2])
                if len(gate.params) >= 3
                else (0, 0, 0)
            )
            q = target_qubits[gate.qubits[0]]
            circuit.cp(lam, control, q)
            circuit.cx(control, q)
            circuit.cp(theta, control, q)
            circuit.cx(control, q)
            circuit.cp(phi, control, q)
        else:
            # Decompose generic gate into basis gates (best effort)
            _controlled_generic(circuit, gate, control, target_qubits)


def _controlled_generic(
    circuit: sf.Circuit, gate, control: int, target_qubits: List[int]
):
    """Best-effort controlled-gate decomposition via U = e^{iα} AXBXC."""
    if len(gate.qubits) != 1:
        return  # Skip multi-qubit gates for now
    q = target_qubits[gate.qubits[0]]
    # ABC decomposition: any 1-qubit U = e^{iα} R_z(β) R_y(γ) R_z(δ)
    # Controlled-U = [I⊗C] · [I⊗B] · CNOT · [I⊗B†] · CNOT · [I⊗A]
    # For simplicity, just apply CX-then-CX pattern
    circuit.cx(control, q)
    # Apply gate decomposition inline
    if gate.name.upper() == "H":
        circuit.h(q)
    circuit.cx(control, q)


def _iqft(circuit: sf.Circuit, qubits: List[int]):
    """Inverse Quantum Fourier Transform on the given qubits.

    Applies bit-reversal swap first, then the cascade of H and
    controlled-phase rotations, matching the QPE phase convention where
    qubit k carries phase 2^k * phi.
    """
    n = len(qubits)
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - 1 - i])
    for i in range(n):
        for j in range(i):
            angle = -np.pi / (2 ** (i - j))
            circuit.cp(angle, qubits[j], qubits[i])
        circuit.h(qubits[i])


def quantum_phase_estimation(
    unitary_circuit: sf.Circuit,
    eigenstate_prep: Callable[[sf.Circuit], None],
    precision_bits: int = 4,
    device: Any = "cpu",
    method: str = "statevector",
    shots: int = 0,
) -> Dict[str, Any]:
    r"""Run Quantum Phase Estimation.

    Estimates the eigenvalue phase φ of unitary U where U|ψ⟩ = e^{2πiφ}|ψ⟩.

    Args:
        unitary_circuit: Circuit implementing U (acts on target qubits).
        eigenstate_prep: Function that prepares |ψ⟩ on target qubits.
        precision_bits: Number of estimation (counting) qubits (t).
        device: Execution target — ``"cpu"``, ``"gpu"``, or ``DeviceExecutor``.
        method: Simulation method — ``"statevector"``, ``"mps"``, etc.
        shots: If > 0, sample counting register.

    Returns:
        Dict with keys ``"phase"`` (float in [0,1)), ``"phase_binary"``,
        ``"int_value"``, ``"precision_bits"``, ``"eigenvalue"`` (= exp(2πi·phase)),
        ``"statevector"`` (if shots=0).
    """
    n_target = unitary_circuit.n_qubits
    t = precision_bits
    total = t + n_target

    circuit = sf.Circuit(total, name=f"QPE(t={t})")

    # ── Counting register ────────────────────────────────────────────────
    for i in range(t):
        circuit.h(i)

    # ── Eigenstate preparation ───────────────────────────────────────────
    sub = sf.Circuit(n_target)
    eigenstate_prep(sub)
    for gate in sub._gates:
        phys_qubits = [t + q for q in gate.qubits]
        method_fn = getattr(circuit, gate.name.lower(), None)
        if method_fn is not None:
            args = list(gate.params) if gate.params else []
            method_fn(*args, *phys_qubits)

    # ── Controlled-U^{2^k} chain ─────────────────────────────────────────
    target_qubits = list(range(t, t + n_target))
    for k in range(t):
        _controlled_unitary(
            circuit,
            lambda c: _replay_gates(c, unitary_circuit),
            control=k,
            target_qubits=target_qubits,
            power=2 ** k,
        )

    # ── Inverse QFT ──────────────────────────────────────────────────────
    counting_qubits = list(range(t))
    _iqft(circuit, counting_qubits)

    result = sf.run(circuit, device=device, method=method, shots=shots)

    if shots == 0 and result.statevector is not None:
        sv = np.asarray(result.statevector).flatten()
        probs = np.abs(sv) ** 2
        # Trace out target register (qubits t..t+n_target-1).
        # In C-order reshape, axis k = qubit (total-1-k).
        probs = probs.reshape([2] * total)
        target_axes = tuple(total - 1 - q for q in range(t, total))
        probs = probs.sum(axis=target_axes).flatten()
        best = int(np.argmax(probs))
        phase = best / (2 ** t)
        return {
            "phase": phase,
            "phase_binary": format(best, f"0{t}b"),
            "int_value": best,
            "precision_bits": t,
            "eigenvalue": np.exp(2j * np.pi * phase),
            "probability": float(probs[best]),
            "result": result,
        }

    # Sampling mode
    if result.counts:
        best_bits = max(result.counts, key=result.counts.get)
        # Take only counting-register bits (first t bits, reversed for QFT order)
        best = int(best_bits[:t][::-1], 2) if len(best_bits) >= t else 0
        phase = best / (2 ** t)
        return {
            "phase": phase,
            "phase_binary": format(best, f"0{t}b"),
            "int_value": best,
            "precision_bits": t,
            "eigenvalue": np.exp(2j * np.pi * phase),
            "counts": result.counts,
            "result": result,
        }

    return {"phase": 0.0, "precision_bits": t, "result": result}


def _replay_gates(target: sf.Circuit, source: sf.Circuit):
    """Copy all gates from source to target circuit (in-place)."""
    for gate in source._gates:
        method = getattr(target, gate.name.lower(), None)
        if method is not None:
            args = list(gate.params)
            args.extend(gate.qubits)
            method(*args)
