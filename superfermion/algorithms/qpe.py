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


def _param(gate, index: int, default: float = 0.0):
    """Read ``gate.params[index]`` without assuming the list is long enough."""
    try:
        return gate.params[index]
    except (IndexError, TypeError):
        return default


def _controlled_single(circuit, gate, control: int, q: int) -> bool:
    """Emit the controlled version of a **single-qubit** gate.

    Returns ``True`` when the gate was handled here and ``False`` when the
    caller should fall back to :func:`_controlled_generic`.  Every entry uses
    a real primitive rather than a hand-rolled decomposition, because the
    decompositions that used to live here were wrong:

    * controlled-RX was written ``RX(−θ/2), CX, RX(θ/2), CX``, which is the
      wrong sign order and produced the identity for the QPE input;
    * controlled-RZ was emitted as ``CP(θ)``, but ``CP(θ) = diag(1,1,1,e^{iθ})``
      while ``CRZ(θ) = diag(1,1,e^{−iθ/2},e^{iθ/2})`` — they differ by a phase
      on the control branch, so the QPE phase came out 8 bins low;
    * controlled-U used a ``CP/CX/CP/CX/CP`` cascade that implements no
      single-qubit unitary at all.

    ``Circuit`` has ``crx``, ``cry``, ``crz``, ``cu``, ``cp`` and ``ch``
    natively, and each has been checked against the analytic matrix on
    ``to_unitary()``, so the exact primitives are used instead.
    """
    name = gate.name.upper()
    if name == "H":
        circuit.ch(control, q)
    elif name in ("X", "NOT"):
        circuit.cx(control, q)
    elif name == "Y":
        circuit.cy(control, q)
    elif name == "Z":
        circuit.cz(control, q)
    elif name == "S":
        circuit.cp(np.pi / 2, control, q)
    elif name == "SDG":
        circuit.cp(-np.pi / 2, control, q)
    elif name == "T":
        circuit.cp(np.pi / 4, control, q)
    elif name == "TDG":
        circuit.cp(-np.pi / 4, control, q)
    elif name in ("P", "PHASE"):
        circuit.cp(_param(gate, 0), control, q)
    elif name in ("R1",):
        circuit.cp(_param(gate, 0), control, q)
    elif name in ("RX",):
        circuit.crx(_param(gate, 0), control, q)
    elif name in ("RY",):
        circuit.cry(_param(gate, 0), control, q)
    elif name in ("RZ",):
        circuit.crz(_param(gate, 0), control, q)
    elif name in ("U", "U3"):
        circuit.cu(
            _param(gate, 0), _param(gate, 1), _param(gate, 2), control, q
        )
    elif name in ("SX",):
        # SX = U3(−π/2, π/2, π/2); CU carries the phase correctly.
        circuit.cu(-np.pi / 2, np.pi / 2, np.pi / 2, control, q)
    else:
        return False
    return True


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
        name = gate.name.upper()

        if name == "UNITARY":
            _controlled_generic(circuit, gate, control, target_qubits)
            continue

        n_gate_qubits = len(gate.qubits)
        phys = [target_qubits[i] for i in gate.qubits]

        # ── single-qubit gates: exact primitives ─────────────────────────
        if n_gate_qubits == 1 and _controlled_single(circuit, gate, control, phys[0]):
            continue

        # ── two-qubit gates with an exact primitive ──────────────────────
        if n_gate_qubits == 2:
            q0, q1 = phys
            if name in ("CX", "CNOT"):
                circuit.ccx(control, q0, q1)
                continue
            if name == "CZ":
                circuit.h(q1)
                circuit.ccx(control, q0, q1)
                circuit.h(q1)
                continue
            if name == "SWAP":
                circuit.fredkin(control, q0, q1)
                continue

        # ── anything else: exact block-diagonal embedding ────────────────
        _controlled_generic(circuit, gate, control, target_qubits)


def _gate_matrix(gate) -> "np.ndarray":
    """Matrix of ``gate`` with **bit 0 = gate.qubits[0]** (little-endian).

    Rebuilds the gate on a small ``len(gate.qubits)``-qubit circuit so the
    result does not depend on where the gate sits in the caller's register.
    """
    k = len(gate.qubits)
    name = gate.name.upper()
    if name == "UNITARY":
        m = getattr(gate, "matrix", None)
        if m is None:
            raise ValueError("UNITARY gate carries no matrix")
        m = np.asarray(m, dtype=complex)
        if m.shape != (2 ** k, 2 ** k):
            raise ValueError(
                f"UNITARY matrix shape {m.shape} does not match "
                f"{k} qubit(s) (expected ({2 ** k}, {2 ** k}))"
            )
        return m
    mini = sf.Circuit(k)
    apply = getattr(mini, gate.name.lower(), None)
    if apply is None:
        raise ValueError(
            f"QPE cannot build a controlled version of gate {gate.name!r}: "
            f"Circuit has no method {gate.name.lower()!r}. "
            "Rebuild the unitary out of named gates, or pass it as a single "
            "circuit.unitary(...) gate (which is handled)."
        )
    apply(*list(gate.params), *range(k))
    return np.asarray(mini.to_unitary(), dtype=complex)


def _controlled_generic(
    circuit: sf.Circuit, gate, control: int, target_qubits: List[int]
):
    """Control an arbitrary (multi-qubit) gate by exact embedding.

    Emits a single ``UNITARY`` gate holding ``blockdiag(I, U)`` instead of
    dropping the gate.  ``Circuit.unitary(M, qubits)`` maps ``qubits[0]`` to
    the **most significant** index bit of ``M``, so the control is listed
    first and the targets follow in *reverse* order — that ordering is the
    one measured against ``ccx`` and ``cx``, not assumed.
    """
    k = len(gate.qubits)
    u = _gate_matrix(gate)
    embedded = np.eye(2 ** (k + 1), dtype=complex)
    embedded[2 ** k:, 2 ** k:] = u
    qubits = [control] + [target_qubits[q] for q in reversed(gate.qubits)]
    circuit.unitary(embedded, qubits)


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
        Dict. Every mode carries ``"phase"`` (float in [0, 1)), ``"phase_binary"``
        and ``"int_value"`` (the same bin, MSB-first / as an index),
        ``"precision_bits"``, ``"eigenvalue"`` (= exp(2πi·phase)) and
        ``"result"`` (the ``RunResult`` of the QPE circuit, kept so the raw
        statevector or counts are reachable as ``result.statevector`` /
        ``result.counts``).

        Additional keys by mode:

        * ``shots=0``: ``"probability"`` -- the population of the winning
          counting bin after tracing out the target register.
        * ``shots>0``: ``"counts"`` (the sampled distribution) and
          ``"counting_bits"`` (the counting-register suffix that was read).
        * when the run produced neither a statevector nor counts
          (``shots=0`` on a target without one): ``"phase"`` is ``0.0`` and the
          remaining keys above are absent -- the ``"result"`` is still there.
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
        # In a counts key the **leftmost character is qubit n-1**, so the
        # counting register (qubits 0..t-1, the least significant ones) is the
        # *tail* of the string. Inside that tail the leftmost character is
        # qubit t-1, so reading it left-to-right already gives the value
        # MSB-first — `int(tail, 2)` is the bin index with no reversal.
        # Reading the head instead (what this did before) parses the
        # target-register bits, which is why the sampled phase was one bin low
        # whenever the target ended in 1.
        if len(best_bits) >= t:
            counting_bits = best_bits[len(best_bits) - t:]
        else:
            counting_bits = best_bits.rjust(t, "0")
        best = int(counting_bits, 2)
        phase = best / (2 ** t)
        return {
            "phase": phase,
            "phase_binary": format(best, f"0{t}b"),
            "int_value": best,
            "precision_bits": t,
            "eigenvalue": np.exp(2j * np.pi * phase),
            "counts": result.counts,
            "counting_bits": counting_bits,
            "result": result,
        }

    return {"phase": 0.0, "precision_bits": t, "result": result}


def _replay_gates(target: sf.Circuit, source: sf.Circuit):
    """Copy all gates from source to target circuit (in-place).

    ``UNITARY`` gates carry their matrix on ``gate.matrix`` instead of in
    ``gate.params``, so a ``params + qubits`` replay would call
    ``unitary(q0, q1, matrix)`` and fail. They are handled explicitly.
    """
    for gate in source._gates:
        if gate.name.upper() == "UNITARY":
            matrix = getattr(gate, "matrix", None)
            if matrix is None:
                raise ValueError(
                    "UNITARY gate has no matrix; cannot replay it into the "
                    "QPE circuit"
                )
            target.unitary(matrix, list(gate.qubits))
            continue
        method = getattr(target, gate.name.lower(), None)
        if method is not None:
            args = list(gate.params)
            args.extend(gate.qubits)
            method(*args)
