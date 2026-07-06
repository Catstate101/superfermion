"""Seeded QASM 2.0 circuit generators.

One QASM string per circuit is the single source of truth. Each framework
loads it through its own QASM parser so there is no opportunity for
gate-list divergence between frameworks.

Gate vocabulary is constrained to the intersection of what all three
frameworks + every SuperFermion backend understand:
  h, x, y, z, s, t, rx, ry, rz, cx, cz, swap, ccx
(see superfermion/backends/singularity.py:45-47 for the SF-side floor.)
"""
from __future__ import annotations

import math
import random
from typing import List


def _header(n_qubits: int, n_cbits: int | None = None) -> List[str]:
    # Intentionally no creg line. The Qiskit runner adds its own 'meas'
    # register via measure_all(); having a second empty creg would cause
    # get_counts() to emit space-separated multi-register keys, breaking
    # bitstring comparisons across frameworks.
    _ = n_cbits
    return [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n_qubits}];",
    ]


def bell_qasm() -> str:
    """Standard 2-qubit Bell state H(0) CX(0,1)."""
    lines = _header(2)
    lines.append("h q[0];")
    lines.append("cx q[0],q[1];")
    return "\n".join(lines)


def ghz_qasm(n: int) -> str:
    """n-qubit GHZ: H(0) then CX(i, i+1) for i in [0, n-1)."""
    if n < 2:
        raise ValueError("GHZ requires n >= 2")
    lines = _header(n)
    lines.append("h q[0];")
    for i in range(n - 1):
        lines.append(f"cx q[{i}],q[{i + 1}];")
    return "\n".join(lines)


def random_u3cz_qasm(n: int, depth: int = 4, seed: int = 42) -> str:
    """Random hardware-efficient layers: per layer, RZ-RX-RZ on every qubit,
    then CZ on every adjacent pair (brick-wall). Uses RX/RZ/CZ which all
    frameworks + SF backends support natively.

    Single-qubit angles drawn from Unif(-pi, pi) with the given seed.
    """
    if n < 1 or depth < 1:
        raise ValueError("n and depth must be positive")
    rng = random.Random(seed)
    lines = _header(n)
    for layer in range(depth):
        for q in range(n):
            theta = rng.uniform(-math.pi, math.pi)
            phi = rng.uniform(-math.pi, math.pi)
            lam = rng.uniform(-math.pi, math.pi)
            lines.append(f"rz({theta}) q[{q}];")
            lines.append(f"rx({phi}) q[{q}];")
            lines.append(f"rz({lam}) q[{q}];")
        # Brick-wall CZ: alternate even-odd starting pairs by layer
        start = layer % 2
        for q in range(start, n - 1, 2):
            lines.append(f"cz q[{q}],q[{q + 1}];")
    return "\n".join(lines)


def qft_qasm(n: int) -> str:
    """Standard QFT without swaps (canonical gate sequence)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    lines = _header(n)
    for j in range(n):
        lines.append(f"h q[{j}];")
        for k in range(j + 1, n):
            angle = math.pi / (2 ** (k - j))
            # Controlled-phase via cx-rz-cx-rz decomposition would bloat;
            # use 'cp' which Qiskit understands and SF's QASM parser maps to 'p'.
            # Fall back to a 3-gate decomposition of controlled-phase so every
            # framework handles it identically: CX; RZ(-a/2); CX; RZ(a/2) on
            # target, plus RZ(a/2) on control up to a global phase.
            half = angle / 2.0
            # decomposition: Rz(half) on control, CX, Rz(-half) on target, CX, Rz(half) on target
            lines.append(f"rz({half}) q[{j}];")
            lines.append(f"cx q[{j}],q[{k}];")
            lines.append(f"rz({-half}) q[{k}];")
            lines.append(f"cx q[{j}],q[{k}];")
            lines.append(f"rz({half}) q[{k}];")
    # Reverse with swaps so amplitudes come out in standard order
    for i in range(n // 2):
        lines.append(f"swap q[{i}],q[{n - 1 - i}];")
    return "\n".join(lines)


def vqe_ansatz_qasm(n: int, layers: int = 2, seed: int = 43) -> str:
    """Hardware-efficient ansatz: (RY then RZ on every qubit, then linear
    CX entangler) * layers. Small-angle seeded params make the state
    non-trivial but still near-product — good for observable-expectation
    sanity checks.
    """
    if n < 2:
        raise ValueError("VQE needs n >= 2")
    rng = random.Random(seed)
    lines = _header(n)
    for _ in range(layers):
        for q in range(n):
            lines.append(f"ry({rng.uniform(-0.5, 0.5)}) q[{q}];")
            lines.append(f"rz({rng.uniform(-0.5, 0.5)}) q[{q}];")
        for q in range(n - 1):
            lines.append(f"cx q[{q}],q[{q + 1}];")
    return "\n".join(lines)


def all_circuits(n: int, *, include_vqe: bool = True, include_qft: bool = True) -> dict[str, str]:
    """Convenience: bundle all circuit families at a given qubit count."""
    out: dict[str, str] = {"ghz": ghz_qasm(n), "random": random_u3cz_qasm(n, depth=4, seed=42)}
    if n == 2:
        out["bell"] = bell_qasm()
    if include_qft and 2 <= n <= 20:
        out["qft"] = qft_qasm(n)
    if include_vqe and 2 <= n <= 16:
        out["vqe"] = vqe_ansatz_qasm(n, layers=2, seed=43)
    return out
