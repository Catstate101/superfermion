"""
QML Circuit Templates — drop-in replacements for PennyLane templates and Qiskit feature maps.

Templates return SF Circuit objects that can be run on any SF backend.

Cross-validated against:
  pennylane: qml.AngleEmbedding, qml.BasicEntanglerLayers, qml.StronglyEntanglingLayers
  qiskit: ZZFeatureMap, EfficientSU2, TwoLocal

Convention:
    All templates use the SF MSB-first convention (qubit 0 = leftmost / most significant bit).
    Data is loaded into qubits 0, 1, ..., n-1 in order — same as PennyLane's default.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.parameters import SymbolicParameter


# ── Embedding templates ────────────────────────────────────────────────────────

def AngleEmbedding(
    features: Sequence[float],
    n_qubits: int,
    rotation: str = "RY",
) -> Circuit:
    """Encode classical data as rotation angles on each qubit.

    Equivalent to ``qml.AngleEmbedding(features, wires, rotation=rotation)``.

    Args:
        features:  1-D sequence of floats, length ≤ n_qubits.
        n_qubits:  Total number of qubits.
        rotation:  Gate to use: 'RX', 'RY', or 'RZ'.

    Returns:
        SF Circuit with the encoding applied.
    """
    c = Circuit(n_qubits)
    gate_fn = getattr(c, rotation.lower())
    for i, val in enumerate(features[:n_qubits]):
        gate_fn(float(val), i)
    return c


def ZZFeatureMap(
    features: Sequence[float],
    n_qubits: int,
    reps: int = 2,
) -> Circuit:
    """ZZ feature map — Qiskit-compatible quantum feature map.

    Implements:
        for each rep:
          H on all qubits
          P(2 * x_i) on qubit i
          for each adjacent pair (i, i+1):
            CX(i, i+1)
            P(2 * (π - x_i)(π - x_{i+1}), i+1)
            CX(i, i+1)

    This matches Qiskit's ``ZZFeatureMap(n_qubits, reps=reps, entanglement='linear')``.

    Args:
        features:  Data vector, length == n_qubits.
        n_qubits:  Number of qubits.
        reps:      Number of repetitions.

    Returns:
        SF Circuit encoding the data.
    """
    x = list(features[:n_qubits])
    while len(x) < n_qubits:
        x.append(0.0)

    c = Circuit(n_qubits)
    for _ in range(reps):
        # Hadamard layer
        for i in range(n_qubits):
            c.h(i)
        # First-order terms
        for i in range(n_qubits):
            c.p(2.0 * x[i], i)
        # Second-order (ZZ) terms — linear entanglement
        for i in range(n_qubits - 1):
            c.cx(i, i + 1)
            c.p(2.0 * (math.pi - x[i]) * (math.pi - x[i + 1]), i + 1)
            c.cx(i, i + 1)
    return c


# ── Variational ansatz templates ───────────────────────────────────────────────

def BasicEntanglerLayers(
    weights: np.ndarray,
    n_qubits: int,
    rotation: str = "RX",
) -> Circuit:
    """Basic entangler layers — single-qubit rotations + ring CNOT entanglement.

    Matches ``qml.BasicEntanglerLayers(weights, wires, rotation=qml.RX)``.

    PennyLane default rotation is RX.  Pass rotation='RY' or 'RZ' to override.

    Args:
        weights:  2-D array of shape (n_layers, n_qubits).
        n_qubits: Number of qubits.
        rotation: Gate name to use: 'RX' (default), 'RY', or 'RZ'.

    Returns:
        SF Circuit.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    n_layers = weights.shape[0]

    c = Circuit(n_qubits)
    gate_fn = getattr(c, rotation.lower())
    for layer in range(n_layers):
        for i in range(n_qubits):
            gate_fn(float(weights[layer, i % weights.shape[1]]), i)
        # Ring CNOT: 0→1, 1→2, ..., (n-1)→0
        for i in range(n_qubits):
            c.cx(i, (i + 1) % n_qubits)
    return c


def StronglyEntanglingLayers(
    weights: np.ndarray,
    n_qubits: int,
    ranges: Optional[List[int]] = None,
) -> Circuit:
    """Strongly entangling layers — Rot + CNOT with specified ranges.

    Matches ``qml.StronglyEntanglingLayers(weights, wires, ranges=ranges)``.

    Each layer applies:
      Rot(α, β, γ) = RZ(γ) RY(β) RZ(α) on each qubit
      CNOT(i, (i + range) % n_qubits) for each qubit

    Args:
        weights:  3-D array of shape (n_layers, n_qubits, 3).
                  weights[l, i, :] = (alpha, beta, gamma) for qubit i in layer l.
        n_qubits: Number of qubits.
        ranges:   CNOT ranges per layer. Defaults to [1, 2, 3, ...].

    Returns:
        SF Circuit.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 2:
        weights = weights.reshape(weights.shape[0], -1, 3)
    n_layers = weights.shape[0]

    if ranges is None:
        ranges = [(l % (n_qubits - 1) + 1) if n_qubits > 1 else 1 for l in range(n_layers)]

    c = Circuit(n_qubits)
    for l in range(n_layers):
        r = ranges[l] if l < len(ranges) else 1
        for i in range(n_qubits):
            alpha = float(weights[l, i % weights.shape[1], 0])
            beta  = float(weights[l, i % weights.shape[1], 1])
            gamma = float(weights[l, i % weights.shape[1], 2])
            # Rot(alpha, beta, gamma) = RZ(gamma) RY(beta) RZ(alpha)
            c.rz(alpha, i).ry(beta, i).rz(gamma, i)
        for i in range(n_qubits):
            c.cx(i, (i + r) % n_qubits)
    return c


def HardwareEfficientAnsatz(
    n_qubits: int,
    n_layers: int,
    param_prefix: str = "w",
    parametric: bool = True,
) -> Circuit:
    """Hardware-efficient ansatz: alternating RY/RZ rotations + CZ entanglement.

    Matches Qiskit's ``EfficientSU2`` (entanglement='linear').

    Args:
        n_qubits:    Number of qubits.
        n_layers:    Number of repetitions.
        param_prefix: Prefix for symbolic parameter names.
        parametric:  If True, gates use ``sf.param()`` symbols (for VQE/optimization).
                     If False, gates use zeros (circuit structure only).

    Returns:
        Parametric SF Circuit.
    """
    c = Circuit(n_qubits)
    idx = 0
    for layer in range(n_layers + 1):
        for i in range(n_qubits):
            pname_y = f"{param_prefix}_{idx}"
            pname_z = f"{param_prefix}_{idx + 1}"
            w_y = sf.param(pname_y) if parametric else 0.0
            w_z = sf.param(pname_z) if parametric else 0.0
            c.ry(w_y, i).rz(w_z, i)
            idx += 2
        if layer < n_layers:
            for i in range(n_qubits - 1):
                c.cx(i, i + 1)
    return c


def TwoLocal(
    n_qubits: int,
    rotation_blocks: List[str],
    entanglement_blocks: str = "cx",
    entanglement: str = "linear",
    reps: int = 3,
    param_prefix: str = "θ",
    parametric: bool = True,
) -> Circuit:
    """General two-local ansatz — Qiskit TwoLocal equivalent.

    Args:
        n_qubits:           Number of qubits.
        rotation_blocks:    List of rotation gate names, e.g. ['ry', 'rz'].
        entanglement_blocks: Entanglement gate: 'cx' (default) or 'cz'.
        entanglement:       'linear' (nearest-neighbor) or 'full'.
        reps:               Number of repetitions.
        param_prefix:       Symbolic parameter prefix.
        parametric:         Whether to use symbolic parameters.

    Returns:
        SF Circuit.
    """
    c = Circuit(n_qubits)
    idx = 0

    def _get_param():
        nonlocal idx
        name = f"{param_prefix}[{idx}]"
        idx += 1
        return sf.param(name) if parametric else 0.0

    for rep in range(reps + 1):
        # Rotation layer
        for gate_name in rotation_blocks:
            for i in range(n_qubits):
                getattr(c, gate_name.lower())(_get_param(), i)

        # Entanglement layer (skip after last rep)
        if rep < reps:
            if entanglement == "linear":
                pairs = [(i, i + 1) for i in range(n_qubits - 1)]
            else:  # full
                pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
            for i, j in pairs:
                if entanglement_blocks.lower() in ("cx", "cnot"):
                    c.cx(i, j)
                elif entanglement_blocks.lower() == "cz":
                    c.cz(i, j)

    return c


# ── Data re-uploading QNN ──────────────────────────────────────────────────────

def DataReuploadingCircuit(
    features: Sequence[float],
    n_qubits: int,
    n_layers: int,
    param_prefix: str = "v",
    parametric: bool = True,
) -> Circuit:
    """Data re-uploading classifier circuit.

    Each layer: AngleEmbedding(data) + variational rotations + entanglement.
    Matches ``qml.qnn.TorchLayer`` with data-reuploading structure.

    Args:
        features:    Classical data vector.
        n_qubits:    Number of qubits.
        n_layers:    Number of re-uploading layers.
        param_prefix: Symbolic parameter prefix.
        parametric:  Use symbolic parameters for variational part.
    """
    x = list(features[:n_qubits])
    while len(x) < n_qubits:
        x.append(0.0)

    c = Circuit(n_qubits)
    idx = 0

    for layer in range(n_layers):
        # Encode data
        for i in range(n_qubits):
            c.ry(x[i], i)
        # Variational rotations
        for i in range(n_qubits):
            pname = f"{param_prefix}_{layer}_{i}"
            w = sf.param(pname) if parametric else 0.0
            c.rz(w, i).ry(w, i)
            idx += 1
        # Entanglement
        for i in range(n_qubits - 1):
            c.cx(i, i + 1)

    return c
