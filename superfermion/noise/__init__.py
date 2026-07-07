"""
Noise Models — Quantum channel simulation for realistic circuit execution.

Unified noise model supporting both statevector (JAX) and density matrix
(Kraus operator) simulation paths.

Usage:
    noise = sf.NoiseModel()
    noise.add_depolarizing(0.01)
    noise.add_readout_error(0.02)
    result = sf.run(circuit, method="density_matrix", noise_model=noise)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np



_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kraus_depolarizing_1q(p: float) -> List[np.ndarray]:
    """Single-qubit depolarizing: rho -> (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ)"""
    return [
        math.sqrt(1 - p) * _I2,
        math.sqrt(p / 3) * _X,
        math.sqrt(p / 3) * _Y,
        math.sqrt(p / 3) * _Z,
    ]


def kraus_depolarizing_2q(p: float) -> List[np.ndarray]:
    """Two-qubit depolarizing channel with 15 non-identity Pauli terms."""
    paulis_1q = [_I2, _X, _Y, _Z]
    kraus = [math.sqrt(1 - p) * np.kron(_I2, _I2)]
    for a in paulis_1q:
        for b in paulis_1q:
            if not (np.allclose(a, _I2) and np.allclose(b, _I2)):
                kraus.append(math.sqrt(p / 15) * np.kron(a, b))
    return kraus


def kraus_amplitude_damping(gamma: float) -> List[np.ndarray]:
    """Amplitude damping (T1 decay): K0 = diag(1, sqrt(1-g)), K1 = [[0,sqrt(g)],[0,0]]"""
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128),
    ]


def kraus_phase_damping(gamma: float) -> List[np.ndarray]:
    """Phase damping (T2 dephasing): K0 = diag(1, sqrt(1-g)), K1 = diag(0, sqrt(g))"""
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, 0], [0, math.sqrt(gamma)]], dtype=np.complex128),
    ]


def kraus_bit_flip(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - p) * _I2, math.sqrt(p) * _X]


def kraus_phase_flip(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - p) * _I2, math.sqrt(p) * _Z]


def kraus_bit_phase_flip(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - p) * _I2, math.sqrt(p) * _Y]


@dataclass
class NoiseChannel:
    """A single noise channel applied after a gate."""
    name: str
    error_rate: float
    apply: Optional[Callable] = None

    @property
    def gate(self):
        return self.name

    @property
    def rate(self):
        return self.error_rate


class NoiseModel:
    """Unified noise model for all simulation methods.

    Stores both JAX callables (statevector path) and Kraus operators
    (density matrix path). The ``has_noise`` property indicates whether
    any noise channels are registered.

    Usage::

        noise = NoiseModel()
        noise.add_depolarizing(0.01)
        noise.add_amplitude_damping(0.005)
        noise.add_readout_error(0.02)

        # Use with sf.run()
        result = sf.run(circuit, method="density_matrix", noise_model=noise)
    """

    def __init__(self) -> None:
        self.single_qubit_channels: List[NoiseChannel] = []
        self.two_qubit_channels: List[NoiseChannel] = []
        self.readout_error: float = 0.0
        self._1q_kraus: List[List[np.ndarray]] = []
        self._2q_kraus: List[List[np.ndarray]] = []
        self._readout_p: float = 0.0

    @property
    def has_noise(self) -> bool:
        return bool(
            self.single_qubit_channels
            or self.two_qubit_channels
            or self._1q_kraus
            or self._2q_kraus
            or self.readout_error > 0
            or self._readout_p > 0
        )

    def add_depolarizing(self, p: float, n_qubits: int = 1) -> "NoiseModel":
        """Add depolarizing noise with error probability p."""
        channel = NoiseChannel("depolarizing", p)
        if n_qubits == 1:
            self.single_qubit_channels.append(channel)
            self._1q_kraus.append(kraus_depolarizing_1q(p))
        else:
            self.two_qubit_channels.append(channel)
            self._2q_kraus.append(kraus_depolarizing_2q(p))
        return self

    def add_amplitude_damping(self, gamma: float) -> "NoiseModel":
        """Add amplitude damping (T1 decay) with rate gamma."""
        channel = NoiseChannel("amplitude_damping", gamma)
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_amplitude_damping(gamma))
        return self

    def add_phase_damping(self, gamma: float) -> "NoiseModel":
        """Add phase damping (T2 dephasing) with rate gamma."""
        channel = NoiseChannel("phase_damping", gamma)
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_phase_damping(gamma))
        return self

    def add_bit_flip(self, p: float) -> "NoiseModel":
        """Add bit-flip noise with probability p."""
        channel = NoiseChannel("bit_flip", p)
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_bit_flip(p))
        return self

    def add_phase_flip(self, p: float) -> "NoiseModel":
        """Add phase-flip noise with probability p."""
        channel = NoiseChannel("phase_flip", p)
        self.single_qubit_channels.append(channel)
        self._1q_kraus.append(kraus_phase_flip(p))
        return self

    def add_readout_error(self, p: float) -> "NoiseModel":
        """Add readout bit-flip error with probability p."""
        self.readout_error = p
        self._readout_p = p
        return self

    def add_two_qubit_depolarizing(self, p: float) -> "NoiseModel":
        """Convenience wrapper for add_depolarizing with n_qubits=2."""
        return self.add_depolarizing(p, n_qubits=2)

    def apply_1q(self, rho: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply all 1-qubit Kraus channels to density matrix."""
        from superfermion.backends.density_matrix import _apply_kraus_1q
        for kraus_set in self._1q_kraus:
            rho = _apply_kraus_1q(rho, kraus_set, qubit, n)
        return rho

    def apply_2q(self, rho: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
        """Apply all 2-qubit Kraus channels to density matrix."""
        from superfermion.backends.density_matrix import _apply_kraus_2q
        for kraus_set in self._2q_kraus:
            rho = _apply_kraus_2q(rho, kraus_set, q0, q1, n)
        return rho

    def apply_to_counts(self, counts: Dict[str, int], rng=None) -> Dict[str, int]:
        """Apply readout error to measurement counts."""
        p = self.readout_error or self._readout_p
        if p <= 0:
            return counts
        if rng is None:
            rng = np.random.default_rng(42)
        elif not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(int(np.asarray(rng).flat[0]))
        noisy: Dict[str, int] = {}
        for bs, count in counts.items():
            for _ in range(count):
                bits = list(bs)
                for i in range(len(bits)):
                    if rng.random() < p:
                        bits[i] = '1' if bits[i] == '0' else '0'
                new_bs = ''.join(bits)
                noisy[new_bs] = noisy.get(new_bs, 0) + 1
        return noisy

    def to_rust_kraus_ops(self, n_qubits: int) -> List[Tuple[int, List[float]]]:
        """Convert 1-qubit Kraus channels to the flat format expected by Rust.

        Returns list of (qubit, flat_kraus) tuples where flat_kraus encodes all
        Kraus matrices for that qubit as [re00, im00, re01, im01, ...] per matrix.
        Applied after every gate touching the qubit.
        """
        ops: List[Tuple[int, List[float]]] = []
        if not self._1q_kraus:
            return ops

        for q in range(n_qubits):
            flat: List[float] = []
            for kraus_set in self._1q_kraus:
                for K in kraus_set:
                    for r in range(2):
                        for c in range(2):
                            flat.append(float(K[r, c].real))
                            flat.append(float(K[r, c].imag))
            ops.append((q, flat))
        return ops

    def to_dict(self) -> Dict[str, Any]:
        """Serialize noise model to a dictionary."""
        return {
            "single_qubit_channels": [
                {"name": ch.name, "error_rate": ch.error_rate}
                for ch in self.single_qubit_channels
            ],
            "two_qubit_channels": [
                {"name": ch.name, "error_rate": ch.error_rate}
                for ch in self.two_qubit_channels
            ],
            "readout_error": self.readout_error,
        }

    def __repr__(self) -> str:
        channels = len(self.single_qubit_channels) + len(self.two_qubit_channels)
        return f"NoiseModel(channels={channels}, readout_error={self.readout_error})"


def ibm_eagle_noise() -> NoiseModel:
    """Approximate noise model for IBM Eagle (127-qubit) processor."""
    return (NoiseModel()
        .add_depolarizing(0.001)
        .add_depolarizing(0.01, n_qubits=2)
        .add_amplitude_damping(0.0005)
        .add_phase_damping(0.001)
        .add_readout_error(0.01))


def ideal_noise() -> NoiseModel:
    """No noise (ideal simulator)."""
    return NoiseModel()
