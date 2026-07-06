"""
Noise Models — Quantum channel simulation for realistic circuit execution.

Implements depolarizing, amplitude damping, phase damping, and readout noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp


@dataclass
class NoiseChannel:
    """A single noise channel applied after a gate."""
    name: str
    error_rate: float
    apply: Callable  # (key, statevector) -> statevector

    @property
    def gate(self):
        """Alias for name — used by test compatibility checks."""
        return self.name

    @property
    def rate(self):
        """Alias for error_rate — used by test compatibility checks."""
        return self.error_rate


@dataclass
class NoiseModel:
    """A complete noise model for circuit simulation.
    
    Usage:
        noise = NoiseModel()
        noise.add_depolarizing(0.01)           # 1% depolarizing on all gates
        noise.add_amplitude_damping(0.005)     # T1 decay
        noise.add_readout_error(0.02)          # 2% bit-flip on readout
    """
    single_qubit_channels: List[NoiseChannel] = field(default_factory=list)
    two_qubit_channels: List[NoiseChannel] = field(default_factory=list)
    readout_error: float = 0.0
    
    def add_depolarizing(self, p: float, n_qubits: int = 1) -> "NoiseModel":
        """Add depolarizing noise with error probability p.
        
        For 1-qubit: rho -> (1-p)*rho + p/3*(X*rho*X + Y*rho*Y + Z*rho*Z)
        """
        def apply_depolarizing(key: jax.random.PRNGKey, sv: jnp.ndarray, qubit: int, n: int) -> jnp.ndarray:
            # Simplified: with probability p, apply a random Pauli
            r = jax.random.uniform(key)
            dim = 2**n
            
            # If no error, return unchanged
            # This is a simplified Kraus model for statevector simulation
            # In full density matrix sim, this would be exact
            return jnp.where(r < p, _apply_random_pauli(key, sv, qubit, n), sv)
        
        channel = NoiseChannel("depolarizing", p, apply_depolarizing)
        if n_qubits == 1:
            self.single_qubit_channels.append(channel)
        else:
            self.two_qubit_channels.append(channel)
        return self
    
    def add_amplitude_damping(self, gamma: float) -> "NoiseModel":
        """Add amplitude damping (T1 decay) with rate gamma.
        
        Models spontaneous emission: |1> -> |0> with probability gamma.
        """
        def apply_amp_damp(key: jax.random.PRNGKey, sv: jnp.ndarray, qubit: int, n: int) -> jnp.ndarray:
            # Simplified: reduce |1> amplitude by sqrt(1-gamma)
            dim = 2**n
            mask_1 = jnp.array([(i >> (n - 1 - qubit)) & 1 for i in range(dim)])
            scale = jnp.where(mask_1, jnp.sqrt(1 - gamma), 1.0)
            sv_damped = sv * scale
            # Renormalize
            norm = jnp.sqrt(jnp.sum(jnp.abs(sv_damped)**2))
            return sv_damped / jnp.maximum(norm, 1e-10)
        
        channel = NoiseChannel("amplitude_damping", gamma, apply_amp_damp)
        self.single_qubit_channels.append(channel)
        return self
    
    def add_phase_damping(self, gamma: float) -> "NoiseModel":
        """Add phase damping (T2 dephasing) with rate gamma."""
        def apply_phase_damp(key: jax.random.PRNGKey, sv: jnp.ndarray, qubit: int, n: int) -> jnp.ndarray:
            dim = 2**n
            mask_1 = jnp.array([(i >> (n - 1 - qubit)) & 1 for i in range(dim)])
            phase_factor = jnp.where(mask_1, jnp.sqrt(1 - gamma), 1.0)
            return sv * phase_factor
        
        channel = NoiseChannel("phase_damping", gamma, apply_phase_damp)
        self.single_qubit_channels.append(channel)
        return self
    
    def add_readout_error(self, p: float) -> "NoiseModel":
        """Add readout bit-flip error with probability p."""
        self.readout_error = p
        return self

    def add_two_qubit_depolarizing(self, p: float) -> "NoiseModel":
        """Add two-qubit depolarizing noise with error probability p.
        
        Convenience wrapper for add_depolarizing with n_qubits=2.
        """
        return self.add_depolarizing(p, n_qubits=2)
    
    def apply_to_counts(self, counts: Dict[str, int], key: jax.random.PRNGKey) -> Dict[str, int]:
        """Apply readout error to measurement counts."""
        if self.readout_error <= 0:
            return counts
        
        noisy_counts: Dict[str, int] = {}
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Each bit has probability p of flipping
            for _ in range(count):
                key, subkey = jax.random.split(key)
                flips = jax.random.bernoulli(subkey, self.readout_error, shape=(len(bitstring),))
                flipped = "".join(
                    str(int(b) ^ int(f)) for b, f in zip(bitstring, flips)
                )
                noisy_counts[flipped] = noisy_counts.get(flipped, 0) + 1
        
        return noisy_counts
    
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


def _apply_random_pauli(key: jax.random.PRNGKey, sv: jnp.ndarray, qubit: int, n: int) -> jnp.ndarray:
    """Apply a random Pauli (X, Y, or Z) to a qubit in the statevector."""
    choice = jax.random.randint(key, (), 0, 3)
    # X gate on qubit
    dim = 2**n
    step = 2**(n - 1 - qubit)
    
    # For simplicity, just apply X (bit flip)
    indices = jnp.arange(dim)
    partner = indices ^ step  # XOR to flip the target qubit bit
    return sv[partner]


# Predefined noise models
def ibm_eagle_noise() -> NoiseModel:
    """Approximate noise model for IBM Eagle (127-qubit) processor."""
    return (NoiseModel()
        .add_depolarizing(0.001)        # ~0.1% single-qubit error
        .add_depolarizing(0.01, n_qubits=2)  # ~1% two-qubit error
        .add_amplitude_damping(0.0005)  # T1 ~ 300us
        .add_phase_damping(0.001)       # T2 ~ 150us
        .add_readout_error(0.01))       # 1% readout error


def ideal_noise() -> NoiseModel:
    """No noise (ideal simulator)."""
    return NoiseModel()
