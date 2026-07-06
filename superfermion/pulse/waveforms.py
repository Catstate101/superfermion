"""
Waveforms — Pulse envelope generators for quantum gate implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Waveform:
    """Base waveform class.

    Attributes:
        duration: Pulse duration in dt units.
        amp: Pulse amplitude (0 to 1).
        name: Waveform name.
    """
    duration: int
    amp: float = 1.0
    name: str = "custom"

    def samples(self, dt: float = 1.0) -> np.ndarray:
        """Generate waveform samples.

        Args:
            dt: Time step in nanoseconds.

        Returns:
            Complex array of waveform samples.
        """
        raise NotImplementedError("Subclasses must implement samples()")

    @property
    def n_samples(self) -> int:
        return self.duration

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, amp={self.amp})"


class GaussianPulse(Waveform):
    """Gaussian envelope pulse.

    Standard Gaussian: A * exp(-(t-center)^2 / (2*sigma^2))

    Args:
        duration: Pulse duration in dt.
        sigma: Gaussian width parameter.
        amp: Peak amplitude.
    """

    def __init__(self, duration: int, sigma: float, amp: float = 1.0) -> None:
        super().__init__(duration=duration, amp=amp, name="gaussian")
        self.sigma = sigma

    def samples(self, dt: float = 1.0) -> np.ndarray:
        center = self.duration / 2
        t = np.arange(self.duration, dtype=np.float64)
        gauss = self.amp * np.exp(-0.5 * ((t - center) / self.sigma) ** 2)
        # Zero out edges below 1% to reduce leakage
        gauss[gauss < 0.01 * self.amp] = 0.0
        return gauss.astype(np.complex128)


class DRAGPulse(Waveform):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse.

    Reduces leakage to non-computational states in transmon qubits.
    DRAG = Gaussian + i * beta * dGaussian/dt

    Args:
        duration: Pulse duration.
        sigma: Gaussian width.
        amp: Peak amplitude.
        beta: DRAG correction parameter.
    """

    def __init__(
        self,
        duration: int,
        sigma: float,
        amp: float = 1.0,
        beta: float = 0.5,
    ) -> None:
        super().__init__(duration=duration, amp=amp, name="drag")
        self.sigma = sigma
        self.beta = beta

    def samples(self, dt: float = 1.0) -> np.ndarray:
        center = self.duration / 2
        t = np.arange(self.duration, dtype=np.float64)

        # Gaussian component
        gauss = self.amp * np.exp(-0.5 * ((t - center) / self.sigma) ** 2)

        # Derivative component
        d_gauss = -(t - center) / (self.sigma ** 2) * gauss

        # DRAG: I + i*Q
        return (gauss + 1j * self.beta * d_gauss).astype(np.complex128)


class SquarePulse(Waveform):
    """Constant-amplitude square pulse."""

    def __init__(self, duration: int, amp: float = 1.0) -> None:
        super().__init__(duration=duration, amp=amp, name="square")

    def samples(self, dt: float = 1.0) -> np.ndarray:
        return np.full(self.duration, self.amp, dtype=np.complex128)


class GaussianSquarePulse(Waveform):
    """Gaussian-square pulse: flat top with Gaussian rise/fall.

    Args:
        duration: Total pulse duration.
        sigma: Gaussian rise/fall width.
        amp: Peak amplitude.
        width: Duration of the flat-top portion.
    """

    def __init__(
        self,
        duration: int,
        sigma: float,
        amp: float = 1.0,
        width: Optional[int] = None,
    ) -> None:
        super().__init__(duration=duration, amp=amp, name="gaussian_square")
        self.sigma = sigma
        self.width = width or max(1, duration - 6 * int(sigma))

    def samples(self, dt: float = 1.0) -> np.ndarray:
        t = np.arange(self.duration, dtype=np.float64)
        rise_time = (self.duration - self.width) // 2
        fall_start = rise_time + self.width

        envelope = np.zeros(self.duration, dtype=np.float64)

        # Rise (Gaussian)
        if rise_time > 0:
            t_rise = t[:rise_time]
            envelope[:rise_time] = np.exp(-0.5 * ((t_rise - rise_time) / self.sigma) ** 2)

        # Flat top
        envelope[rise_time:fall_start] = 1.0

        # Fall (Gaussian)
        if fall_start < self.duration:
            t_fall = t[fall_start:] - fall_start
            envelope[fall_start:] = np.exp(-0.5 * (t_fall / self.sigma) ** 2)

        return (self.amp * envelope).astype(np.complex128)


class CosinePulse(Waveform):
    """Cosine-shaped pulse envelope."""

    def __init__(self, duration: int, amp: float = 1.0) -> None:
        super().__init__(duration=duration, amp=amp, name="cosine")

    def samples(self, dt: float = 1.0) -> np.ndarray:
        t = np.linspace(0, np.pi, self.duration)
        envelope = self.amp * 0.5 * (1 - np.cos(2 * t / self.duration * np.pi * self.duration / (2*np.pi)))
        # Simplified: raised cosine
        envelope = self.amp * np.sin(t) ** 2
        return envelope.astype(np.complex128)
