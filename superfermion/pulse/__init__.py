"""
Superfermion Pulse — Pulse-level quantum control.

Python API for pulse schedule construction, waveform generation,
and hardware-level gate calibration.

Usage:
    >>> from superfermion.pulse import GaussianPulse, DRAGPulse, Schedule
    >>> pulse = GaussianPulse(duration=160, sigma=40, amp=0.5)
    >>> schedule = Schedule().add(pulse, channel="d0", t0=0)
"""

from __future__ import annotations

from superfermion.pulse.waveforms import (
    Waveform, GaussianPulse, DRAGPulse, SquarePulse,
    GaussianSquarePulse, CosinePulse,
)
from superfermion.pulse.schedule import (
    Schedule, PulseInstruction, Channel, ChannelType,
)
from superfermion.pulse.calibration import (
    GateCalibration, CalibrationSet,
)

__all__ = [
    "Waveform", "GaussianPulse", "DRAGPulse", "SquarePulse",
    "GaussianSquarePulse", "CosinePulse",
    "Schedule", "PulseInstruction", "Channel", "ChannelType",
    "GateCalibration", "CalibrationSet",
]
