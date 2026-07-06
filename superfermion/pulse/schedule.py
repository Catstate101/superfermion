"""
Pulse Schedule — Time-ordered sequence of pulse instructions on channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from superfermion.pulse.waveforms import Waveform


class ChannelType(Enum):
    """Types of pulse channels."""
    DRIVE = "d"         # Qubit drive channel
    CONTROL = "u"       # Cross-resonance / control
    MEASURE = "m"       # Measurement stimulus
    ACQUIRE = "a"       # Acquisition channel


@dataclass
class Channel:
    """A pulse channel targeting a specific qubit.

    Args:
        channel_type: Type of channel (drive, control, measure, acquire).
        index: Channel index (typically corresponds to qubit index).
    """
    channel_type: ChannelType
    index: int

    @property
    def name(self) -> str:
        return f"{self.channel_type.value}{self.index}"

    @classmethod
    def drive(cls, qubit: int) -> Channel:
        return cls(ChannelType.DRIVE, qubit)

    @classmethod
    def control(cls, index: int) -> Channel:
        return cls(ChannelType.CONTROL, index)

    @classmethod
    def measure(cls, qubit: int) -> Channel:
        return cls(ChannelType.MEASURE, qubit)

    @classmethod
    def from_string(cls, name: str) -> Channel:
        """Parse 'd0', 'u1', 'm0' etc."""
        ch_type = ChannelType(name[0])
        index = int(name[1:])
        return cls(ch_type, index)

    def __repr__(self) -> str:
        return f"Channel('{self.name}')"


@dataclass
class PulseInstruction:
    """A single instruction in a pulse schedule.

    Attributes:
        waveform: The pulse waveform to play.
        channel: Target channel.
        t0: Start time in dt units.
        phase: Optional phase shift (radians).
        frequency: Optional frequency shift (Hz).
    """
    waveform: Waveform
    channel: Channel
    t0: int = 0
    phase: float = 0.0
    frequency: float = 0.0

    @property
    def t1(self) -> int:
        """End time."""
        return self.t0 + self.waveform.duration

    @property
    def duration(self) -> int:
        return self.waveform.duration

    def __repr__(self) -> str:
        return (
            f"PulseInstruction({self.waveform.name}, "
            f"ch={self.channel.name}, t=[{self.t0}:{self.t1}])"
        )


class Schedule:
    """A pulse schedule — time-ordered sequence of pulse instructions.

    Supports building complex pulse programs with multiple channels,
    alignment, and sequential/parallel composition.

    Examples:
        >>> from superfermion.pulse import GaussianPulse, Schedule
        >>> s = Schedule(name="x_gate")
        >>> s.add(GaussianPulse(160, sigma=40, amp=0.5), channel="d0")
        >>> s.add(GaussianPulse(160, sigma=40, amp=0.3), channel="d1", t0=0)
        >>> print(s.duration)
        160
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._instructions: List[PulseInstruction] = []
        self._current_time: Dict[str, int] = {}  # Per-channel cursor

    def add(
        self,
        waveform: Waveform,
        channel: str = "d0",
        t0: Optional[int] = None,
        phase: float = 0.0,
        frequency: float = 0.0,
    ) -> Schedule:
        """Add a pulse to the schedule.

        Args:
            waveform: Waveform to play.
            channel: Channel name (e.g., 'd0', 'u1', 'm0').
            t0: Start time. None = append after last pulse on this channel.
            phase: Phase shift in radians.
            frequency: Frequency shift in Hz.

        Returns:
            self for chaining.
        """
        ch = Channel.from_string(channel)

        if t0 is None:
            t0 = self._current_time.get(channel, 0)

        inst = PulseInstruction(
            waveform=waveform,
            channel=ch,
            t0=t0,
            phase=phase,
            frequency=frequency,
        )
        self._instructions.append(inst)
        self._current_time[channel] = t0 + waveform.duration

        return self

    def shift_phase(self, phase: float, channel: str) -> Schedule:
        """Apply a virtual phase shift on a channel.

        Args:
            phase: Phase shift in radians.
            channel: Target channel.

        Returns:
            self for chaining.
        """
        # Virtual Z rotation via frame change
        ch = Channel.from_string(channel)
        self._instructions.append(PulseInstruction(
            waveform=Waveform(duration=0, amp=0, name="shift_phase"),
            channel=ch,
            t0=self._current_time.get(channel, 0),
            phase=phase,
        ))
        return self

    def barrier(self, *channels: str) -> Schedule:
        """Synchronize channels — all specified channels align to the latest time.

        Args:
            *channels: Channel names to synchronize.

        Returns:
            self for chaining.
        """
        if not channels:
            channels = tuple(self._current_time.keys())
        max_time = max(self._current_time.get(ch, 0) for ch in channels)
        for ch in channels:
            self._current_time[ch] = max_time
        return self

    @property
    def duration(self) -> int:
        """Total schedule duration."""
        if not self._instructions:
            return 0
        return max(inst.t1 for inst in self._instructions)

    @property
    def instructions(self) -> List[PulseInstruction]:
        """All instructions sorted by start time."""
        return sorted(self._instructions, key=lambda i: (i.t0, i.channel.name))

    @property
    def channels(self) -> List[str]:
        """List of all channels used."""
        return sorted(set(inst.channel.name for inst in self._instructions))

    @property
    def n_instructions(self) -> int:
        return len(self._instructions)

    def get_channel_waveform(self, channel: str, total_duration: Optional[int] = None) -> np.ndarray:
        """Get the composite waveform for a specific channel.

        Args:
            channel: Channel name.
            total_duration: Total length. Defaults to schedule duration.

        Returns:
            Complex waveform array.
        """
        dur = total_duration or self.duration
        if dur <= 0:
            return np.array([], dtype=np.complex128)

        waveform = np.zeros(dur, dtype=np.complex128)

        for inst in self._instructions:
            if inst.channel.name != channel:
                continue
            if inst.waveform.duration <= 0:
                continue
            samples = inst.waveform.samples()
            # Apply phase
            if inst.phase != 0:
                samples = samples * np.exp(1j * inst.phase)
            # Place in timeline
            end = min(inst.t0 + len(samples), dur)
            n_copy = end - inst.t0
            waveform[inst.t0:end] = samples[:n_copy]

        return waveform

    def __add__(self, other: Schedule) -> Schedule:
        """Concatenate two schedules sequentially."""
        combined = Schedule(name=f"{self.name}+{other.name}")
        offset = self.duration

        for inst in self._instructions:
            combined._instructions.append(inst)

        for inst in other._instructions:
            shifted = PulseInstruction(
                waveform=inst.waveform,
                channel=inst.channel,
                t0=inst.t0 + offset,
                phase=inst.phase,
                frequency=inst.frequency,
            )
            combined._instructions.append(shifted)

        # Update cursors
        for ch, t in self._current_time.items():
            combined._current_time[ch] = t
        for ch, t in other._current_time.items():
            combined._current_time[ch] = t + offset

        return combined

    def __repr__(self) -> str:
        name = f"'{self.name}', " if self.name else ""
        return (
            f"Schedule({name}instructions={self.n_instructions}, "
            f"duration={self.duration}dt, channels={self.channels})"
        )
