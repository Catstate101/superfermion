"""Pulse module domain tests — waveforms and schedules."""

import numpy as np
import pytest

from superfermion.pulse import (
    DRAGPulse,
    GaussianPulse,
    Schedule,
    SquarePulse,
)


pytestmark = pytest.mark.domain


class TestWaveforms:
    def test_gaussian_pulse_samples(self):
        pulse = GaussianPulse(duration=40, sigma=8.0, amp=0.5)
        samples = pulse.samples()
        assert samples.shape == (40,)
        assert np.iscomplexobj(samples)
        assert float(np.max(np.abs(samples))) <= 0.5 + 1e-12
        assert samples.dtype == np.complex128

    def test_drag_pulse_has_imaginary_component(self):
        pulse = DRAGPulse(duration=40, sigma=8.0, amp=0.5, beta=0.5)
        samples = pulse.samples()
        assert samples.shape == (40,)
        assert np.any(np.imag(samples) != 0)

    def test_square_pulse_constant_amplitude(self):
        pulse = SquarePulse(duration=20, amp=0.75)
        samples = pulse.samples()
        assert samples.shape == (20,)
        assert np.allclose(np.real(samples), 0.75)

    def test_waveform_repr(self):
        pulse = GaussianPulse(duration=160, sigma=40, amp=0.5)
        assert "GaussianPulse" in repr(pulse)
        assert "160" in repr(pulse)


class TestSchedule:
    def test_schedule_add_and_duration(self):
        schedule = Schedule(name="x_gate")
        schedule.add(GaussianPulse(160, sigma=40, amp=0.5), channel="d0")
        schedule.add(GaussianPulse(80, sigma=20, amp=0.3), channel="d1", t0=0)

        assert schedule.n_instructions == 2
        assert schedule.duration == 160
        assert "d0" in schedule.channels
        assert "d1" in schedule.channels

    def test_schedule_sequential_append(self):
        schedule = Schedule()
        schedule.add(SquarePulse(50, amp=1.0), channel="d0")
        schedule.add(SquarePulse(30, amp=0.5), channel="d0")
        assert schedule.duration == 80
        instructions = schedule.instructions
        assert instructions[0].t0 == 0
        assert instructions[1].t0 == 50

    def test_schedule_barrier_syncs_channels(self):
        schedule = Schedule()
        schedule.add(SquarePulse(40, amp=1.0), channel="d0")
        schedule.add(SquarePulse(20, amp=1.0), channel="d1")
        schedule.barrier("d0", "d1")
        schedule.add(SquarePulse(10, amp=1.0), channel="d0")
        schedule.add(SquarePulse(10, amp=1.0), channel="d1")
        assert schedule.duration == 50

    def test_schedule_concatenation(self):
        s1 = Schedule("a").add(SquarePulse(20, amp=1.0), channel="d0")
        s2 = Schedule("b").add(SquarePulse(15, amp=0.5), channel="d0")
        combined = s1 + s2
        assert combined.duration == 35
        assert combined.n_instructions == 2

    def test_get_channel_waveform(self):
        schedule = Schedule()
        schedule.add(GaussianPulse(10, sigma=2.0, amp=1.0), channel="d0", t0=5)
        waveform = schedule.get_channel_waveform("d0")
        assert waveform.shape == (15,)
        assert float(np.max(np.abs(waveform[5:15]))) > 0
