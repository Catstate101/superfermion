"""
Gate Calibration — Map quantum gates to calibrated pulse schedules.
"""
from __future__ import annotations


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from superfermion.pulse.waveforms import GaussianPulse, DRAGPulse, GaussianSquarePulse
from superfermion.pulse.schedule import Schedule, Channel


@dataclass
class GateCalibration:
    """Calibration data for a specific gate on specific qubits.

    Attributes:
        gate_name: Gate name (e.g., 'x', 'cx').
        qubits: Tuple of qubit indices.
        schedule: Calibrated pulse schedule.
        fidelity: Measured gate fidelity.
        calibration_date: When calibration was performed.
        metadata: Additional calibration metadata.
    """
    gate_name: str
    qubits: Tuple[int, ...]
    schedule: Schedule
    fidelity: float = 1.0
    calibration_date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Unique key for this calibration."""
        qubits_str = ",".join(str(q) for q in self.qubits)
        return f"{self.gate_name}({qubits_str})"

    def __repr__(self) -> str:
        return (
            f"GateCalibration('{self.gate_name}', qubits={self.qubits}, "
            f"fidelity={self.fidelity:.4f})"
        )


class CalibrationSet:
    """A complete set of gate calibrations for a backend.

    Manages pulse schedules for all calibrated gates.

    Args:
        backend_name: Name of the target backend.
        dt: Time step in nanoseconds.

    Examples:
        >>> cals = CalibrationSet("ibm_brisbane", dt=0.222)
        >>> cals.add_default_single_qubit(qubit=0)
        >>> schedule = cals.get("x", (0,))
    """

    def __init__(
        self,
        backend_name: str = "",
        dt: float = 0.222,  # ns, typical IBM dt
    ) -> None:
        self.backend_name = backend_name
        self.dt = dt
        self._calibrations: Dict[str, GateCalibration] = {}

    def add(self, calibration: GateCalibration) -> None:
        """Add a gate calibration."""
        self._calibrations[calibration.key] = calibration

    def get(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
    ) -> Optional[Schedule]:
        """Get the calibrated schedule for a gate.

        Args:
            gate_name: Gate name.
            qubits: Qubit indices.

        Returns:
            Calibrated Schedule, or None if not calibrated.
        """
        key = f"{gate_name}({','.join(str(q) for q in qubits)})"
        cal = self._calibrations.get(key)
        return cal.schedule if cal else None

    def get_calibration(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
    ) -> Optional[GateCalibration]:
        """Get full calibration data."""
        key = f"{gate_name}({','.join(str(q) for q in qubits)})"
        return self._calibrations.get(key)

    def add_default_single_qubit(
        self,
        qubit: int,
        duration: int = 160,
        sigma: float = 40.0,
        x_amp: float = 0.5,
        drag_beta: float = 0.3,
    ) -> None:
        """Add default calibrations for standard single-qubit gates.

        Creates X, SX, and RZ calibrations using DRAG pulses.

        Args:
            qubit: Qubit index.
            duration: Pulse duration in dt.
            sigma: Gaussian width.
            x_amp: X gate amplitude.
            drag_beta: DRAG correction parameter.
        """
        ch_name = f"d{qubit}"

        # X gate: π rotation
        x_sched = Schedule(name=f"x_q{qubit}")
        x_sched.add(DRAGPulse(duration, sigma, amp=x_amp, beta=drag_beta), channel=ch_name)
        self.add(GateCalibration("x", (qubit,), x_sched, fidelity=0.9995))

        # SX gate: π/2 rotation
        sx_sched = Schedule(name=f"sx_q{qubit}")
        sx_sched.add(DRAGPulse(duration, sigma, amp=x_amp / 2, beta=drag_beta), channel=ch_name)
        self.add(GateCalibration("sx", (qubit,), sx_sched, fidelity=0.9998))

        # RZ gate: virtual Z (zero duration, frame change only)
        rz_sched = Schedule(name=f"rz_q{qubit}")
        rz_sched.shift_phase(0.0, ch_name)  # Phase set at runtime
        self.add(GateCalibration("rz", (qubit,), rz_sched, fidelity=1.0))

    def add_default_two_qubit(
        self,
        control: int,
        target: int,
        cx_duration: int = 640,
        sigma: float = 64.0,
        cx_amp: float = 0.3,
    ) -> None:
        """Add default CX gate calibration.

        Uses cross-resonance (GaussianSquare) on the control channel.

        Args:
            control: Control qubit.
            target: Target qubit.
            cx_duration: CX gate duration.
            sigma: Rise/fall sigma.
            cx_amp: Cross-resonance amplitude.
        """
        # Cross-resonance pulse on control channel
        cr_idx = control * 2 + (1 if target > control else 0)
        ch_name = f"u{cr_idx}"

        cx_sched = Schedule(name=f"cx_q{control}_q{target}")
        cx_sched.add(
            GaussianSquarePulse(cx_duration, sigma, amp=cx_amp),
            channel=ch_name,
        )
        self.add(GateCalibration(
            "cx", (control, target), cx_sched, fidelity=0.995,
        ))

    def list_calibrations(self) -> List[GateCalibration]:
        """List all calibrations."""
        return list(self._calibrations.values())

    @property
    def n_calibrations(self) -> int:
        return len(self._calibrations)

    # ── Noise parameter extraction (ZNE → calibration bridge) ────────

    def extract_noise_params(self) -> Dict[str, Any]:
        """Extract noise parameters from calibration data for ZNE and
        noise model construction.

        Computes average gate fidelities from stored GateCalibration
        entries and converts them to standard noise-model parameters
        (depolarizing rates, amplitude damping, readout error, etc.).

        Returns:
            Dict with keys:
            - ``avg_1q_fidelity``: Average single-qubit gate fidelity
            - ``avg_2q_fidelity``: Average two-qubit gate fidelity
            - ``readout_error``: Estimated readout error rate
            - ``depolarizing_1q``: 1Q depolarizing probability
            - ``depolarizing_2q``: 2Q depolarizing probability
            - ``amplitude_damping``: T1-derived amplitude damping rate
            - ``phase_damping``: T2-derived phase damping rate
            - ``noise_factors``: Richardson ZNE scale factors [1, 2, 3]
            - ``backend_name``: Source calibration backend
        """
        oneq_fids = []
        twoq_fids = []

        for cal in self._calibrations.values():
            if len(cal.qubits) == 1:
                oneq_fids.append(cal.fidelity)
            elif len(cal.qubits) == 2:
                twoq_fids.append(cal.fidelity)

        avg_1q = float(np.mean(oneq_fids)) if oneq_fids else 1.0
        avg_2q = float(np.mean(twoq_fids)) if twoq_fids else 1.0

        # Convert fidelity to depolarizing probability:
        # For n_qubits = 1, p = (1 - F) * (d^2) / (d^2 - 1) where d = 2
        # Simplified: p ≈ (1 - F) * 4/3 for 1Q, (1 - F) * 16/15 for 2Q
        dp_1q = max(0.0, (1.0 - avg_1q) * 4.0 / 3.0)
        dp_2q = max(0.0, (1.0 - avg_2q) * 16.0 / 15.0)

        # Readout error: estimated from worst 1Q fidelity deviation
        readout_err = max(0.0, min(0.15, (1.0 - avg_1q) * 0.5))

        # Amplitude damping (T1): approximated from 1Q infidelity
        amp_damp = max(0.0, (1.0 - avg_1q) * 0.3)

        # Phase damping (T2): approximated from 1Q infidelity
        phase_damp = max(0.0, (1.0 - avg_1q) * 0.6)

        return {
            "avg_1q_fidelity": avg_1q,
            "avg_2q_fidelity": avg_2q,
            "readout_error": readout_err,
            "depolarizing_1q": dp_1q,
            "depolarizing_2q": dp_2q,
            "amplitude_damping": amp_damp,
            "phase_damping": phase_damp,
            "noise_factors": [1, 2, 3],
            "backend_name": self.backend_name,
        }

    def to_noise_model(self) -> "NoiseModel":
        """Convert calibration data to a NoiseModel for simulation.

        Uses extracted noise parameters to construct a
        ``superfermion.noise.NoiseModel`` with appropriate
        depolarizing, damping, and readout channels.

        Returns:
            ``NoiseModel`` populated from calibration.
        """
        from superfermion.noise import NoiseModel

        params = self.extract_noise_params()
        model = NoiseModel()

        if params["depolarizing_1q"] > 0:
            model.add_depolarizing(params["depolarizing_1q"], n_qubits=1)
        if params["depolarizing_2q"] > 0:
            model.add_depolarizing(params["depolarizing_2q"], n_qubits=2)
        if params["amplitude_damping"] > 0:
            model.add_amplitude_damping(params["amplitude_damping"])
        if params["phase_damping"] > 0:
            model.add_phase_damping(params["phase_damping"])
        if params["readout_error"] > 0:
            model.add_readout_error(params["readout_error"])

        return model

    def __repr__(self) -> str:
        return (
            f"CalibrationSet('{self.backend_name}', "
            f"calibrations={self.n_calibrations}, dt={self.dt}ns)"
        )
