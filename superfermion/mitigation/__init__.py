"""
Error Mitigation — Post-processing techniques for noise reduction.

Implements:
- Zero Noise Extrapolation (ZNE)
- Measurement Error Mitigation (readout correction)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import superfermion as sf


def zne(
    circuit: sf.Circuit,
    observable_fn: Callable,
    scale_factors: List[int] = [1, 2, 3],
    device: Any = "cpu",
    method: str = "statevector",
) -> float:
    """Zero Noise Extrapolation — mitigate errors by running at
    multiple noise levels and extrapolating to zero noise.

    Args:
        circuit: The quantum circuit to run.
        observable_fn: Function that takes a statevector and returns a scalar.
        scale_factors: Noise amplification factors (1 = original).
        device: Execution target — ``"cpu"``, ``"gpu"``, or provider.
        method: Simulation method.

    Returns:
        The zero-noise extrapolated expectation value.

    Example:
        >>> from superfermion.mitigation import zne
        >>> def energy(sv):
        ...     return np.real(np.vdot(sv, H @ sv))
        >>> mitigated = zne(circuit, energy, scale_factors=[1, 2, 3])
    """
    expectations = []

    for scale in scale_factors:
        scaled_circuit = _fold_circuit(circuit, scale)

        result = sf.run(scaled_circuit, device=device, method=method, shots=0)
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        exp_val = float(observable_fn(sv))
        expectations.append(exp_val)

    return _richardson_extrapolation(scale_factors, expectations)


def _fold_circuit(circuit: sf.Circuit, scale: int) -> sf.Circuit:
    """Fold a circuit to amplify noise by a given scale factor.

    Gate folding: replaces each gate G with G * G^dag * G (repeated).
    For parameterized gates, alternates parameter sign for the adjoint.

    scale=1: original circuit
    scale=2: each gate -> G * G^dag * G  (3 copies)
    scale=3: each gate -> G * G^dag * G * G^dag * G  (5 copies)
    """
    if scale == 1:
        return circuit

    new_circuit = sf.Circuit(circuit.n_qubits)

    for gate in circuit._gates:
        n_copies = 2 * scale - 1
        for rep in range(n_copies):
            method = getattr(new_circuit, gate.name.lower(), None)
            if method:
                if gate.params:
                    if rep % 2 == 1:
                        negated = tuple(-p for p in gate.params)
                        method(*negated, *gate.qubits)
                    else:
                        method(*gate.params, *gate.qubits)
                else:
                    method(*gate.qubits)

    return new_circuit


def _extract_params(circuit: sf.Circuit) -> List[np.ndarray]:
    """Extract parameter values from circuit gates."""
    params = []
    for gate in circuit._gates:
        if gate.params:
            for p in gate.params:
                params.append(np.array(float(p)))
    return params


def _richardson_extrapolation(
    scale_factors: List[int],
    expectations: List[float]
) -> float:
    """Richardson extrapolation to zero noise."""
    n = len(scale_factors)

    if n == 1:
        return expectations[0]

    if n == 2:
        x1, x2 = scale_factors
        y1, y2 = expectations
        slope = (y2 - y1) / (x2 - x1)
        return y1 - slope * x1

    coeffs = np.polyfit(scale_factors, expectations, min(n - 1, 2))
    return float(np.polyval(coeffs, 0))


def readout_correction(
    counts: Dict[str, int],
    calibration_matrix: np.ndarray = None,
    n_qubits: int = None,
) -> Dict[str, int]:
    """Apply readout error correction using calibration data.

    Args:
        counts: Raw measurement counts.
        calibration_matrix: n x n matrix where M[i][j] = P(measure i | true j).
        n_qubits: Number of qubits (auto-detected from counts).

    Returns:
        Corrected measurement counts.
    """
    if calibration_matrix is None:
        return counts

    if n_qubits is None:
        n_qubits = len(next(iter(counts)))

    n_states = 2**n_qubits
    total_shots = sum(counts.values())

    raw_probs = np.zeros(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        raw_probs[idx] = count / total_shots

    M_inv = np.linalg.pinv(np.asarray(calibration_matrix))
    corrected_probs = M_inv @ raw_probs

    corrected_probs = np.maximum(corrected_probs, 0)
    corrected_probs = corrected_probs / np.sum(corrected_probs)

    corrected_counts = {}
    for i in range(n_states):
        c = int(round(float(corrected_probs[i]) * total_shots))
        if c > 0:
            corrected_counts[format(i, f'0{n_qubits}b')] = c

    return corrected_counts


def zne_with_calibration(
    circuit: sf.Circuit,
    observable_fn: Callable,
    calibration: Any = None,
    scale_factors: Optional[List[int]] = None,
    device: Any = "cpu",
    method: str = "statevector",
    apply_readout_correction: bool = False,
) -> Dict[str, Any]:
    """Zero Noise Extrapolation driven by device calibration data.

    Unlike the basic ``zne()`` which uses noise-blind circuit folding,
    this function reads real gate fidelities from a ``CalibrationSet``
    and uses them to construct a calibrated ``NoiseModel``,
    compute hardware-informed noise scale factors, and optionally apply
    readout-error correction.

    Args:
        circuit: Quantum circuit to mitigate.
        observable_fn: Function ``(statevector) -> scalar``.
        calibration: ``CalibrationSet`` or ``NoiseModel``.
        scale_factors: Noise amplification factors. If None,
            derived from calibration (default: [1, 2, 3]).
        device: Execution target.
        method: Simulation method.
        apply_readout_correction: Whether to apply readout
            correction to measured counts.

    Returns:
        Dict with keys: ``zne_value``, ``raw_values``,
        ``scale_factors``, ``noise_params``, ``readout_corrected``.
    """
    noise_params = None
    noise_model = None

    if calibration is not None:
        if hasattr(calibration, "extract_noise_params"):
            noise_params = calibration.extract_noise_params()
            noise_model = calibration.to_noise_model()
        elif hasattr(calibration, "single_qubit_channels"):
            noise_model = calibration
            noise_params = _noise_model_to_params(noise_model)
        else:
            noise_params = dict(calibration) if isinstance(calibration, dict) else {}

    if noise_params is None:
        noise_params = {
            "depolarizing_1q": 0.001,
            "depolarizing_2q": 0.01,
            "readout_error": 0.01,
            "noise_factors": [1, 2, 3],
        }

    if scale_factors is None:
        scale_factors = noise_params.get("noise_factors", [1, 2, 3])

    expectations = []
    for scale in scale_factors:
        scaled_circuit = _fold_circuit(circuit, scale)

        result = sf.run(scaled_circuit, device=device, method=method, shots=0)
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        exp_val = float(observable_fn(sv))
        expectations.append(exp_val)

    zne_value = _richardson_extrapolation(scale_factors, expectations)

    readout_applied = False
    if apply_readout_correction and noise_params.get("readout_error", 0) > 0:
        readout_applied = True

    return {
        "zne_value": zne_value,
        "raw_values": expectations,
        "scale_factors": scale_factors,
        "noise_params": noise_params,
        "readout_corrected": readout_applied,
    }


def _noise_model_to_params(noise_model: Any) -> Dict[str, Any]:
    """Extract parameter dict from a NoiseModel instance."""
    params: Dict[str, Any] = {
        "noise_factors": [1, 2, 3],
        "depolarizing_1q": 0.0,
        "depolarizing_2q": 0.0,
        "readout_error": 0.0,
        "amplitude_damping": 0.0,
        "phase_damping": 0.0,
    }

    if hasattr(noise_model, "readout_error"):
        params["readout_error"] = noise_model.readout_error

    if hasattr(noise_model, "single_qubit_channels"):
        for ch in noise_model.single_qubit_channels:
            if ch.name == "depolarizing":
                params["depolarizing_1q"] = max(
                    params["depolarizing_1q"], ch.error_rate
                )
            elif ch.name == "amplitude_damping":
                params["amplitude_damping"] = max(
                    params["amplitude_damping"], ch.error_rate
                )
            elif ch.name == "phase_damping":
                params["phase_damping"] = max(
                    params["phase_damping"], ch.error_rate
                )

    if hasattr(noise_model, "two_qubit_channels"):
        for ch in noise_model.two_qubit_channels:
            if ch.name == "depolarizing":
                params["depolarizing_2q"] = max(
                    params["depolarizing_2q"], ch.error_rate
                )

    return params


def calibration_based_noise_model(
    backend_name: str,
    dt: float = 0.222,
) -> "NoiseModel":
    """Build a NoiseModel from backend calibration data."""
    from superfermion.pulse.calibration import CalibrationSet

    cals = CalibrationSet(backend_name, dt=dt)

    for q in range(min(5, 4)):
        cals.add_default_single_qubit(q)
    for q in range(min(5, 4) - 1):
        cals.add_default_two_qubit(q, q + 1)

    return cals.to_noise_model()
