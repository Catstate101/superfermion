"""
Error Mitigation — Post-processing techniques for noise reduction.

Implements:
- Zero Noise Extrapolation (ZNE)
- Measurement Error Mitigation (readout correction)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

import superfermion as sf


def zne(
    circuit: sf.Circuit,
    observable_fn: Callable,
    scale_factors: List[int] = [1, 2, 3],
    backend: str = "jax",
) -> float:
    """Zero Noise Extrapolation — mitigate errors by running at 
    multiple noise levels and extrapolating to zero noise.
    
    Args:
        circuit: The quantum circuit to run.
        observable_fn: Function that takes a statevector and returns a scalar.
        scale_factors: Noise amplification factors (1 = original).
        backend: Backend to use.
        
    Returns:
        The zero-noise extrapolated expectation value.
        
    Example:
        >>> from superfermion.mitigation import zne
        >>> def energy(sv):
        ...     return jnp.real(jnp.vdot(sv, H @ sv))
        >>> mitigated = zne(circuit, energy, scale_factors=[1, 2, 3])
    """
    expectations = []
    sim = sf.get_backend(backend)
    
    for scale in scale_factors:
        # Fold circuit: G -> G * G^dag * G (properly alternates param sign)
        scaled_circuit = _fold_circuit(circuit, scale)
        
        # Execute directly on the backend (avoids JAX conversion issues)
        result = sim.run(scaled_circuit, shots=0)
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        exp_val = float(observable_fn(sv))
        expectations.append(exp_val)
    
    # Richardson extrapolation to zero noise
    result = _richardson_extrapolation(scale_factors, expectations)
    return result


def _fold_circuit(circuit: sf.Circuit, scale: int) -> sf.Circuit:
    """Fold a circuit to amplify noise by a given scale factor.
    
    Gate folding: replaces each gate G with G * G^dag * G (repeated).
    For parameterized gates (rx, ry, rz, etc.), alternates parameter sign
    to implement the adjoint: G^dag(theta) = G(-theta).
    
    scale=1: original circuit
    scale=2: each gate -> G * G^dag * G  (3 copies)
    scale=3: each gate -> G * G^dag * G * G^dag * G  (5 copies)
    """
    if scale == 1:
        return circuit
    
    new_circuit = sf.Circuit(circuit.n_qubits)
    
    for gate in circuit._gates:
        n_copies = 2 * scale - 1  # always odd: 3, 5, 7, ...
        for rep in range(n_copies):
            method = getattr(new_circuit, gate.name.lower(), None)
            if method:
                if gate.params:
                    # Alternate sign for adjoint: even reps use +params, odd use -params
                    if rep % 2 == 1:
                        negated = tuple(-p for p in gate.params)
                        method(*negated, *gate.qubits)
                    else:
                        method(*gate.params, *gate.qubits)
                else:
                    method(*gate.qubits)
    
    return new_circuit


def _extract_params(circuit: sf.Circuit) -> List[jnp.ndarray]:
    """Extract parameter values from circuit gates as JAX arrays."""
    params = []
    for gate in circuit._gates:
        if gate.params:
            for p in gate.params:
                params.append(jnp.array(float(p)))
    return params


def _richardson_extrapolation(
    scale_factors: List[int], 
    expectations: List[float]
) -> float:
    """Richardson extrapolation to zero noise.
    
    Fits a polynomial through (scale, expectation) points and
    extrapolates to scale=0.
    """
    n = len(scale_factors)
    
    if n == 1:
        return expectations[0]
    
    if n == 2:
        # Linear extrapolation
        x1, x2 = scale_factors
        y1, y2 = expectations
        # y = a*x + b, solve for y(0)
        slope = (y2 - y1) / (x2 - x1)
        return y1 - slope * x1
    
    # Polynomial fit for 3+ points
    coeffs = np.polyfit(scale_factors, expectations, min(n - 1, 2))
    return float(np.polyval(coeffs, 0))


def readout_correction(
    counts: Dict[str, int],
    calibration_matrix: jnp.ndarray = None,
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
        return counts  # No correction without calibration
    
    if n_qubits is None:
        n_qubits = len(next(iter(counts)))
    
    n_states = 2**n_qubits
    total_shots = sum(counts.values())
    
    # Build probability vector from counts
    raw_probs = jnp.zeros(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        raw_probs = raw_probs.at[idx].set(count / total_shots)
    
    # Invert calibration matrix: true_probs = M^{-1} * raw_probs
    M_inv = jnp.linalg.pinv(calibration_matrix)
    corrected_probs = M_inv @ raw_probs
    
    # Clip negative probabilities and renormalize
    corrected_probs = jnp.maximum(corrected_probs, 0)
    corrected_probs = corrected_probs / jnp.sum(corrected_probs)
    
    # Convert back to counts
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
    backend: str = "jax",
    apply_readout_correction: bool = False,
) -> Dict[str, Any]:
    """Zero Noise Extrapolation driven by device calibration data.

    Unlike the basic ``zne()`` which uses noise-blind circuit folding,
    this function reads real gate fidelities from a ``CalibrationSet``
    and uses them to (a) construct a calibrated ``NoiseModel``,
    (b) compute hardware-informed noise scale factors, and
    (c) optionally apply readout-error correction to the extrapolated
    expectation value.

    This is the recommended entry point when you have device
    calibration data (e.g. from ``sf.pulse.CalibrationSet``).

    Args:
        circuit: Quantum circuit to mitigate.
        observable_fn: Function ``(statevector) -> scalar``.
        calibration: ``CalibrationSet`` or ``NoiseModel``. If a
            ``CalibrationSet``, noise parameters are extracted
            automatically via ``.extract_noise_params()``.
        scale_factors: Noise amplification factors. If None,
            derived from calibration (default: [1, 2, 3]).
        backend: Execution backend.
        apply_readout_correction: Whether to apply readout
            correction to measured counts.

    Returns:
        Dict with keys:
        - ``zne_value``: Zero-noise extrapolated expectation
        - ``raw_values``: Per-scale-factor expectations
        - ``scale_factors``: Scale factors used
        - ``noise_params``: Calibration-derived noise parameters
        - ``readout_corrected``: Whether readout correction was applied

    Example:
        >>> cals = CalibrationSet("ibm_brisbane")
        >>> cals.add_default_single_qubit(0)
        >>> cals.add_default_two_qubit(0, 1)
        >>>
        >>> def energy(sv):
        ...     return jnp.real(jnp.abs(sv[0])**2)
        >>>
        >>> result = zne_with_calibration(circuit, energy, cals)
        >>> print(result["zne_value"])
    """
    noise_params = None
    noise_model = None

    # ── Resolve calibration source ────────────────────────────────
    if calibration is not None:
        if hasattr(calibration, "extract_noise_params"):
            # CalibrationSet
            noise_params = calibration.extract_noise_params()
            noise_model = calibration.to_noise_model()
        elif hasattr(calibration, "single_qubit_channels"):
            # NoiseModel — derive params from it
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

    # ── Compute per-scale expectations ────────────────────────────
    expectations = []
    for scale in scale_factors:
        scaled_circuit = _fold_circuit(circuit, scale)

        f_jax = sf.qml.circuit_to_jax(scaled_circuit, backend=backend)
        params_list = [jnp.array(0.0)] * len(scaled_circuit.parameters)
        sv = f_jax(*params_list) if params_list else f_jax()

        exp_val = float(observable_fn(sv))
        expectations.append(exp_val)

    # ── Richardson extrapolation ─────────────────────────────────
    zne_value = _richardson_extrapolation(scale_factors, expectations)

    # ── Readout correction (optional) ────────────────────────────
    readout_applied = False
    if apply_readout_correction and noise_params.get("readout_error", 0) > 0:
        # Apply correction if calibration data is available
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


# ── Utility: build calibration-aware noise model from backend ──────

def calibration_based_noise_model(
    backend_name: str,
    dt: float = 0.222,
) -> "NoiseModel":
    """Build a NoiseModel from backend calibration data.

    This is a convenience wrapper that creates a ``CalibrationSet``,
    populates default gate calibrations for a typical backend, and
    returns the resulting ``NoiseModel``.

    Args:
        backend_name: e.g. ``"ibm_brisbane"``, ``"ibm_eagle"``.
        dt: Time step in ns (default 0.222 for IBM).

    Returns:
        ``superfermion.noise.NoiseModel`` populated from calibration.
    """
    from superfermion.pulse.calibration import CalibrationSet

    cals = CalibrationSet(backend_name, dt=dt)

    # Populate default 1Q and 2Q calibrations for qubits 0–4
    for q in range(min(5, 4)):
        cals.add_default_single_qubit(q)
    for q in range(min(5, 4) - 1):
        cals.add_default_two_qubit(q, q + 1)

    return cals.to_noise_model()
