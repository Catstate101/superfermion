"""
Superfermion Utils — Exceptions, validation, and common utilities.
"""

from superfermion.utils.exceptions import (
    SuperfermionError,
    CircuitError,
    QubitIndexError,
    ParameterError,
    UnboundParameterError,
    BackendError,
    BackendNotFoundError,
    CompilationError,
    OptimizationError,
    ConvergenceError,
    SerializationError,
    HardwareError,
    NoiseModelError,
    GateNotSupportedError,
    ProviderNotConnectedError,
)

from superfermion.utils.validation import (
    validate_n_qubits,
    validate_qubit_index,
    validate_statevector,
    validate_probability,
    validate_shots,
    validate_angle,
)

from superfermion.utils.logging import logger, info, debug, warning, error, set_level

from superfermion.utils.encoding import ensure_utf8

__all__ = [
    "SuperfermionError", "CircuitError", "QubitIndexError",
    "ParameterError", "UnboundParameterError",
    "BackendError", "BackendNotFoundError",
    "CompilationError", "OptimizationError", "ConvergenceError",
    "SerializationError", "HardwareError", "NoiseModelError",
    "GateNotSupportedError", "ProviderNotConnectedError",
    "validate_n_qubits", "validate_qubit_index", "validate_statevector",
    "validate_probability", "validate_shots", "validate_angle",
    "logger", "info", "debug", "warning", "error", "set_level",
        "ensure_utf8"
]
