"""
Superfermion Exceptions — Structured error hierarchy.
"""


class SuperfermionError(Exception):
    """Base exception for all Superfermion errors."""
    pass


class CircuitError(SuperfermionError):
    """Error in circuit construction or validation."""
    pass


class QubitIndexError(CircuitError):
    """Invalid qubit index."""
    def __init__(self, qubit: int, n_qubits: int, gate: str = ""):
        self.qubit = qubit
        self.n_qubits = n_qubits
        msg = f"Qubit index {qubit} is out of range for {n_qubits}-qubit circuit"
        if gate:
            msg += f" (gate: {gate})"
        super().__init__(msg)


class ParameterError(SuperfermionError):
    """Error with circuit parameters."""
    pass


class UnboundParameterError(ParameterError):
    """Circuit has unbound parameters that need values."""
    def __init__(self, params: list):
        super().__init__(
            f"Circuit has {len(params)} unbound parameters: {params}. "
            f"Use circuit.bind({{...}}) to provide values."
        )


class MethodError(SuperfermionError):
    """Operation not supported by the current simulation method.

    Raised when calling an sf.State method that isn't available for the
    underlying representation (e.g. grad() on a stabilizer state).
    """
    pass


class BackendError(SuperfermionError):
    """Error with backend execution."""
    pass


class BackendNotFoundError(BackendError):
    """Requested backend is not available."""
    def __init__(self, name: str, available: list = None):
        msg = f"Backend '{name}' not found."
        if available:
            msg += f" Available: {available}"
        super().__init__(msg)


class CompilationError(SuperfermionError):
    """Error during circuit compilation."""
    pass


class OptimizationError(SuperfermionError):
    """Error during variational optimization."""
    pass


class ConvergenceError(OptimizationError):
    """Algorithm did not converge."""
    def __init__(self, algorithm: str, iterations: int, final_value: float):
        super().__init__(
            f"{algorithm} did not converge after {iterations} iterations. "
            f"Final value: {final_value:.6f}"
        )


class SerializationError(SuperfermionError):
    """Error during circuit serialization/deserialization."""
    pass


class HardwareError(SuperfermionError):
    """Error communicating with quantum hardware."""
    pass


class NoiseModelError(SuperfermionError):
    """Error in noise model configuration."""
    pass


class GateNotSupportedError(CompilationError):
    """A gate is not supported by the target backend or provider.

    Provides actionable suggestions for decomposition or backend switching.
    """
    def __init__(self, gate_name: str, backend_or_provider: str,
                 supported: list = None, workaround: str = ""):
        self.gate_name = gate_name
        self.backend = backend_or_provider
        msg = f"Gate '{gate_name}' is not supported by {backend_or_provider}."
        if supported:
            msg += f"\n  Supported gates: {', '.join(supported)}"
        if workaround:
            msg += f"\n  Workaround: {workaround}"
        else:
            msg += ("\n  Tip: Use sf.compile(circuit, target='...') to auto-decompose "
                    "unsupported gates.")
        super().__init__(msg)


class ProviderNotConnectedError(SuperfermionError):
    """No connection established to a cloud provider."""
    def __init__(self, provider: str, instructions: str = ""):
        msg = f"Not connected to {provider.upper()} provider."
        if not instructions:
            if provider.lower() == "ibm":
                instructions = (
                    "from superfermion.devices.ibm import IBMDevice\n"
                    "  device = IBMDevice(token='YOUR_IBM_TOKEN')"
                )
            elif provider.lower() == "ionq":
                instructions = (
                    "from superfermion.devices.ionq import IonQDevice\n"
                    "  device = IonQDevice(api_key='YOUR_IONQ_KEY')"
                )
            elif provider.lower() == "openquantum":
                instructions = (
                    "from superfermion.devices.openquantum import OpenQuantumDevice\n"
                    "  device = OpenQuantumDevice(client_id='...', client_secret='...')"
                )
        if instructions:
            msg += f"\n  To connect:\n  {instructions}"
        super().__init__(msg)
