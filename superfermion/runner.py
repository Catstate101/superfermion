"""
sf.run() — the single entry point for quantum circuit execution.

Usage::

    import superfermion as sf

    circuit = sf.Circuit(2).h(0).cnot(0, 1)

    # Local CPU simulation (default)
    result = sf.run(circuit, shots=1000)
    print(result.counts)  # {'00': 503, '11': 497}

    # GPU simulation
    result = sf.run(circuit, device="gpu")

    # Simulation method control
    result = sf.run(circuit, method="mps", bond_dim=128)  # tensor network
    result = sf.run(circuit, method="stabilizer")          # Clifford only

    # QPU via provider objects
    from superfermion.devices.ibm import IBMDevice
    ibm = IBMDevice(token="...")
    result = sf.run(circuit, device=ibm("ibm_fez"))
"""

from __future__ import annotations

from typing import Any, Optional, Union

from superfermion.circuit import Circuit
from superfermion.results import RunResult
from superfermion.devices import DeviceExecutor


def run(
    circuit: Circuit,
    device: Union[str, DeviceExecutor, None] = None,
    shots: int = 1000,
    method: Optional[str] = None,
    target: Optional[str] = None,
    tracker: Optional[Any] = None,
    params: Optional[dict] = None,
    **kwargs: Any,
) -> RunResult:
    """Execute a quantum circuit and return measurement results.

    Args:
        circuit: The quantum circuit to execute.
        device: Where to run. Either ``"cpu"`` (default), ``"gpu"``, or
            a ``DeviceExecutor`` object (e.g. ``IBMDevice``, ``IonQDevice``).
        shots: Number of measurement repetitions.
        method: Simulation algorithm for local devices. One of:
            ``"statevector"`` (default) — exact simulation, 2^n memory.
            ``"mps"`` — tensor network, for large weakly-entangled circuits.
            ``"stabilizer"`` — Clifford-only, exponentially fast.
            Ignored when ``device`` is a QPU (DeviceExecutor object).
        target: Optional hardware target name (e.g. ``"ionq_aria"``).
            If provided, the circuit is compiled to the target's basis gates
            and topology before execution.
        tracker: An explicit ``TrackerProtocol`` object. If omitted, the
            runner checks for an active ``sf.experiment()`` context.
        params: Parameter values for symbolic circuits.  If provided,
            ``circuit.bind(params)`` is called automatically.
        **kwargs: Passed through to the device executor (e.g. ``bond_dim``
            for MPS, ``seed`` for sampling).

    Returns:
        ``RunResult`` with counts, statevector, probabilities, and metadata.

    Raises:
        RuntimeError: If the circuit has unbound symbolic parameters.
        RuntimeError: If GPU is requested but unavailable.
        RuntimeError: If stabilizer method is used on non-Clifford circuit.
        ValueError: If device or method string is unrecognized.
    """
    # 0. Bind parameters if provided
    if params is not None:
        circuit = circuit.bind(params)

    # 1. Parameter validation
    if circuit.n_parameters > 0:
        unbound = circuit.parameters
        raise RuntimeError(
            f"Circuit has {len(unbound)} unbound parameter(s): {unbound}\n"
            f"  Fix: Call sf.run(circuit, params={{{unbound[0]!r}: 0.5, ...}})"
        )

    # 2. Resolve device
    executor = _resolve_device(device, method)

    # 3. Hardware-aware compilation
    exec_circuit = circuit
    if target:
        exec_circuit = _compile_for_target(circuit, target)

    # 4. Gate fusion is handled in Rust during simulation
    caps = executor.capabilities()

    # 5. Resolve tracker (explicit > context > None)
    active_tracker = tracker
    if active_tracker is None:
        from superfermion.experiment.context import _get_active_tracker
        active_tracker = _get_active_tracker()

    # 6. Execute with tracker lifecycle
    device_label = _device_label(device)
    if active_tracker is not None:
        active_tracker.on_run_start(exec_circuit, device_label, shots)

    try:
        result = executor.execute(exec_circuit, shots=shots, **kwargs)
    except Exception as exc:
        if active_tracker is not None:
            active_tracker.on_run_error(exc)
        raise

    if active_tracker is not None:
        active_tracker.on_run_complete(result)

    return result


def _resolve_device(
    device: Union[str, DeviceExecutor, None],
    method: Optional[str] = None,
) -> DeviceExecutor:
    """Convert device + method arguments into a ``DeviceExecutor``."""
    if device is None:
        device = "cpu"

    # If it's already a DeviceExecutor object (IBMDevice, IonQDevice, etc.)
    if isinstance(device, DeviceExecutor):
        return device

    if isinstance(device, str):
        # For string devices ("cpu", "gpu"), apply method= parameter
        from superfermion.devices.rust_device import RustDevice
        hardware = device.lower().strip()
        sim_method = method or "statevector"
        return RustDevice(hardware=hardware, method=sim_method)

    raise TypeError(
        f"device= must be 'cpu', 'gpu', or a DeviceExecutor object, got {type(device).__name__}"
    )


def _compile_for_target(circuit: Circuit, target: str) -> Circuit:
    """Compile a circuit for a hardware target."""
    from superfermion.compiler.specs import get_spec
    from superfermion.compiler.manager import compile as sf_compile

    spec = get_spec(target)
    if spec is None:
        raise ValueError(f"Hardware target '{target}' not found in compiler/specs.py")
    return sf_compile(circuit, target=spec)


def _device_label(device: Union[str, DeviceExecutor, None]) -> str:
    """Human-readable label for tracker metadata."""
    if device is None:
        return "cpu"
    if isinstance(device, str):
        return device
    return repr(device)
