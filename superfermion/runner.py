"""
sf.run() — the single entry point for quantum circuit execution.

Usage::

    import superfermion as sf

    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    result = sf.run(circuit, device="cpu", shots=1000)
    print(result.counts)  # {'00': 503, '11': 497}

    # With a real QPU device:
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
    target: Optional[str] = None,
    tracker: Optional[Any] = None,
    **kwargs: Any,
) -> RunResult:
    """Execute a quantum circuit and return measurement results.

    This is the primary entry point for all circuit execution. It handles:

    1. Parameter validation (unbound parameters raise ``RuntimeError``)
    2. Hardware-aware compilation (if ``target=`` is set)
    3. Gate fusion (unless the device signals ``skip_fusion``)
    4. Execution via ``DeviceExecutor`` protocol
    5. Tracker lifecycle (``on_run_start`` / ``on_run_complete`` / ``on_run_error``)

    Args:
        circuit: The quantum circuit to execute.
        device: A device string (``"cpu"``, ``"gpu"``, ``"statevector"``,
            ``"rust"``, etc.) or a ``DeviceExecutor`` object.
            Defaults to ``"cpu"`` (SingularityBackend auto-router).
        shots: Number of measurement repetitions.
        target: Optional hardware target name (e.g. ``"ionq_aria"``).
            If provided, the circuit is compiled to the target's basis gates
            and topology before execution.
        tracker: An explicit ``TrackerProtocol`` object. If omitted, the
            runner checks for an active ``sf.experiment()`` context.
        **kwargs: Passed through to the device executor.

    Returns:
        ``RunResult`` with counts, statevector, probabilities, and metadata.

    Raises:
        RuntimeError: If the circuit has unbound symbolic parameters.
    """
    # 1. Parameter validation
    if circuit.n_parameters > 0:
        unbound = circuit.parameters
        raise RuntimeError(
            f"Circuit has {len(unbound)} unbound parameter(s): {unbound}\n"
            f"  Fix: Call circuit.bind({{{unbound[0]!r}: 0.5, ...}}) before sf.run()"
        )

    # 2. Resolve device
    executor = _resolve_device(device)

    # 3. Hardware-aware compilation
    exec_circuit = circuit
    if target:
        exec_circuit = _compile_for_target(circuit, target)

    # 4. Gate fusion (skip if device says so)
    caps = executor.capabilities()
    if not caps.skip_fusion:
        try:
            from superfermion.backends.turbo import fuse_single_qubit_gates
            exec_circuit = fuse_single_qubit_gates(exec_circuit)
        except ImportError:
            pass

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


def _resolve_device(device: Union[str, DeviceExecutor, None]) -> DeviceExecutor:
    """Convert a device argument into a ``DeviceExecutor``."""
    if device is None:
        device = "cpu"

    if isinstance(device, str):
        from superfermion.devices import _resolve_builtin
        return _resolve_builtin(device)

    if isinstance(device, DeviceExecutor):
        return device

    raise TypeError(
        f"device= must be a string or DeviceExecutor, got {type(device).__name__}"
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
