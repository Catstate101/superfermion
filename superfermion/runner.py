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


class _Unset:
    """Sentinel for "the caller did not pass this keyword at all".

    Used as the default of ``run(seed=...)`` because a plain ``None`` default
    would collapse two different requests: ``seed=None`` asks the engine to
    draw fresh entropy for *this* run, while omitting ``seed`` asks for the
    historical fixed default of 42.  A sentinel keeps them distinguishable and
    lets the parameter appear in ``inspect.signature``, which it could not
    while it was swallowed by ``**kwargs`` (xq8/STAB-3).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


#: ``run(seed=...)``'s default: forward nothing, so the device keeps its own
#: default (42) and every pre-existing call returns the counts it always did.
_SEED_UNSET = _Unset()


def run(
    circuit: Circuit,
    device: Union[str, DeviceExecutor, None] = None,
    shots: int = 1000,
    method: Optional[str] = None,
    target: Optional[str] = None,
    tracker: Optional[Any] = None,
    params: Optional[dict] = None,
    *,
    seed: Any = _SEED_UNSET,
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
        seed: Sampling seed, keyword-only.  ``seed=<int>`` reproduces a sample
            exactly; omitting it uses the historical default of 42, so repeated
            unseeded runs are bit-identical; ``seed=None`` draws a fresh 32-bit
            seed from OS entropy and reports it back as
            ``result.metadata["seed"]`` so the run can be replayed.  See the
            section below.
        **kwargs: Passed through to the device executor (e.g. ``bond_dim``
            for MPS, ``densify=False`` to skip the shots=0 MPS densification
            and receive only the state handle).

    Sampling seed:
        ``seed=<int>`` reproduces a sample exactly.  Omitting it uses the
        historical default of 42, so repeated unseeded runs are bit-identical
        -- repeat runs must vary the seed to build an error bar.  ``seed=None``
        asks for a fresh 32-bit seed drawn from OS entropy; it is reported back
        as ``result.metadata["seed"]`` so the run can be replayed:

            >>> a = sf.run(c, shots=1000, seed=None)
            >>> b = sf.run(c, shots=1000, seed=a.metadata["seed"])
            >>> a.counts == b.counts
            True

    Returns:
        ``RunResult`` with counts, statevector, probabilities, and metadata.

    Raises:
        RuntimeError: If the circuit has unbound symbolic parameters.
        RuntimeError: If GPU is requested but unavailable.
        RuntimeError: If stabilizer method is used on non-Clifford circuit.
        ValueError: If device or method string is unrecognized.
    """
    # 0. Bind parameters if provided
    symbolic_circuit = circuit
    if seed is not _SEED_UNSET:
        # Keep the device-side contract: ``_resolve_seed`` reads kwargs["seed"]
        # and tells "absent" (fixed 42) apart from "None" (draw entropy) by
        # key presence, so the sentinel must not be forwarded.
        kwargs["seed"] = seed
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
    _caps = executor.capabilities()  # noqa: F841 — validated during dispatch

    # Remote devices advertise a fixed width, and the value used to be thrown
    # away: a 5-qubit circuit dispatched to a 2-qubit stub device and came back
    # with ``{'00000': 8}``.  Only enforce this for hardware (simulators size
    # themselves per run); ``max_qubits <= 0`` means "unknown, do not gate".
    _max_qubits = int(getattr(_caps, "max_qubits", 0) or 0)
    if (
        not getattr(_caps, "is_simulator", True)
        and _max_qubits > 0
        and exec_circuit.n_qubits > _max_qubits
    ):
        raise ValueError(
            f"Circuit needs {exec_circuit.n_qubits} qubits but device "
            f"'{_device_label(device)}' supports at most {_max_qubits}."
        )

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

    # RES-2: ``bind()`` substitutes the numbers, so the symbolic names are
    # gone from ``result.circuit``.  Keep the symbolic circuit and the values
    # that were bound on the result itself (not in ``metadata``, which is
    # serialised) so ``RunResult.grad`` can rebuild the parameterised IR
    # without the caller passing a dag.
    try:
        result._sf_grad_context = {
            "circuit": symbolic_circuit,
            "params": dict(params) if params is not None else {},
        }
    except Exception:  # pragma: no cover - exotic result types
        pass

    return result


def simulate(circuit: Circuit, **kwargs: Any):
    """Convenience shortcut: run a circuit and return the State directly.

    Equivalent to ``sf.run(circuit, **kwargs).state``.

    Args:
        circuit: The quantum circuit to simulate.
        **kwargs: Passed through to ``sf.run()`` (device, method, shots, etc.).

    Returns:
        ``sf.State`` — the Rust-native quantum state handle.

    Raises:
        RuntimeError: If the execution doesn't produce a state (e.g. QPU).
    """
    result = run(circuit, **kwargs)
    if result.state is None:
        raise RuntimeError(
            "simulate() requires a simulation device that produces a state. "
            "Got no state — use sf.run() for QPU results."
        )
    return result.state


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
