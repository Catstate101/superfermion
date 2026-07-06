"""
Circuit execution runner — the main entry point for running circuits.

Usage:
    >>> import superfermion as sf
    >>> circuit = sf.Circuit(2).h(0).cnot(0, 1)
    >>> result = sf.run(circuit, backend="simulator", shots=1000)
    >>> print(result.counts)
    {'00': 503, '11': 497}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

import numpy as np
from numpy.typing import NDArray

from superfermion.circuit import Circuit
from superfermion.results import RunResult
from superfermion.backends.registry import get_backend
from superfermion.backends.names import BackendName


def run(
    circuit: Circuit,
    backend: Optional[str | BackendName] = None,
    shots: int = 1000,
    target: Optional[str] = None,
    **kwargs: Any,
) -> RunResult:
    """Execute a quantum circuit and return measurement results.

    This is the primary entry point for quantum circuit execution. 
    It dispatches to the selected backend via the BackendRegistry.

    Args:
        circuit: The quantum circuit to execute.
        backend: Backend name (e.g., 'statevector', 'cuda', 'jax').
        shots: Number of measurement repetitions.
        target: Optional hardware target (e.g., 'ibm_eagle', 'rigetti_aspen_m3').
                If provided, the circuit is automatically compiled for this target.
        **kwargs: Backend-specific options.

    Returns:
        RunResult object containing the results.
    """
    from superfermion.compiler.manager import compile
    from superfermion.runtime.specs import get_spec
    from superfermion.utils import info, debug

    # 1. Parameter Validation
    if circuit.n_parameters > 0:
        unbound = circuit.parameters
        raise RuntimeError(
            f"Circuit has {len(unbound)} unbound parameter(s): {unbound}\n"
            f"  Fix: Call circuit.bind({{{unbound[0]!r}: 0.5, ...}}) before sf.run()\n"
        )

    # 2. Hardware-Aware Compilation
    exec_circuit = circuit
    if target:
        spec = get_spec(target)
        if spec:
            info(f"Targeting world-class hardware: {target}")
            exec_circuit = compile(circuit, target=spec)
            debug(f"  Optimized circuit depth: {exec_circuit.depth}")
        else:
            from superfermion.utils.exceptions import HardwareError
            raise HardwareError(f"Hardware target '{target}' not found in specs.")

    # 3. Backends choice (resolve BEFORE optimization so we can skip if needed)
    from superfermion.runtime.arbiter import arbiter
    if backend is None:
        backend = "simulator"
        backend = arbiter.select_best_backend(exec_circuit, requested_target=target)
        info(f"Auto-Routing circuit to optimal backend: {backend}")

    # 4. TURBO OPTIMIZATION: Single-pass gate fusion (replaces 3 legacy passes)
    # Merges all consecutive 1Q gates into single U via matrix multiplication.
    # SKIP for stabilizer backend — it needs per-gate Clifford identity;
    # gate fusion destroys gate names and breaks Clifford detection.
    _TURBO_SKIP_BACKENDS = {"stabilizer"}
    if backend not in _TURBO_SKIP_BACKENDS:
        from superfermion.backends.turbo import fuse_single_qubit_gates
        exec_circuit = fuse_single_qubit_gates(exec_circuit)

    target_backend = get_backend(backend)

    # 5. Execute
    return target_backend.run(exec_circuit, shots=shots, **kwargs)
