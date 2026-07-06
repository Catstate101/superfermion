"""CPU / RAM safety gate. If a (backend, n_qubits) combo would plausibly OOM,
skip it and record the reason rather than crashing the sweep.

Rules:
  - Statevector backends (sf.statevector, sf.jax, sf.rust, qiskit-aer
    statevector, pennylane default.qubit / lightning.qubit) are hard-capped
    at n_qubits <= MAX_STATEVECTOR_QUBITS because 2^n complex128 amplitudes
    grow to tens of GB fast.
  - Additionally, require that the estimated statevector size fit in
    RAM_HEADROOM_FRAC of currently-available RAM.
  - MPS / tensor-network backends (sf.mps, sf.jax_mps, sf.cuda_mps,
    sf.singularity, sf.supremacy, qiskit matrix_product_state) are assumed
    safe up to any n in this sweep because their memory scales in bond
    dim, not 2^n.
"""
from __future__ import annotations

import psutil

MAX_STATEVECTOR_QUBITS = 24
RAM_HEADROOM_FRAC = 0.6  # use at most 60% of available RAM
COMPLEX128_BYTES = 16

STATEVECTOR_BACKENDS = {
    "sf.statevector", "sf.jax", "sf.rust", "sf.cuda",
    "qiskit.statevector", "pennylane.default.qubit", "pennylane.lightning.qubit",
}
TENSORNETWORK_BACKENDS = {
    "sf.mps", "sf.jax_mps", "sf.cuda_mps", "sf.singularity", "sf.supremacy",
    "qiskit.matrix_product_state",
}


def statevector_bytes(n_qubits: int) -> int:
    """Rough peak footprint of a complex128 statevector + 1 work buffer."""
    return 2 * COMPLEX128_BYTES * (2 ** n_qubits)


def available_ram_bytes() -> int:
    return int(psutil.virtual_memory().available)


def would_oom(backend_id: str, n_qubits: int) -> tuple[bool, str]:
    """Return (skip, reason). reason is '' if the combo is safe to run."""
    if backend_id in STATEVECTOR_BACKENDS:
        if n_qubits > MAX_STATEVECTOR_QUBITS:
            return True, f"statevector cap n<={MAX_STATEVECTOR_QUBITS}"
        need = statevector_bytes(n_qubits)
        have = available_ram_bytes()
        if need > RAM_HEADROOM_FRAC * have:
            return True, (
                f"statevector would use {need / 1e9:.2f} GB > "
                f"{RAM_HEADROOM_FRAC:.0%} of {have / 1e9:.2f} GB avail"
            )
    # Tensor-network / unknown: allow
    return False, ""
