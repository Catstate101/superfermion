"""Python-side memory guards for exact-state simulation.

A ``method='statevector'`` run at 40 qubits asks the Rust core for ``2**40``
complex amplitudes -- 16 TiB -- and Rust's global allocator **aborts** the
process instead of unwinding, so the caller never sees a Python exception:
the interpreter dies with exit code ``0xC0000409`` (``3221226505``) and any
surrounding ``try``/``except`` is powerless.  The guard here estimates the
allocation *before* it is attempted and raises a catchable ``MemoryError``.

The ceiling is the host's **total** physical RAM, not its currently free
RAM: a state that cannot fit even in theory is refused, while anything that
could conceivably fit is left to the allocator so existing workloads keep
their behaviour.  The ceiling can be overridden per run with the
``max_state_gib`` keyword or globally with ``SF_STATE_MEMORY_LIMIT_GIB``
(``0`` disables the guard entirely).

Methods whose cost is not ``2**n`` -- ``mps`` and ``stabilizer`` -- are never
guarded here; they have their own caps.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional

__all__ = [
    "STATE_BYTES_PER_AMPLITUDE",
    "bytes_for_state",
    "guard_state_allocation",
    "limit_from_kwargs",
    "physical_memory_bytes",
    "state_memory_limit_bytes",
]

#: One ``complex128`` amplitude.  Every exact-state path in sf-ir stores the
#: state as a contiguous ``Vec<Complex64>``.
STATE_BYTES_PER_AMPLITUDE = 16

#: Used only when the platform cannot report its own RAM.
_FALLBACK_LIMIT_BYTES = 64 * 2**30

_DENSITY_MATRIX = "density_matrix"
_UNGUARDED_METHODS = frozenset({"mps", "stabilizer"})
#: Methods whose state is indexed by *two* qubit registers, i.e. ``4**n``.
_SQUARE_STATES = frozenset({_DENSITY_MATRIX, "unitary"})


def bytes_for_state(
    n_qubits: Any, method: str = "statevector"
) -> Optional[int]:
    """Bytes an exact state needs, or ``None`` when no ``2**n`` state exists.

    ``statevector`` holds ``2**n`` amplitudes and ``density_matrix`` holds
    ``4**n``, both as ``complex128``; ``unitary`` is a ``4**n`` matrix with the
    same element size.  ``mps`` and ``stabilizer`` do not materialise such a
    vector, so they return ``None``.

    Args:
        n_qubits: Register width.
        method: Simulation method name.

    Returns:
        Size in bytes, or ``None`` for tensor-network / tableau methods.
    """
    try:
        n = int(n_qubits)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return STATE_BYTES_PER_AMPLITUDE
    name = str(method or "statevector").strip().lower()
    if name in _UNGUARDED_METHODS:
        return None
    exponent = 2 * n if name in _SQUARE_STATES else n
    # ``16 << e`` == 16 * 2**e; Python ints do not overflow, so a 40-qubit
    # request simply becomes a 17 592 186 044 416-byte number.
    return STATE_BYTES_PER_AMPLITUDE << exponent


def physical_memory_bytes() -> Optional[int]:
    """Total physical RAM of this host, or ``None`` when it cannot be read.

    Reads ``GlobalMemoryStatusEx`` on Windows and ``sysconf`` on POSIX
    without pulling in ``psutil``.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            )
            if ok:
                return int(status.ullTotalPhys)
        except Exception:
            return None
        return None

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def state_memory_limit_bytes() -> float:
    """Ceiling for a single exact-state allocation, in bytes.

    ``SF_STATE_MEMORY_LIMIT_GIB`` wins when set (``0`` means "no ceiling");
    otherwise the host's total physical RAM is used, falling back to 64 GiB
    when the platform will not say.  ``math.inf`` means "never refuse".
    """
    raw = os.environ.get("SF_STATE_MEMORY_LIMIT_GIB")
    if raw:
        try:
            gib = float(raw)
        except (TypeError, ValueError):
            gib = None
        if gib is not None:
            if gib <= 0:
                return math.inf
            return gib * 2**30
    total = physical_memory_bytes()
    if total is None or total <= 0:
        return float(_FALLBACK_LIMIT_BYTES)
    return float(total)


def limit_from_kwargs(kwargs: Optional[Dict[str, Any]]) -> Optional[float]:
    """Per-run ceiling from the ``max_state_gib`` keyword, if given.

    ``max_state_gib=0`` disables the guard for that run.  ``None`` (the
    default) defers to :func:`state_memory_limit_bytes`.
    """
    if not kwargs:
        return None
    raw = kwargs.get("max_state_gib")
    if raw is None:
        return None
    try:
        gib = float(raw)
    except (TypeError, ValueError):
        return None
    if gib <= 0:
        return math.inf
    return gib * 2**30


def guard_state_allocation(
    n_qubits: Any,
    method: str = "statevector",
    *,
    limit_bytes: Optional[float] = None,
    what: str = "simulation",
) -> Optional[int]:
    """Raise ``MemoryError`` when an exact state cannot possibly fit.

    Args:
        n_qubits: Register width.
        method: Simulation method name.
        limit_bytes: Ceiling override; ``None`` defers to
            :func:`state_memory_limit_bytes`, ``math.inf`` disables.
        what: Called-out operation name in the error message.

    Returns:
        The estimated size in bytes (``None`` for ``mps``/``stabilizer``).

    Raises:
        MemoryError: when the estimate exceeds the ceiling.
    """
    requested = bytes_for_state(n_qubits, method)
    if requested is None:
        return None
    limit = (
        limit_bytes if limit_bytes is not None else state_memory_limit_bytes()
    )
    if limit is None or limit == math.inf:
        return requested
    if requested <= limit:
        return requested
    try:
        n = int(n_qubits)
    except (TypeError, ValueError):
        n = -1
    raise MemoryError(
        f"{what} with method='{str(method).strip().lower()}' at {n} qubits "
        f"needs {requested / 2**30:,.1f} GiB for the state "
        f"(2^{n} complex128 amplitudes), above the "
        f"{limit / 2**30:,.1f} GiB ceiling in force.  The Rust allocator "
        "aborts the whole process when that allocation fails, so this is "
        "refused up front instead.\n"
        "  Fix: method='mps' (bond_dim=...) for large weakly-entangled "
        "circuits, method='stabilizer' for Clifford circuits, "
        "return_statevector=False to sample without keeping the state, or "
        "raise the ceiling with max_state_gib= / SF_STATE_MEMORY_LIMIT_GIB."
    )


def validate_bond_dim(bond_dim: Any) -> int:
    """Return ``bond_dim`` as an ``int``, refusing values the MPS cannot use.

    ``bond_dim=0`` is not "no truncation".  The QR step caps the bond at
    ``min(bond_dim, rank) = 0``, so every bond collapses to a single state and
    the sampler draws from the uniform distribution instead of the circuit:
    measured on a GHZ-3 circuit, ``bond_dim=0`` returned all eight bitstrings
    at ~25 counts each out of 200 shots, where the state has weight on two.
    A negative value never reaches the truncation logic at all -- pyo3 raises
    ``OverflowError: can't convert negative int to unsigned``, which names
    nothing the caller can act on.

    Args:
        bond_dim: Caller-supplied bond dimension.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: when the value is not an integer, or is below 1.
    """
    try:
        value = int(bond_dim)
    except (TypeError, ValueError):
        raise ValueError(
            f"bond_dim must be a positive integer, got {bond_dim!r}"
        ) from None
    if value < 1:
        raise ValueError(
            f"bond_dim must be at least 1, got {value}.  A bond dimension of "
            "0 collapses every MPS bond to a single state, so the result is "
            "sampled from the uniform distribution rather than from the "
            "circuit; negative values are not convertible to a bond index."
        )
    return value
