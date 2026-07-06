"""
Canonical backend names enum for Superfermion.

Usage:
    >>> sf.run(circuit, backend=BackendName.JAX)
    >>> sf.get_backend(BackendName.RUST)
"""

from __future__ import annotations

import warnings
from enum import Enum


class BackendName(str, Enum):
    """Canonical backend identifiers.

    Inherits ``str`` so existing string-based code works transparently:
    ``BackendName.JAX == "jax"`` is True.
    """

    # -- Simulators --
    STATEVECTOR = "statevector"
    JAX = "jax"
    RUST = "rust"
    CUDA = "cuda"
    CUPY = "cupy"
    MPS = "mps"
    CLUSTER = "cluster"
    JAX_MPS = "jax_mps"
    CUDA_MPS = "cuda_mps"
    SINGULARITY = "singularity"
    SUPREMACY = "supremacy"
    DENSITY_MATRIX = "density_matrix"
    STABILIZER = "stabilizer"

    # -- Hardware providers --
    DWAVE = "dwave"
    IBM = "ibm"

    # -- Convenience aliases --
    SIMULATOR = "simulator"
    AUTO = "auto"


# Deprecated aliases — preserved with warning for backwards compatibility.
_DEPRECATED: dict[str, BackendName] = {
    "god": BackendName.SINGULARITY,
    "lightning": BackendName.SINGULARITY,
    "omnipotent": BackendName.SINGULARITY,
}


def resolve_backend_name(name: BackendName | str) -> BackendName:
    """Resolve a backend name (enum value or string) to canonical BackendName.

    Deprecated aliases emit a warning and resolve to the canonical name.
    """
    if isinstance(name, BackendName):
        return name
    if name in _DEPRECATED:
        warnings.warn(
            f"Backend name '{name}' is deprecated. Use '{_DEPRECATED[name].value}' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return _DEPRECATED[name]
    try:
        return BackendName(name)
    except ValueError:
        raise ValueError(
            f"Unknown backend: '{name}'. Valid backends: {[b.value for b in BackendName]}"
        )
