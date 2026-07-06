"""
GPU-accelerated statevector simulator.

Note: This backend extends CuPyBackend (CuPy-based) with additional
cuda-specific optimizations, not the NVIDIA cuQuantum SDK. For full
cuQuantum integration, use the ``cupy`` backend directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from superfermion.backends.base import Backend
from superfermion.circuit import Circuit
from superfermion.runner import RunResult


from superfermion.backends.cupy_sim import CupyBackend

class CUSimulatorBackend(CupyBackend):
    """NVIDIA cuQuantum/CuPy backend for high-performance GPU simulation."""

    def __init__(self, name: str = "cuda", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        self._n_max_qubits = 32
