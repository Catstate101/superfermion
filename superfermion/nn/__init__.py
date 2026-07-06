"""
Quantum Neural Networks (QNN) — Hybrid quantum-classical layers.

Heavy dependencies (flax, torch, tensorflow) are loaded lazily on first access.
"""

from __future__ import annotations

__all__ = [
    "QuantumLayer",
    "TorchQuantumLayer", "torch_quantum_layer",
    "TFQuantumLayer", "tf_quantum_layer",
]

_LAZY_MAP = {
    "QuantumLayer": "superfermion.nn.quantum_layer",
}


def _try_load_torch():
    """Lazy-load TorchQuantumLayer (requires torch)."""
    try:
        from superfermion.nn.torch_layer import TorchQuantumLayer, torch_quantum_layer
        return TorchQuantumLayer, torch_quantum_layer
    except ImportError:
        return None, None

def _try_load_tf():
    """Lazy-load TFQuantumLayer (requires tensorflow)."""
    try:
        from superfermion.nn.tf_layer import TFQuantumLayer, tf_quantum_layer
        return TFQuantumLayer, tf_quantum_layer
    except ImportError:
        return None, None


def __getattr__(name):
    if name == "_LAZY_MAP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "TorchQuantumLayer":
        TorchQL, _ = _try_load_torch()
        if TorchQL is None:
            raise ImportError("TorchQuantumLayer requires torch. Install with: pip install torch")
        globals()["TorchQuantumLayer"] = TorchQL
        return TorchQL

    if name == "torch_quantum_layer":
        _, tql = _try_load_torch()
        if tql is None:
            raise ImportError("torch_quantum_layer requires torch. Install with: pip install torch")
        globals()["torch_quantum_layer"] = tql
        return tql

    if name == "TFQuantumLayer":
        TFQL, _ = _try_load_tf()
        if TFQL is None:
            raise ImportError("TFQuantumLayer requires tensorflow. Install with: pip install tensorflow")
        globals()["TFQuantumLayer"] = TFQL
        return TFQL

    if name == "tf_quantum_layer":
        _, tql = _try_load_tf()
        if tql is None:
            raise ImportError("tf_quantum_layer requires tensorflow. Install with: pip install tensorflow")
        globals()["tf_quantum_layer"] = tql
        return tql

    mod_name = _LAZY_MAP.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(mod_name)
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
