"""QML (Quantum Machine Learning) module for Superfermion.

Dependencies are loaded lazily on first access.
Measurement functions (expval, entropy, fidelity, etc.) are now on sf.State.
"""

__all__ = [
    "parameter_shift_grad",
    "parameter_shift_grad_vector",
    "finite_diff_grad",
    "AngleEmbedding",
    "ZZFeatureMap",
    "BasicEntanglerLayers",
    "StronglyEntanglingLayers",
    "HardwareEfficientAnsatz",
    "TwoLocal",
    "DataReuploadingCircuit",
    "ansatz",
    "encoding",
    "mitigation",
]

_LAZY_MAP = {
    "parameter_shift_grad":        "superfermion.qml.gradient.parameter_shift",
    "parameter_shift_grad_vector": "superfermion.qml.gradient.parameter_shift",
    "finite_diff_grad":            "superfermion.qml.gradient.parameter_shift",
    "AngleEmbedding":         "superfermion.qml.templates",
    "ZZFeatureMap":           "superfermion.qml.templates",
    "BasicEntanglerLayers":   "superfermion.qml.templates",
    "StronglyEntanglingLayers": "superfermion.qml.templates",
    "HardwareEfficientAnsatz": "superfermion.qml.templates",
    "TwoLocal":               "superfermion.qml.templates",
    "DataReuploadingCircuit": "superfermion.qml.templates",
    "ansatz":       ("superfermion.qml.ansatz", None),
    "encoding":     ("superfermion.qml.encoding", None),
    "mitigation":   ("superfermion.mitigation", None),
}


def __getattr__(name):
    if name == "_LAZY_MAP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    entry = _LAZY_MAP.get(name)
    if entry is not None:
        import importlib
        if isinstance(entry, tuple):
            mod_name, attr_name = entry
        else:
            mod_name, attr_name = entry, name
        mod = importlib.import_module(mod_name)
        attr = mod if attr_name is None else getattr(mod, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
