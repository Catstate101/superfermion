"""QML (Quantum Machine Learning) module for Superfermion.

Heavy dependencies (jax, flax) are loaded lazily on first access.
"""

__all__ = [
    "execute_circuit",
    "circuit_to_jax",
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
    "measurements",
    "fidelity",
    "expval",
    "expectation_value",
    "vn_entropy",
    "von_neumann_entropy",
    "purity",
    "state_fidelity_metric",
    "mutual_info",
    "participation_ratio",
    "compute_all_metrics",
    "partial_trace",
    "mitigation",
]

_LAZY_MAP = {
    "execute_circuit":   "superfermion.qml.gradient.core",
    "circuit_to_jax":    "superfermion.qml.gradient.core",
    "parameter_shift_grad":       "superfermion.qml.gradient.parameter_shift",
    "parameter_shift_grad_vector": "superfermion.qml.gradient.parameter_shift",
    "finite_diff_grad":           "superfermion.qml.gradient.parameter_shift",
    "AngleEmbedding":         "superfermion.qml.templates",
    "ZZFeatureMap":           "superfermion.qml.templates",
    "BasicEntanglerLayers":   "superfermion.qml.templates",
    "StronglyEntanglingLayers": "superfermion.qml.templates",
    "HardwareEfficientAnsatz": "superfermion.qml.templates",
    "TwoLocal":               "superfermion.qml.templates",
    "DataReuploadingCircuit": "superfermion.qml.templates",
    "ansatz":       ("superfermion.qml.ansatz", None),
    "encoding":     ("superfermion.qml.encoding", None),
    "measurements": ("superfermion.qml.measurements", None),
    "fidelity":     ("superfermion.qml.fidelity", None),
    "expval":                   "superfermion.qml.measurements",
    "expectation_value":        "superfermion.qml.measurements",
    "vn_entropy":               "superfermion.qml.measurements",
    "von_neumann_entropy":      "superfermion.qml.measurements",
    "purity":                   "superfermion.qml.measurements",
    "state_fidelity_metric":    ("superfermion.qml.measurements", "fidelity"),
    "mutual_info":              "superfermion.qml.measurements",
    "participation_ratio":      "superfermion.qml.measurements",
    "compute_all_metrics":      "superfermion.qml.measurements",
    "partial_trace":            "superfermion.qml.measurements",
    "mitigation": ("superfermion.mitigation", None),
}


def __getattr__(name):
    if name == "_LAZY_MAP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    entry = _LAZY_MAP.get(name)
    if entry is not None:
        import importlib
        # Support two formats:
        #   "module"              → import module, return getattr(mod, name)
        #   ("module", None)      → import module, return the module itself (submodule)
        #   ("module", "attr")    → import module, return getattr(mod, attr) (alias)
        if isinstance(entry, tuple):
            mod_name, attr_name = entry
        else:
            mod_name, attr_name = entry, name
        mod = importlib.import_module(mod_name)
        attr = mod if attr_name is None else getattr(mod, attr_name)
        # Cache on this module so we don't re-import every time
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
