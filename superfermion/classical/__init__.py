"""
Superfermion Classical - The Quantum-Free High-Performance JAX Engine.

Heavy dependencies (jax) are loaded lazily on first access.
"""

__all__ = [
    "solve_heat_equation",
    "complex_matrix_decomposition",
    "jacobian_computation",
    "CNN",
    "RNN",
    "DeepMLP",
    "ResNetBlock",
    "DeepCNN",
    "ClassicalTransformer",
    "ClassicalLLM",
    "JAX_SVM",
    "JAX_Regression",
    "KMeans",
    "SVM",
    "Regression",
    "simulate_classical_vibration",
    "classical_dynamics_step",
    "sv",
    "GCN",
    "normalize_adjacency",
]

_LAZY_MAP = {
    "solve_heat_equation":           "superfermion.classical.math",
    "complex_matrix_decomposition":   "superfermion.classical.math",
    "jacobian_computation":           "superfermion.classical.math",
    "CNN":          "superfermion.classical.nn",
    "RNN":          "superfermion.classical.nn",
    "DeepMLP":      "superfermion.classical.nn",
    "ResNetBlock":  "superfermion.classical.nn",
    "DeepCNN":      "superfermion.classical.nn",
    "ClassicalTransformer":  "superfermion.classical.nlp",
    "ClassicalLLM":          "superfermion.classical.nlp",
    "JAX_SVM":         "superfermion.classical.ml",
    "JAX_Regression":  "superfermion.classical.ml",
    "KMeans":          "superfermion.classical.ml",
    "simulate_classical_vibration":  "superfermion.classical.sv",
    "classical_dynamics_step":       "superfermion.classical.sv",
    "sv":              "superfermion.classical.sv",
    "GCN":             "superfermion.classical.gnn",
    "normalize_adjacency": "superfermion.classical.gnn",
}


def __getattr__(name):
    if name == "_LAZY_MAP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name = _LAZY_MAP.get(name)
    if mod_name is not None:
        import importlib
        mod = importlib.import_module(mod_name)
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    # Friendly aliases
    if name == "SVM":
        return __getattr__("JAX_SVM")
    if name == "Regression":
        return __getattr__("JAX_Regression")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
