"""
Quantum Algorithms — VQE, QAOA, Grover, QPE, HHL, QSVM, QRL, QBM, Amplitude Estimation.

Heavy dependencies (scipy, jax) are loaded lazily on first access.
"""

__all__ = [
    "VQE", "QAOA", "QSVM", "QuantumREINFORCE", "QBM",
    "QuantumKernel", "grover_search", "GroverOracle",
    "quantum_phase_estimation", "hhl_solve", "amplitude_estimation",
    "AlgorithmResult",
]

_LAZY_MAP = {
    "VQE":                    "superfermion.algorithms.variational",
    "QAOA":                   "superfermion.algorithms.variational",
    "AlgorithmResult":        "superfermion.algorithms.core",
    "QSVM":                   "superfermion.algorithms.qsvm",
    "QuantumREINFORCE":       "superfermion.algorithms.qrl",
    "QBM":                    "superfermion.algorithms.qbm",
    "QuantumKernel":          "superfermion.qml.algorithms.quantum_kernel",
    "grover_search":          "superfermion.algorithms.grover",
    "GroverOracle":           "superfermion.algorithms.grover",
    "quantum_phase_estimation": "superfermion.algorithms.qpe",
    "hhl_solve":              "superfermion.algorithms.hhl",
    "amplitude_estimation":   "superfermion.algorithms.amplitude_estimation",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
