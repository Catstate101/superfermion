"""
Superfermion — One framework. Every qubit. Every gradient. Every model.

A hardware-agnostic quantum-classical framework that makes quantum circuits
differentiable, compilable to any hardware, and trainable end-to-end with JAX.

Quick Start:
    >>> import superfermion as sf
    >>> circuit = sf.Circuit(2).h(0).cnot(0, 1)
    >>> result = sf.run(circuit, backend="simulator", shots=1000)
    >>> print(result.counts)
    {'00': 503, '11': 497}

Heavy dependencies (jax, flax, optax, scipy, torch, tensorflow) are loaded
lazily on first access via PEP 562 __getattr__.  The core package imports
with only numpy.
"""

import sys
from superfermion._lazy import LazyModule

__version__ = "0.1.0"

# ── Eager imports (only numpy required) ────────────────────────────────
from superfermion.circuit import Circuit
from superfermion.runner import run
from superfermion.compiler.manager import compile
from superfermion.parameters import param
from superfermion.backends import list_backends, get_backend, BackendName
from superfermion.utils.analytics import estimate_cost, benchmark

from superfermion import utils, runtime, security, serialization, telemetry, config, data, experiment, pulse

from superfermion.observables.core import PauliString, SparsePauliOp, Hamiltonian, expval
from superfermion.primitives import SFEstimator, SFSampler

# ── Lazy-loading config (delegated to LazyModule) ──────────────────────
_LAZY_SUBMODULES = {
    "qml": "superfermion.qml", "nn": "superfermion.nn",
    "qec": "superfermion.qec", "mitigation": "superfermion.mitigation",
    "chemistry": "superfermion.chemistry", "classical": "superfermion.classical",
    "algorithms": "superfermion.algorithms",
}

_LAZY_ATTRS = {
    "train": "superfermion.train", "Pipeline": "superfermion.pipeline",
    "VQE": "superfermion.algorithms.variational", "QAOA": "superfermion.algorithms.variational",
    "adjoint_grad_vector": "superfermion.qml.gradient.adjoint",
    "param_shift_grad": "superfermion.qml.gradient.parameter_shift",
    "param_shift_grad_vector": "superfermion.qml.gradient.parameter_shift",
    "finite_diff_grad": "superfermion.qml.gradient.parameter_shift",
    "SPSAGradient": "superfermion.qml.gradient.spsa",
    "QNGradient": "superfermion.qml.gradient.qng",
    "RiemannianGradient": "superfermion.qml.gradient.riemannian",
    "execute_circuit": "superfermion.qml.gradient.core",
    "circuit_to_jax": "superfermion.qml.gradient.core",
    "AngleEmbedding": "superfermion.qml.templates",
    "ZZFeatureMap": "superfermion.qml.templates",
    "BasicEntanglerLayers": "superfermion.qml.templates",
    "StronglyEntanglingLayers": "superfermion.qml.templates",
    "HardwareEfficientAnsatz": "superfermion.qml.templates",
    "TwoLocal": "superfermion.qml.templates",
    "DataReuploadingCircuit": "superfermion.qml.templates",
    "QuantumLayer": "superfermion.nn.quantum_layer",
    "TorchQuantumLayer": "superfermion.nn.torch_layer",
    "TFQuantumLayer": "superfermion.nn.tf_layer",
    "SurfaceCode2D": "superfermion.qec.codes.topological",
    "MWPMDecoder": "superfermion.qec.decoders",
    "UnionFindDecoder": "superfermion.qec.decoders",
    "BPOSD_Decoder": "superfermion.qec.decoders",
    "NeuralDecoder": "superfermion.qec.decoders",
    "QECManager": "superfermion.qec.manager",
    "FermionicOperator": "superfermion.chemistry.hamiltonians",
    "JordanWigner": "superfermion.chemistry.hamiltonians",
    "BravyiKitaev": "superfermion.chemistry.hamiltonians",
    "UCCSD": "superfermion.chemistry.ansatz",
    "PySCFBridge": "superfermion.chemistry.pyscf_bridge",
    "ClassicalNN": "superfermion.classical.nn",
    "JAX_SVM": "superfermion.classical.ml",
    "JAX_Regression": "superfermion.classical.ml",
    "ClassicalTransformer": "superfermion.classical.nlp",
    "ClassicalLLM": "superfermion.classical.nlp",
}

__all__ = [
    "Circuit", "run", "compile", "param", "list_backends", "get_backend", "BackendName",
    "estimate_cost", "benchmark",
    "qml", "nn", "qec", "mitigation", "utils", "runtime", "chemistry", "classical",
    "security", "serialization", "telemetry", "config", "data", "experiment", "pulse",
    "train", "Pipeline", "VQE", "QAOA",
    "adjoint_grad_vector", "param_shift_grad", "param_shift_grad_vector", "finite_diff_grad",
    "SPSAGradient", "QNGradient", "RiemannianGradient",
    "execute_circuit", "circuit_to_jax",
    "AngleEmbedding", "ZZFeatureMap", "BasicEntanglerLayers", "StronglyEntanglingLayers",
    "HardwareEfficientAnsatz", "TwoLocal", "DataReuploadingCircuit",
    "QuantumLayer", "TorchQuantumLayer", "TFQuantumLayer",
    "SurfaceCode2D", "MWPMDecoder", "UnionFindDecoder", "BPOSD_Decoder", "NeuralDecoder", "QECManager",
    "FermionicOperator", "JordanWigner", "BravyiKitaev", "UCCSD", "PySCFBridge",
    "ClassicalNN", "JAX_SVM", "JAX_Regression", "ClassicalTransformer", "ClassicalLLM",
    "PauliString", "SparsePauliOp", "Hamiltonian", "expval",
    "SFEstimator", "SFSampler",
    "__version__",
]

# Replace the module type so __getattr__ / __dir__ are handled by LazyModule
sys.modules[__name__].__class__ = type(
    __name__, (LazyModule,), {}
)
