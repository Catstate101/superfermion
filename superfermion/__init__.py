"""
Superfermion — One framework. Every qubit. Every gradient. Every model.

A hardware-agnostic quantum computation framework that makes quantum circuits
differentiable, compilable to any hardware, and trainable end-to-end.

Quick Start::

    import superfermion as sf

    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    result = sf.run(circuit, device="cpu", shots=1000)
    print(result.counts)   # {'00': 503, '11': 497}

    # With experiment tracking:
    with sf.experiment("bell-test"):
        result = sf.run(circuit, device="cpu")

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
from superfermion.results import RunResult
from superfermion.backends.factory import get_backend, list_backends

from superfermion.devices import DeviceExecutor, DeviceCapabilities
from superfermion.experiment.protocols import TrackerProtocol
from superfermion.experiment.context import experiment
from superfermion.experiment.local_tracker import LocalTracker

from superfermion.observables.core import PauliString, SparsePauliOp, Hamiltonian, expval
from superfermion.primitives import SFEstimator, SFSampler

from superfermion import utils, serialization, experiment as _experiment_mod, pulse

# ── Lazy-loading config (delegated to LazyModule) ──────────────────────
_LAZY_SUBMODULES = {
    "qml": "superfermion.qml",
    "nn": "superfermion.nn",
    "qec": "superfermion.qec",
    "mitigation": "superfermion.mitigation",
    "chemistry": "superfermion.chemistry",
    "algorithms": "superfermion.algorithms",
}

_LAZY_ATTRS = {
    "VQE": "superfermion.algorithms.variational",
    "QAOA": "superfermion.algorithms.variational",
    "adjoint_grad_vector": "superfermion.qml.gradient.adjoint",
    "param_shift_grad": "superfermion.qml.gradient.parameter_shift",
    "param_shift_grad_vector": "superfermion.qml.gradient.parameter_shift",
    "finite_diff_grad": "superfermion.qml.gradient.parameter_shift",
    "spsa_grad": "superfermion.qml.gradient.spsa",
    "qng_step": "superfermion.qml.gradient.qng",
    "riemannian_gradient": "superfermion.qml.gradient.riemannian",
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
    "uccsd_ansatz": "superfermion.chemistry.ansatz",
    "PySCFBridge": "superfermion.chemistry.pyscf_bridge",
}

__all__ = [
    # Core
    "Circuit", "run", "compile", "param", "RunResult",
    "list_backends", "get_backend",
    # Device protocol
    "DeviceExecutor", "DeviceCapabilities",
    # Experiment tracking
    "TrackerProtocol", "experiment", "LocalTracker",
    # Observables & primitives
    "PauliString", "SparsePauliOp", "Hamiltonian", "expval",
    "SFEstimator", "SFSampler",
    # Submodules (eager)
    "utils", "serialization", "pulse",
    # Submodules (lazy)
    "qml", "nn", "qec", "mitigation", "chemistry", "algorithms",
    # Version
    "__version__",
]

# Replace the module type so __getattr__ / __dir__ are handled by LazyModule
sys.modules[__name__].__class__ = type(
    __name__, (LazyModule,), {}
)
