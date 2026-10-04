"""Gradient methods — parameter shift, adjoint, SPSA, QNG, Riemannian.

Lightweight re-exports (lazy, to avoid import cycles). The recommended
default for variational loops is ``adjoint_grad``: one forward + one
backward pass regardless of parameter count (see ``adjoint`` docs for the
MSB-first observable convention).
"""

__all__ = [
    "adjoint_grad",
    "adjoint_grad_vector",
    "parameter_shift_grad",
    "parameter_shift_grad_vector",
    "finite_diff_grad",
    "spsa_grad",
    "qng_step",
    "riemannian_gradient",
    "sr_update",
    "sr_step",
]

_LAZY = {
    "adjoint_grad": "superfermion.qml.gradient.adjoint",
    "adjoint_grad_vector": "superfermion.qml.gradient.adjoint",
    "parameter_shift_grad": "superfermion.qml.gradient.parameter_shift",
    "parameter_shift_grad_vector": "superfermion.qml.gradient.parameter_shift",
    "finite_diff_grad": "superfermion.qml.gradient.parameter_shift",
    "spsa_grad": "superfermion.qml.gradient.spsa",
    "qng_step": "superfermion.qml.gradient.qng",
    "riemannian_gradient": "superfermion.qml.gradient.riemannian",
    "sr_update": "superfermion.qml.gradient.stochastic_reconfig",
    "sr_step": "superfermion.qml.gradient.stochastic_reconfig",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
