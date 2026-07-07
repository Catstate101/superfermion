"""Comprehensive gradient method domain tests."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import superfermion as sf  # noqa: E402
from superfermion.observables.core import SparsePauliOp  # noqa: E402


pytestmark = pytest.mark.domain

parameter_shift = pytest.importorskip(
    "superfermion.qml.gradient.parameter_shift",
    reason="parameter-shift gradient module unavailable",
)
adjoint = pytest.importorskip(
    "superfermion.qml.gradient.adjoint",
    reason="adjoint gradient module unavailable",
)
spsa = pytest.importorskip(
    "superfermion.qml.gradient.spsa",
    reason="spsa gradient module unavailable",
)
qng = pytest.importorskip(
    "superfermion.qml.gradient.qng",
    reason="qng gradient module unavailable",
)


class TestFiniteDiffVsAdjoint:
    def test_finite_diff_agrees_with_adjoint_on_ry(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})
        params = {"theta": 0.5}
        names = ["theta"]
        values = np.array([0.5])

        fd_grad = parameter_shift.finite_diff_grad(
            circuit, observable, params, device="cpu", eps=1e-4
        )
        adj_grad = adjoint.adjoint_grad_vector(circuit, observable, names, values)

        assert abs(fd_grad["theta"] - adj_grad[0]) < 1e-3

    def test_finite_diff_agrees_with_adjoint_two_params(self):
        theta = sf.param("theta")
        phi = sf.param("phi")
        circuit = sf.Circuit(2).ry(theta, 0).ry(phi, 1).cnot(0, 1)
        observable = SparsePauliOp.from_dict({"ZZ": 1.0})
        names = ["theta", "phi"]
        values = np.array([0.3, 0.7])
        params = dict(zip(names, values))

        fd_grad = parameter_shift.finite_diff_grad(
            circuit, observable, params, device="cpu", eps=1e-4
        )
        adj_grad = adjoint.adjoint_grad_vector(circuit, observable, names, values)

        for name, adj_val in zip(names, adj_grad):
            assert abs(fd_grad[name] - adj_val) < 1e-2


class TestSPSA:
    def test_spsa_approximates_gradient(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})

        def loss_fn(p):
            bound = circuit.bind({"theta": float(p[0])})
            result = sf.run(bound, device="cpu", shots=0)
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            return jnp.array(float(np.real(observable._fast_expval(sv))))

        params = jnp.array([0.5])
        key = jax.random.PRNGKey(0)
        grad = spsa.spsa_grad(loss_fn, params, key, delta=0.05)

        expected = -math.sin(0.5)
        assert grad.shape == (1,)
        assert abs(float(grad[0]) - expected) < 0.3


class TestQNG:
    def test_qfim_positive_semidefinite(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        param_vals = {"theta": 0.4}
        bound = circuit.bind(param_vals)
        state = sf.simulate(bound, device="cpu")
        dag = circuit.to_ir()
        g = np.array(state.qfim(dag, param_vals))
        assert g.shape == (1, 1)
        assert float(g[0, 0]) >= 0

    def test_qng_step_updates_parameters(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable_terms = [([3], 1.0, 0.0)]

        param_vals = {"theta": 0.5}
        bound = circuit.bind(param_vals)
        state = sf.simulate(bound, device="cpu")
        dag = circuit.to_ir()

        new_params = qng.qng_step(
            state, dag, observable_terms,
            param_names=["theta"],
            param_values=param_vals,
            learning_rate=0.01,
        )
        assert "theta" in new_params
        assert new_params["theta"] != 0.5


class TestRiemannianGradient:
    def test_riemannian_gradient(self):
        riemannian = pytest.importorskip("superfermion.qml.gradient.riemannian")
        grad = np.array([1.0, 0.5])
        metric = np.eye(2)
        nat_grad = riemannian.riemannian_gradient(grad, metric)
        np.testing.assert_allclose(nat_grad, grad)


class TestStochasticReconfig:
    def test_sr_update(self):
        sr = pytest.importorskip("superfermion.qml.gradient.stochastic_reconfig")
        params = np.array([0.1, 0.2])
        grad = np.array([0.5, -0.3])
        qfim = np.eye(2)
        updated = sr.sr_update(params, grad, qfim, learning_rate=0.01)
        assert updated.shape == (2,)
