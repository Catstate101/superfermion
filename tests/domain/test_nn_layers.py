"""Neural network quantum layer domain tests."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import superfermion as sf  # noqa: E402
from superfermion.observables.core import SparsePauliOp  # noqa: E402

pytestmark = pytest.mark.domain


class TestFlaxQuantumLayer:
    def test_quantum_layer_forward(self):
        flax = pytest.importorskip("flax")
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        from superfermion.nn.quantum_layer import QuantumLayer

        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})

        layer = QuantumLayer(circuit=circuit, observable=observable)
        params = layer.init(jax.random.PRNGKey(0))
        output = layer.apply(params)
        assert jnp.isfinite(output)


class TestTorchQuantumLayer:
    def test_torch_layer_construction(self):
        torch = pytest.importorskip("torch")
        from superfermion.nn.torch_layer import TorchQuantumLayer

        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})

        layer = TorchQuantumLayer(circuit, observable, device="cpu")
        assert layer.weights is not None

    def test_torch_layer_forward(self):
        torch = pytest.importorskip("torch")
        from superfermion.nn.torch_layer import TorchQuantumLayer

        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})

        layer = TorchQuantumLayer(circuit, observable, device="cpu")
        output = layer()
        assert torch.isfinite(output)


class TestTFQuantumLayer:
    def test_tf_layer_construction_and_forward(self):
        tf = pytest.importorskip("tensorflow")
        from superfermion.nn.tf_layer import TFQuantumLayer

        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})

        layer = TFQuantumLayer(circuit, observable, device="cpu")
        layer.build(None)
        output = layer()
        assert tf.math.is_finite(output)
