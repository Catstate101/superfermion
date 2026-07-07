"""
Differentiable quantum circuits via sf.State.

The old JAX primitive system (quantum_circuit_p, custom JVP/VJP/batching/MLIR
lowering) has been replaced. All differentiation now flows through:

- ``sf.State.grad()`` for adjoint gradients (Rust-native, framework-agnostic)
- ``jax.custom_vjp`` in ``nn/quantum_layer.py`` for JAX/Flax integration
- ``torch.autograd.Function`` in ``nn/torch_layer.py`` for PyTorch
- ``tf.custom_gradient`` in ``nn/tf_layer.py`` for TensorFlow

To compute gradients of quantum circuits:

    import superfermion as sf

    circuit = sf.Circuit(2).ry(sf.param('t'), 0).cx(0, 1)
    dag = circuit.to_ir()
    state = sf.simulate(circuit.bind({'t': 0.5}))
    grads = state.grad(observable, dag, {'t': 0.5})
"""
