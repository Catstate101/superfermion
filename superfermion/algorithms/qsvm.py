"""
Quantum Support Vector Machine (QSVM) / Variational Quantum Classifier (VQC).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer
from superfermion.algorithms.core import AlgorithmResult


class VQC(nn.Module):
    """Variational Quantum Classifier (standard QSVM implementation).
    
    Uses a quantum circuit as a feature map and classifier.
    """
    ansatz: sf.Circuit
    num_classes: int = 2
    backend: str = "jax"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch, n_features).
        """
        # 1. Feature injection (Simple addition-based encoding for this session)
        # In a real QSVM, this would be a dedicated FeatureMap circuit.
        
        # 2. Trainable Quantum Layer
        # Circuit execution returns probabilities or statevector
        weights = self.param(
            "weights",
            jax.nn.initializers.uniform(scale=2 * jnp.pi),
            (len(self.ansatz.parameters),)
        )
        
        f_jax = sf.qml.circuit_to_jax(self.ansatz, backend=self.backend)
        
        # Vectorized over batch
        def single_run(xi):
            # Combine xi and weights
            # This is a naive 'Angle Embedding' equivalent
            return f_jax(*(weights + xi))
            
        # Output is the statevector (complex64)
        q_out = jax.vmap(single_run)(x)
        
        # 3. Final projection to class logits
        # Convert complex statevector to real probabilities for differentiability
        q_real = jnp.abs(q_out)**2
        logits = nn.Dense(features=self.num_classes, name="classifier")(q_real)
        return logits


class QSVM:
    """Manager for Quantum Support Vector Machine training."""
    
    def __init__(
        self, 
        ansatz: sf.Circuit,
        num_classes: int = 2,
        optimizer: Optional[optax.GradientTransformation] = None
    ):
        self.ansatz = ansatz
        self.num_classes = num_classes
        self.optimizer = optimizer or optax.adam(0.05)
        self.model = VQC(ansatz, num_classes)

    def fit(
        self, 
        x_train: jnp.ndarray, 
        y_train: jnp.ndarray, 
        iterations: int = 100,
        seed: int = 42
    ) -> AlgorithmResult:
        """Train the QSVM on provided data."""
        from superfermion.utils import info, debug
        info(f"Starting QSVM training: {iterations} iterations, {len(x_train)} samples")
        
        key = jax.random.PRNGKey(seed)
        params = self.model.init(key, x_train)
        opt_state = self.optimizer.init(params)
        
        @jax.jit
        def loss_fn(p, x, y):
            logits = self.model.apply(p, x)
            one_hot = jax.nn.one_hot(y, self.num_classes)
            loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
            return loss

        @jax.jit
        def step(p, opt_st, x, y):
            val_loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
            updates, new_opt_st = self.optimizer.update(grads, opt_st, p)
            new_p = optax.apply_updates(p, updates)
            
            # Compute current accuracy
            logits = self.model.apply(new_p, x)
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return new_p, new_opt_st, val_loss, acc

        history = []
        for i in range(iterations):
            params, opt_state, loss, acc = step(params, opt_state, x_train, y_train)
            current_loss = float(loss)
            history.append(current_loss)
            
            if i % 10 == 0:
                debug(f"  Iteration {i:3d}: Loss = {current_loss:10.6f}, Accuracy = {float(acc)*100:.2f}%")
            
        info(f"QSVM training complete. Final Accuracy: {float(acc)*100:.2f}%")
            
        return AlgorithmResult(
            optimal_value=current_loss,
            optimal_params=params,
            history=history,
            metadata={"num_classes": self.num_classes, "iterations": iterations}
        )

    def predict(self, params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        """Predict class labels."""
        logits = self.model.apply(params, x)
        return jnp.argmax(logits, axis=-1)
