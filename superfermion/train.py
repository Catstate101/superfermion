"""
Superfermion Training Engine — JAX-native hybrid optimization loops.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np

from superfermion.data.dataset import DataLoader


class TrainState(train_state.TrainState):
    """Extended training state to track metrics."""
    metrics: Dict[str, List[float]] = None


def train(
    model: Any,
    train_loader: DataLoader,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    epochs: int = 10,
    val_loader: Optional[DataLoader] = None,
    seed: int = 42,
    verbose: bool = True,
    jit: bool = True,
) -> Tuple[TrainState, Dict[str, List[float]]]:
    """
    Centralized JAX-native training loop for Superfermion models.

    Args:
        model:        Flax nn.Module.
        train_loader: Superfermion DataLoader for training data.
        optimizer:    Optax optimizer (e.g., optax.adam(1e-3)).
        loss_fn:      Loss function taking (params, x, y) and returning scalar.
        epochs:       Number of training epochs.
        val_loader:   Optional DataLoader for validation.
        seed:         Random seed for initialization.
        verbose:      Whether to print progress.
        jit:          Whether to JIT-compile the train step.

    Returns:
        (state, history) where state is the final TrainState and history is a dict of metrics.
    """
    # 1. Initialization
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize parameters with a sample batch
    sample_batch_x, _ = next(iter(train_loader))
    variables = model.init(init_rng, sample_batch_x)
    params = variables['params'] if 'params' in variables else variables

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    history = {
        "train_loss": [],
        "val_loss": [] if val_loader else [],
        "epoch_time": []
    }

    # 2. Define Train Step
    def train_step(state, batch_x, batch_y):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, batch_x, batch_y)
        state = state.apply_gradients(grads=grads)
        return state, loss

    if jit:
        train_step = jax.jit(train_step)

    # 3. Training Loop
    if verbose:
        print(f"Starting Superfermion Training ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        batch_losses = []
        
        for batch_x, batch_y in train_loader:
            # Convert to jax arrays
            bx = jnp.array(batch_x)
            by = jnp.array(batch_y) if batch_y is not None else None
            
            state, loss = train_step(state, bx, by)
            batch_losses.append(loss)
        
        epoch_loss = float(jnp.mean(jnp.array(batch_losses)))
        epoch_time = time.time() - start_time
        
        history["train_loss"].append(epoch_loss)
        history["epoch_time"].append(epoch_time)
        
        # Validation
        val_str = ""
        if val_loader:
            v_losses = []
            for vx, vy in val_loader:
                v_losses.append(loss_fn(state.params, jnp.array(vx), jnp.array(vy)))
            val_loss = float(jnp.mean(jnp.array(v_losses)))
            history["val_loss"].append(val_loss)
            val_str = f" | val_loss: {val_loss:.6f}"

        if verbose:
            print(f"Epoch {epoch}/{epochs} | loss: {epoch_loss:.6f}{val_str} | time: {epoch_time:.2f}s")

    if verbose:
        print("Training Complete.")

    return state, history
