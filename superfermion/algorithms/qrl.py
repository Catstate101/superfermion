"""
Quantum Reinforcement Learning (QRL) — Policy Gradient (REINFORCE).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer


class QuantumPolicy(nn.Module):
    """A quantum-assisted policy network for Reinforcement Learning.
    
    The policy π(a|s) is modeled by a Vequential Quantum Circuit.
    """
    ansatz: sf.Circuit
    num_actions: int
    backend: str = "jax"

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Forward pass to compute action probabilities.
        
        Args:
            state: Environment state observation (batch, state_dim).
        """
        # 1. State Encoding + Trainable Weights
        # Naive encoding for this session's kernel
        weights = self.param(
            "weights",
            jax.nn.initializers.uniform(scale=2 * jnp.pi),
            (len(self.ansatz.parameters),)
        )
        
        f_jax = sf.qml.circuit_to_jax(self.ansatz, backend=self.backend)
        
        def single_run(s):
            # Map state to circuit params
            return f_jax(*(weights + s))
            
        q_out = jax.vmap(single_run)(state)
        
        # 2. Policy Head
        logits = nn.Dense(features=self.num_actions)(jnp.abs(q_out)**2)
        return jax.nn.softmax(logits)


class QuantumREINFORCE:
    """Quantum REINFORCE (Policy Gradient) agent."""
    
    def __init__(
        self, 
        ansatz: sf.Circuit,
        num_actions: int,
        learning_rate: float = 0.01
    ):
        self.num_actions = num_actions
        self.model = QuantumPolicy(ansatz, num_actions)
        self.optimizer = optax.adam(learning_rate)

    def select_action(self, params: Dict[str, Any], state: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[int, jnp.ndarray]:
        """Sample an action from the policy."""
        probs = self.model.apply(params, state[None, :])[0]
        action = jax.random.choice(key, self.num_actions, p=probs)
        return int(action), probs

    def update(self, params: Dict[str, Any], opt_state: Any, trajectories: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """Perform a policy gradient update step."""
        # trajectories: list of {'state', 'action', 'reward'}
        
        @jax.jit
        def loss_fn(p):
            loss = 0.0
            for traj in trajectories:
                s, a, r = traj['state'], traj['action'], traj['reward']
                probs = self.model.apply(p, s[None, :])[0]
                log_prob = jnp.log(probs[a] + 1e-8)
                loss -= log_prob * r
            return loss / len(trajectories)
            
        l, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
