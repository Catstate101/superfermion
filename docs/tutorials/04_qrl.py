"""Tutorial 4 — Quantum REINFORCE on a trivial toy environment.

Demonstrates the full Flax init + action-selection + update loop. The
environment is a 1-step bandit with two arms; arm 1 always pays 1.0,
arm 0 pays 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import superfermion as sf
from superfermion.algorithms.qrl import QuantumREINFORCE


def main() -> float:
    ansatz = sf.Circuit(2)
    ansatz.ry(sf.param("t0"), 0)
    ansatz.ry(sf.param("t1"), 1)
    ansatz.cx(0, 1)

    agent = QuantumREINFORCE(ansatz, num_actions=2, learning_rate=0.05)

    key      = jax.random.PRNGKey(0)
    dummy_s  = jnp.zeros((1, 2))                       # (batch, state_dim)
    params   = agent.model.init(key, dummy_s)          # proper Flax params
    opt_state = agent.optimizer.init(params)

    trajectories = []
    for step in range(20):
        key, sub = jax.random.split(key)
        state    = jnp.zeros(2)                       # matches ansatz param count
        action, probs = agent.select_action(params, state, sub)
        reward   = 1.0 if int(action) == 1 else 0.0
        trajectories.append({"state": state, "action": int(action), "reward": reward})

    params, opt_state = agent.update(params, opt_state, trajectories)

    avg_reward = float(jnp.mean(jnp.array([t["reward"] for t in trajectories])))
    print(f"Mean reward over 20 steps: {avg_reward:.3f}")
    print(f"First action probs      : {probs}")
    return avg_reward


if __name__ == "__main__":
    main()
