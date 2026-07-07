"""
Quantum Reinforcement Learning (QRL) — Policy Gradient (REINFORCE).

Framework-agnostic implementation using sf.State and numpy.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import superfermion as sf


class QuantumREINFORCE:
    """Quantum REINFORCE (Policy Gradient) agent.

    Uses a parameterized quantum circuit as the policy network.
    Gradients computed via parameter-shift rule.

    Args:
        ansatz: Parameterized sf.Circuit.
        num_actions: Number of discrete actions.
        learning_rate: Step size for gradient updates.
        device: Execution target.
        method: Simulation method.
    """

    def __init__(
        self,
        ansatz: sf.Circuit,
        num_actions: int,
        learning_rate: float = 0.01,
        device: str = "cpu",
        method: str = "statevector",
    ):
        self.ansatz = ansatz
        self.num_actions = num_actions
        self.lr = learning_rate
        self.device = device
        self.method = method
        self.param_names = list(ansatz.parameters) if ansatz.parameters else []
        self.n_params = len(self.param_names)
        self.weights = np.random.uniform(0, 2 * np.pi, size=self.n_params)
        self._policy_w = np.random.normal(0, 0.1, (2 ** ansatz.n_qubits, num_actions))
        self._policy_b = np.zeros(num_actions)

    def _get_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities for a single state."""
        params = self.weights + state[:self.n_params] if len(state) >= self.n_params else self.weights
        p_dict = dict(zip(self.param_names, params.tolist()))
        bound = self.ansatz.bind(p_dict)
        q_state = sf.simulate(bound, device=self.device, method=self.method)
        q_probs = q_state.probabilities()

        logits = q_probs @ self._policy_w + self._policy_b
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """Sample an action from the policy."""
        probs = self._get_probs(state)
        action = np.random.choice(self.num_actions, p=probs)
        return int(action), probs

    def update(self, trajectories: List[Dict[str, Any]]) -> float:
        """Perform a policy gradient update step.

        Args:
            trajectories: List of dicts with 'state', 'action', 'reward' keys.

        Returns:
            Average loss.
        """
        total_loss = 0.0

        for traj in trajectories:
            s = np.asarray(traj["state"])
            a = int(traj["action"])
            r = float(traj["reward"])

            probs = self._get_probs(s)
            log_prob = np.log(probs[a] + 1e-8)
            total_loss -= log_prob * r

            grad_softmax = probs.copy()
            grad_softmax[a] -= 1.0
            self._policy_w -= self.lr * r * np.outer(
                self._get_circuit_probs(s), grad_softmax
            )
            self._policy_b -= self.lr * r * grad_softmax

        return total_loss / max(len(trajectories), 1)

    def _get_circuit_probs(self, state: np.ndarray) -> np.ndarray:
        """Get circuit output probabilities for gradient computation."""
        params = self.weights + state[:self.n_params] if len(state) >= self.n_params else self.weights
        p_dict = dict(zip(self.param_names, params.tolist()))
        bound = self.ansatz.bind(p_dict)
        q_state = sf.simulate(bound, device=self.device, method=self.method)
        return q_state.probabilities()
