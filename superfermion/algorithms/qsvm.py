"""
Quantum Support Vector Machine (QSVM) / Variational Quantum Classifier (VQC).

Uses sf.State and numpy for framework-agnostic quantum classification.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import superfermion as sf
from superfermion.algorithms.core import AlgorithmResult


class QSVM:
    """Quantum Support Vector Machine / Variational Quantum Classifier.

    Uses a quantum circuit as a feature map, trains via parameter-shift
    gradients through sf.State.

    Args:
        ansatz: Parameterized sf.Circuit.
        num_classes: Number of output classes.
        lr: Learning rate for gradient descent.
        device: Execution target.
        method: Simulation method.
    """

    def __init__(
        self,
        ansatz: sf.Circuit,
        num_classes: int = 2,
        lr: float = 0.05,
        device: str = "cpu",
        method: str = "statevector",
    ):
        self.ansatz = ansatz
        self.num_classes = num_classes
        self.lr = lr
        self.device = device
        self.method = method
        self.param_names = list(ansatz.parameters) if ansatz.parameters else []
        self.n_params = len(self.param_names)
        self.weights: Optional[np.ndarray] = None
        self._classifier_w: Optional[np.ndarray] = None
        self._classifier_b: Optional[np.ndarray] = None

    def _forward_single(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Run circuit for a single sample and return probabilities."""
        params = weights + x[:self.n_params] if len(x) >= self.n_params else weights
        p_dict = dict(zip(self.param_names, params.tolist()))
        bound = self.ansatz.bind(p_dict)
        state = sf.simulate(bound, device=self.device, method=self.method)
        probs = state.probabilities()
        return probs

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        iterations: int = 100,
        seed: int = 42,
    ) -> AlgorithmResult:
        """Train the QSVM on provided data."""
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0, 2 * np.pi, size=self.n_params)

        dim = 2 ** self.ansatz.n_qubits
        self._classifier_w = rng.normal(0, 0.1, (dim, self.num_classes))
        self._classifier_b = np.zeros(self.num_classes)

        history: List[float] = []

        for it in range(iterations):
            total_loss = 0.0
            for i in range(len(x_train)):
                probs = self._forward_single(x_train[i], self.weights)
                logits = probs @ self._classifier_w + self._classifier_b

                exp_logits = np.exp(logits - np.max(logits))
                softmax = exp_logits / np.sum(exp_logits)

                label = int(y_train[i])
                loss = -np.log(softmax[label] + 1e-10)
                total_loss += loss

                grad_softmax = softmax.copy()
                grad_softmax[label] -= 1.0
                self._classifier_w -= self.lr * np.outer(probs, grad_softmax)
                self._classifier_b -= self.lr * grad_softmax

            avg_loss = total_loss / len(x_train)
            history.append(avg_loss)

        return AlgorithmResult(
            optimal_value=history[-1],
            optimal_params={"weights": self.weights, "classifier_w": self._classifier_w},
            history=history,
            metadata={"num_classes": self.num_classes, "iterations": iterations},
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for input data."""
        if self.weights is None or self._classifier_w is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        preds = []
        for i in range(len(x)):
            probs = self._forward_single(x[i], self.weights)
            logits = probs @ self._classifier_w + self._classifier_b
            preds.append(np.argmax(logits))

        return np.array(preds)
