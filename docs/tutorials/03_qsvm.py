"""Tutorial 3 — Quantum SVM on a tiny 2D dataset.

QSVM uses a parametrised circuit as a kernel feature map and trains via
JAX/optax. Good for very small classification problems where the
feature map is the point, not the dataset size.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import superfermion as sf
from superfermion.algorithms.qsvm import QSVM


def main() -> float:
    # Tiny 2-class dataset (AND-like)
    x_train = jnp.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.9], [0.8, 0.1]])
    y_train = jnp.array([0, 1, 0, 1])

    # Feature-map ansatz: one RY per feature + entangling CX
    ansatz = sf.Circuit(2)
    ansatz.ry(sf.param("a"), 0)
    ansatz.ry(sf.param("b"), 1)
    ansatz.cx(0, 1)

    qsvm = QSVM(ansatz, num_classes=2)
    res  = qsvm.fit(x_train, y_train, iterations=50)

    preds = qsvm.predict(res.optimal_params, x_train)
    acc   = float(np.mean(np.array(preds) == np.array(y_train)))
    print(f"QSVM train accuracy: {acc:.2%}  (final loss: {res.optimal_value:.4f})")
    return acc


if __name__ == "__main__":
    main()
