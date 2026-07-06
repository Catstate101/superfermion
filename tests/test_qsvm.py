"""
Test script for QSVM implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.algorithms.qsvm import QSVM


def test_qsvm_classification():
    print("Testing QSVM on a simple XOR-like classification task...")
    
    # 1. 2-qubit ansatz for classification
    c = sf.Circuit(2)
    c.rx(sf.param("theta0"), 0)
    c.rx(sf.param("theta1"), 1)
    
    # 2. Simple synthetic data (4 points, XOR-like)
    # x: (batch, dim=2)
    x = jnp.array([
        [0.0, 0.0],
        [jnp.pi, jnp.pi],
        [0.0, jnp.pi],
        [jnp.pi, 0.0]
    ])
    # y: Labels 0 for (0,0),(pi,pi) and 1 for (0,pi),(pi,0)
    y = jnp.array([0, 0, 1, 1])
    
    # 3. Setup QSVM
    qsvm = QSVM(c, num_classes=2)
    
    # 4. Fit the model
    print("  Fitting QSVM model...")
    results = qsvm.fit(x, y, iterations=150)
    
    print(f"  Final Loss: {results.optimal_value:.6f}")
    
    # 5. Predict and verify accuracy
    preds = qsvm.predict(results.optimal_params, x)
    accuracy = jnp.mean(preds == y)
    print(f"  Predictions: {preds}")
    print(f"  Labels:      {y}")
    print(f"  Accuracy:    {accuracy * 100:.2f}%")
    
    # For this simple XOR task, it should converge easily
    assert accuracy > 0.75
    print("[PASS] QSVM classification test passed!")

if __name__ == "__main__":
    try:
        test_qsvm_classification()
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
