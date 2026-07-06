"""
Superfermion Classical ML - Traditional Machine Learning using JAX.
"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

@jit
def hinge_loss(w, b, x, y, reg=1.0):
    """
    SVM Hinge Loss with L2 regularization.
    """
    preds = jnp.dot(x, w) + b
    loss = jnp.mean(jnp.maximum(0, 1 - y * preds))
    return loss + 0.5 * reg * jnp.sum(w**2)

class JAX_SVM:
    """
    Ultra-fast Support Vector Machine using JAX gradients.
    """
    def __init__(self, learning_rate=0.01, reg=0.1):
        self.lr = learning_rate
        self.reg = reg
        self.w = None
        self.b = 0.0

    def fit(self, x, y, epochs=1000):
        if self.w is None:
            self.w = jnp.zeros(x.shape[1])
        
        loss_grad_fn = grad(hinge_loss, argnums=(0, 1))
        
        def step(w, b, x, y):
            dw, db = loss_grad_fn(w, b, x, y, self.reg)
            return w - self.lr * dw, b - self.lr * db

        # Optimize using JIT-ed loop
        for _ in range(epochs):
            self.w, self.b = step(self.w, self.b, x, y)
        
        return self.w, self.b

    def predict(self, x):
        return jnp.sign(jnp.dot(x, self.w) + self.b)

@jit
def mse_loss(w, b, x, y):
    """Mean Squared Error for Regression."""
    preds = jnp.dot(x, w) + b
    return jnp.mean((preds - y)**2)

class JAX_Regression:
    """Extreme-performance Linear Regression via JAX gradients."""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = 0.0

    def fit(self, x, y, epochs=500):
        if self.w is None: self.w = jnp.zeros(x.shape[1])
        grad_fn = grad(mse_loss, argnums=(0, 1))
        
        @jit
        def step(w, b, x, y):
            dw, db = grad_fn(w, b, x, y)
            return w - self.lr * dw, b - self.lr * db

        for _ in range(epochs):
            self.w, self.b = step(self.w, self.b, x, y)
        return self.w, self.b

    def predict(self, x):
        return jnp.dot(x, self.w) + self.b

class KMeans:
    """Fast Unsupervised K-Means clustering in JAX."""
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def fit(self, x, iterations=10):
        # Initialize centroids randomly
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, x.shape[0], (self.k,), replace=False)
        self.centroids = x[idx]

        @jit
        def update_step(centroids, x):
            # Compute distances: (nodes, k)
            dists = jnp.linalg.norm(x[:, None] - centroids[None, :], axis=2)
            labels = jnp.argmin(dists, axis=1)
            
            # One-hot encoding of labels: (nodes, k)
            oh_labels = jax.nn.one_hot(labels, self.k)
            
            # Sum features for each cluster: (k, features)
            counts = jnp.sum(oh_labels, axis=0)[:, None]
            sum_x = jnp.matmul(oh_labels.T, x)
            
            # Average (avoiding division by zero)
            new_centroids = jnp.where(counts > 0, sum_x / jnp.maximum(counts, 1), centroids)
            return new_centroids, labels

        for _ in range(iterations):
            self.centroids, _ = update_step(self.centroids, x)
        return self.centroids

# Friendly aliases
SVM = JAX_SVM
Regression = JAX_Regression
