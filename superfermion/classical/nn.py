"""
Superfermion Classical Neural Networks - CNN, RNN, and Deep Learning using Flax.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class CNN(nn.Module):
    """
    A High-Performance Convolutional Neural Network.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

class ResNetBlock(nn.Module):
    """A standard Residual Block."""
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, (3, 3))(x)
        y = nn.relu(y)
        y = nn.Conv(self.features, (3, 3))(y)
        
        if x.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1))(x)
            
        return nn.relu(y + residual)

class DeepCNN(nn.Module):
    """A complex ResNet-style CNN."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2))(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))
        
        for _ in range(3):
            x = ResNetBlock(64)(x)
        
        x = jnp.mean(x, axis=(1, 2)) # Global Average Pooling
        x = nn.Dense(1000)(x)
        return x

class RNN(nn.Module):
    """
    A Recurrent Neural Network (LSTM) for sequence modeling.
    """
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, seq_len, features)
        lstm = nn.LSTMCell(features=self.hidden_size)
        
        batch_size = x.shape[0]
        carry = lstm.initialize_carry(jax.random.PRNGKey(0), (batch_size,))
        
        # Simple loop for the demo
        for i in range(x.shape[1]):
            carry, _ = lstm(carry, x[:, i, :])
        
        return carry[0] # Return the hidden state h

class DeepMLP(nn.Module):
    """
    Standard Deep Learning Multi-Layer Perceptron.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x
