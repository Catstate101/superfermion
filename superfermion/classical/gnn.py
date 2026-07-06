"""
Superfermion Classical GNN - Graph Neural Networks using JAX.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer.
    Equation: H^{(l+1)} = σ(Ã H^{(l)} W^{(l)})
    where Ã is the normalized adjacency matrix.
    """
    features: int

    @nn.compact
    def __call__(self, adj, x):
        # x: (nodes, input_features)
        # adj: (nodes, nodes) - pre-normalized adjacency matrix
        
        # Linear transformation
        x = nn.Dense(self.features)(x)
        
        # Message passing (aggregation)
        x = jnp.matmul(adj, x)
        
        return x

class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network.
    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, adj, x):
        # Layer 1
        x = GCNLayer(self.hidden_dim)(adj, x)
        x = nn.relu(x)
        
        # Layer 2
        x = GCNLayer(self.out_dim)(adj, x)
        
        return x

def normalize_adjacency(adj):
    """
    Normalizes adjacency matrix: D^-1/2 * (A + I) * D^-1/2
    """
    adj_hat = adj + jnp.eye(adj.shape[0])
    degree = jnp.sum(adj_hat, axis=1)
    d_inv_sqrt = jnp.power(degree, -0.5)
    d_inv_sqrt = jnp.where(jnp.isinf(d_inv_sqrt), 0., d_inv_sqrt)
    d_mat = jnp.diag(d_inv_sqrt)
    return jnp.matmul(jnp.matmul(d_mat, adj_hat), d_mat)
