"""
Superfermion Classical NLP - Large Language Models and Transformers.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

class ClassicalTransformer(nn.Module):
    """
    A pure classical Transformer model (LLM core).
    """
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    seq_len: int

    @nn.compact
    def __call__(self, x):
        # 1. Embeddings
        token_emb = nn.Embed(self.vocab_size, self.dim)(x)
        pos_emb = self.param("pos_emb", nn.initializers.normal(), (1, self.seq_len, self.dim))
        x = token_emb + pos_emb

        # 2. Transformer Blocks
        for i in range(self.n_layers):
            # Self-Attention
            norm_x = nn.LayerNorm()(x)
            attn_out = nn.SelfAttention(num_heads=self.n_heads)(norm_x)
            x = x + attn_out
            
            # MLP
            norm_x2 = nn.LayerNorm()(x)
            ff_out = nn.Dense(features=self.dim * 4)(norm_x2)
            ff_out = nn.gelu(ff_out)
            ff_out = nn.Dense(features=self.dim)(ff_out)
            x = x + ff_out

        # 3. Output Head
        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size)(x)

class ClassicalLLM:
    """
    High-level API for the Classical LLM.
    """
    def __init__(self, vocab_size=5000, dim=512):
        self.model = ClassicalTransformer(
            vocab_size=vocab_size, 
            dim=dim, 
            n_layers=6, 
            n_heads=8, 
            seq_len=128
        )
        self.params = None

    def init(self, key):
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
        self.params = self.model.init(key, dummy_input)
        return self.params
