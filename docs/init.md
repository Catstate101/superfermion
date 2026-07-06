# Superfermion User Guide — Advanced Patterns

Companion to [getting_started.md](getting_started.md). Collects runnable snippets for
quantum deep learning, LLM-style hybrid models, serialization, and the test matrix.

---

## Advanced: Quantum Deep Learning (QDL)

### QResNet: Quantum Residual Networks

```python
import jax.numpy as jnp
import superfermion as sf
from superfermion.qdl.resnet import QResNetBlock
from flax import linen as nn

c = sf.Circuit(2)
c.rx(sf.param("t0"), 0)
c.rx(sf.param("t1"), 1)

class MyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = QResNetBlock(c)(x)              # y = x + Q(x)
        return nn.Dense(1)(jnp.abs(x) ** 2)
```

### QuantumGPT: Hybrid Quantum Transformer

```python
import jax, jax.numpy as jnp
import superfermion as sf
from superfermion.qllm import QuantumGPT

q_circuit = sf.Circuit(4)
for i in range(4):
    q_circuit.ry(sf.param(f"t{i}"), i)

model = QuantumGPT(
    vocab_size=1000,
    dim=4,
    n_layers=4,       # classical + quantum layers interleaved
    n_heads=1,
    seq_len=128,
    q_circuit=q_circuit,
)

key    = jax.random.PRNGKey(0)
tokens = jax.random.randint(key, (1, 128), 0, 1000)
params = model.init(key, tokens)
logits = model.apply(params, tokens)         # (1, 128, 1000)
```

---

## Superpositional Intelligence (DEPRECATED)

The intelligence module is deprecated and will be removed. See [intelligence.md](intelligence.md).

```python
# DEPRECATED — do not use in new code
from superfermion.intelligence import SuperpositionalAgent, QNSCore, EntangledBus

agent = SuperpositionalAgent(...)
bus   = EntangledBus()                        # pub/sub channel between agents
core  = QNSCore(agent, license_key="SF-ULTIMATE-SINGULARITY-2025")
```

> `QNSCore.evolve()` is gated behind the license key. Without it, `QNSCore` runs
> in observation mode only.

---

## Serialization

```python
json_str   = circuit.to_json()
restored   = sf.Circuit.from_json(json_str)

qasm_str   = circuit.to_qasm3()               # OpenQASM 3.0
```

---

## Running Tests

```bash
# Full test suite (all pytest targets)
pytest -q

# Fast CLI sanity check
python -m superfermion.cli validate

# Targeted suites
pytest tests/test_algorithms.py
pytest tests/test_qml.py
pytest tests/test_qec.py
```
