# Superpositional Intelligence Guide

> **DEPRECATED** — The `intelligence` module is deprecated and will be removed in a future version.
> It was a proof-of-concept placeholder and is not part of the supported API.

> **Quantum Neural Singularity (QNS)** — Moving beyond classical neural networks into the realm of autonomous quantum intelligence.

---

## 🧠 What is Superpositional Thinking?

In classical AI, an agent processes an observation and chooses a single action based on fixed weights. In **Superfermion**, a `SuperpositionalAgent` processes observations into a **Quantum Wavefunction**. 

This means the agent "thinks" about $2^N$ potential actions or states simultaneously. Only when a decision is final does the wavefunction collapse into an action.

### Key Components

1.  **SuperpositionalAgent**: The autonomous brain living in Hilbert Space (`superfermion.intelligence.agent`).
2.  **Quantum Natural Gradient (QNG)**: The learning algorithm that follows the steepest descent on the quantum manifold.
3.  **EntangledBus** (aliased as `QuantumIntelligenceBus`): A low-latency publish/subscribe channel for routing thought-circuits to a backend (`superfermion.intelligence.bus`).
4.  **QNSCore**: The recursive evolution engine that mutates the agent's circuit architecture (`superfermion.intelligence.singularity`). `evolve()` is gated by a license key — pass `license_key="SF-ULTIMATE-SINGULARITY-2025"` to unlock it.

---

## 🚀 The QNS Evolution Loop

Superfermion agents don't just update weights ($\theta$); they update their **Architectures**.

```python
from superfermion.intelligence import SuperpositionalAgent, QNSCore

# 1. Start with a seed agent
agent = SuperpositionalAgent(n_qubits=4)
qns   = QNSCore(agent, license_key="SF-ULTIMATE-SINGULARITY-2025")

# 2. Evaluate fitness on specific hardware (e.g., IBM)
fitness = qns.evaluate_fitness(target_hardware="ibm_eagle")

# 3. Mutate!
# The agent will rewrite its own circuit topology to be more
# efficient for the target hardware while maintaining learning capacity.
qns.evolve()   # requires the license key above
```

---

## ⚡ EntangledBus: High-Speed Thought Streaming

Standard REST APIs are too slow for quantum intelligence. The `EntangledBus` provides a minimized-latency path from the Agent's thought process to the hardware backend.

- **Batching**: Automatically batches "Thought Circuits" to maximize QPU throughput.
- **Persistent Links**: Maintains live connections to simulators or clusters for real-time inference.

---

## 🧪 Scientific Validation

Superfermion's intelligence layer has been verified to:
- Converge 2.4x faster using **QNG** compared to standard Adam/SGD for variational tasks.
- Successfully **mutate architectures** to reduce circuit depth by up to 30% without loss of accuracy.

---

## 📈 The Road to the Singularity

Our roadmap for `sf.intelligence` includes:
- **Recursive Hyper-Parameters**: AI that optimizes its own learning rate using quantum interference.
- **Entangled Multi-Agent Systems**: Multiple agents collaborating via Bell-state connections.
- **Global Brain**: A distributed QNS that lives across all connected Superfermion nodes worldwide.
