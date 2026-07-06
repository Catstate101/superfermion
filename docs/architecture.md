# Superfermion Architecture Guide

> Complete map of every module, file, and layer in the Superfermion framework.

---

## High-Level Architecture

```
                         USER CODE
                             |
                      import superfermion as sf
                             |
                +------------+------------+-----------------------+
                |                         |                       |
          sf.Circuit()              sf.run() / sf.compile()   sf.intelligence
                |                         |                       |
     +----------+---------+     +---------+---------+     +--------+--------+
     |                    |     |                   |     |                 |
  Parameters         Gate DAG  |            Compiler Pipeline    Superpositional
  (sf.param)         (GateRecord) |          (PassManager)           Agent (QNS)
                          |     |                   |                 |
                     JAX Layer   |          Runtime / Arbiter    Entangled Bus
                (circuit_to_jax) |      (Job & Scaling Logic)         |
                          |     |             |                 QNS Core
                     Differentiable     Backend Registry              |
                     Primitive       +---+---+---+---+---+      Cloud & REST
                          |          |   |   |   |   |   |       (sf-serve)
                     Flax Integration  sim jax cuda mps cluster hardware
                    (QuantumLayer)
                          |
              +------------+------------+
              |            |            |
           QML Layer    QDL Layer    QLLM Layer
         (GAN,VAE,NLP)  (QResNet)   (QuantumGPT)
              |            |
       Classical AI    QEC Layer
       (GNN,CNN,ML)   (LDPCCode)
```

---

## Directory Structure

```
superfermion/
|
|-- crates/                          # Rust core (Layer 0-1)
|   |-- sf-ir/                       # Quantum IR: gate enums, DAG, passes
|   |   |-- src/
|   |       |-- lib.rs               # QuantumGate, QuantumDAG, QuantumOp
|   |       |-- dag.rs               # DAG implementation
|   |       |-- passes.rs            # Optimization passes (gate cancel, etc.)
|   |
|   |-- sf-compiler/                 # Compilation pipeline
|   |   |-- src/
|   |       |-- lib.rs               # PassManager, Pass trait
|   |       |-- decompose/           # Gate decomposition rules
|   |
|   |-- sf-router/                   # Qubit routing (SABRE)
|   |   |-- src/lib.rs               # SABRE router implementation
|   |
|   |-- sf-bindings/                 # PyO3 Python bindings
|       |-- src/lib.rs               # _sf_core module
|
|-- superfermion/                    # Python framework (Layer 2-10)
|   |-- __init__.py                  # Top-level API: Circuit, run, param
|   |-- circuit.py                   # Circuit class (549 lines, 30+ gates)
|   |-- parameters.py                # SymbolicParameter, sf.param()
|   |-- runner.py                    # sf.run() orchestrator
|   |-- results.py                   # ExecutionResult dataclass
|   |
|   |-- backends/                    # Layer 6: Backend registry
|   |   |-- __init__.py              # list_backends(), get_backend()
|   |   |-- base.py                  # Abstract Backend base class
|   |   |-- registry.py              # Auto-discovery & selection
|   |   |-- jax_sim.py               # JAX-native backend (596x faster)
|   |   |-- cluster.py               # GPU Cluster Orchestrator (sharding)
|   |
|   |-- runtime/                     # Layer 8: Cloud & Hardware
|   |   |-- __init__.py              # Job, Runtime, LocalJob
|   |   |-- specs.py                 # HardwareSpec (IBM, AWS, IonQ)
|   |   |-- arbiter.py               # Resource routing & scaling logic
|   |   |-- providers/
|   |       |-- ibm.py               # IBM Quantum Cloud interface
|   |       |-- aws.py               # AWS Braket / IonQ / Rigetti
|   |
|   |-- intelligence/                # Layer 9: Autonomous AI (DEPRECATED)
|   |   |-- __init__.py              # SuperpositionalAgent, EntangledBus, QNSCore
|   |   |-- bus.py                   # EntangledBus (High-speed thoughts)
|   |   |-- singularity.py           # QNS Core (Architectural Evolution)
|   |
|   |-- serve/                       # Layer 10: Cloud API
|   |   |-- app.py                   # FastAPI Gateway (sf-serve)
|   |   |-- auth.py                  # API Keys, Quotas, Multi-tenancy
|   |
|   |-- compiler/                    # Layer 5: Compiler pipeline
|   |   |-- manager.py               # sf.compile(), PassManager
|   |   |-- passes.py                # BasisTranslation, Routing, SABRE
|   |
|   |   |-- quantum_ai.py           # QNN, QGNN, QGAN, QVAE, QNLP
|   |   |-- measurements.py         # Purity, Entropy, Fidelity, Adv. metrics
|   |   |-- ansatz/
|   |   |   |-- hardware_efficient.py # Standard HE ansatz template
|   |
|   |-- classical/                  # Layer 7.5: JAX-Accelerated Classical AI
|   |   |-- math.py                 # PDE Solvers, Heat Equation
|   |   |-- ml.py                   # SVM, K-Means, Regression
|   |   |-- nn.py                   # CNN, RNN (LSTM), ResNet
|   |   |-- gnn.py                  # Graph Neural Networks (GCN)
|   |   |-- sv.py                   # State Vector Dynamical Systems
|   |
|   |-- nn/                          # Layer 4: Neural network module
|   |   |-- quantum_layer.py         # QuantumLayer (Flax nn.Module)
|   |
|   |-- observables/                 # Layer 7: Measurement operators
|   |   |-- core.py                  # Observable, PauliString, Hamiltonian
|   |
|   |-- algorithms/                  # Layer 7: Variational algorithms
|   |   |-- vqe.py                   # Variational Quantum Eigensolver
|   |   |-- qaoa.py                  # Quantum Approximate Optimization
|   |   |-- qsvm.py                  # Quantum SVM Classifier
|   |
|   |-- qdl/                         # Layer 7: Quantum Deep Learning
|   |   |-- resnet.py                # Quantum residual block
|   |
|   |-- qllm/                        # Layer 7: Quantum LLMs
|   |   |-- transformer.py           # Quantum Transformer + GPT
|   |
|   |-- qec/                         # Layer 7.5: Fault-Tolerant Engine
|   |   |-- manager.py               # QECManager (Orchestrator)
|   |   |-- decoders/                # MWPM, Union-Find, BP+OSD
|   |   |-- codes/                   # Surface, Toric, LDPC, 4D Hypercube
|   |
|   |-- utils/                       # Layer 2: Utilities
|       |-- logging.py               # Rich-based colorful logging
|       |-- exceptions.py            # Superfermion-specific error types
|
|-- docs/                            # Documentation
|   |-- getting_started.md           # Quick start guide
|   |-- architecture.md              # This file
|   |-- conventions.md               # Coding conventions
|   |-- intelligence.md              # Autonomous AI Guide (Singularity)
|   |-- cloud_guide.md               # REST API & Cluster Scaling
|
|-- test_*.py                        # Test suite (120+ tests)
```

---

## Layer Architecture

### Layer 0-1: Rust Core (`crates/`)
The high-performance substrate. Handles Intermediate Representation (IR) and low-level compilation (SABRE routing, gate cancellation).

### Layer 2-3: JAX Differentiable Layer (`superfermion/qml/`)
Custom JAX primitive that enables the "Holy Grail": end-to-end differentiation of quantum circuits using JIT, vmap, and autograd.

### Layer 5-6: Advanced Backends (`superfermion/backends/`)
Modular execution engines. The **JAX Backend** is 596x faster than NumPy, and the **Cluster Backend** enables distributed scaling across many GPUs.

### Layer 8: Runtime & Arbiter (`superfermion/runtime/`)
The "System" logic. Automatically routes jobs between local simulators, GPU clusters, or hardware targets based on resource requirements.

### Layer 9: Superpositional Intelligence (`superfermion/intelligence/`) — DEPRECATED
The autonomous AI layer (deprecated). Contains the **Quantum Neural Singularity (QNS)** proof-of-concept — will be removed in a future version.

### Layer 10: Cloud Gateway (`superfermion/serve/`)
The public-facing portal. Provides a secured REST/WebSocket API with API Keys and usage quotas for global enterprise access.

---

## Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Hardware-Agnostic** | Define logic once, run on JAX, CUDA, IBM, or AWS Braket seamlessly. |
| **Differentiable-First** | Entire framework is built on JAX for seamless gradient flow. |
| **Autonomous Scaling** | Arbiter handles the transition from laptop to GPU-cluster to QPU. |
| **Recursive Improvement** | QNS allows the AI to improve its own circuit topology. |
| **Enterprise-Grade** | Secured API with multi-tenancy, logging, and extensive testing. |
