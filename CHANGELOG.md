# Changelog

All notable changes to the Superfermion framework.

## [0.1.0] — Unreleased

### Added
- **De Facto CLI (26 commands)**: Added 6 new commands for universal quantum CLI:
  - `sf plugin list/install/create` — Plugin management for backends, templates, passes
  - `sf auth login/logout/status` — Multi-provider credential management (IBM, IonQ, AWS)
  - `sf convert` — Format conversion (JSON, QASM, SFC)
  - `sf estimate` — QPU cost estimation across providers
  - `sf compare` — Multi-backend execution comparison
  - `sf jobs list/status/cancel` — Cross-provider job management
- **New Bridge Functions**:
  - `from_cirq(cirq_circuit)` — Import Cirq circuits to Superfermion
  - `to_circuit(circuit)` — Export Superfermion circuits to Cirq
  - `to_pennylane(circuit)` — Export to PennyLane quantum functions
- **BP+OSD Decoder**: Full Belief Propagation + Ordered Statistics Decoding implementation
  for quantum error correction, replacing the previous stub.
- **Neural Decoder**: Model registry with `load_pretrained()`, `train()`, and built-in
  pre-trained models for repetition codes (d=3, d=5) and surface codes (d=3).
- **9 New CLI Commands**: `compile`, `transpile`, `noise`, `gradient`, `train`,
  `circuit`, `statevector`, `qpu`, `config` — enabling full quantum computing
  workflows entirely through the CLI.
- **Enhanced `sf validate`**: Expanded from 9 to 22 validation checks covering
  Rust backend, MPS, CuPy GPU, noise model, QEC decode, configuration, telemetry,
  security, compiler, gradient, pulse, and data pipeline.
- **Enhanced `sf benchmark`**: Multi-backend comparison (simulator, rust, jax, mps, cuda),
  compilation benchmark suite, and Benchpress integration.
- **`--no-banner` flag**: Skip the golden banner on CLI entry for faster cold starts.
- **SECURITY.md**: Vulnerability reporting policy, dependency audit, and security features.
- **macOS CI**: Python test and Rust test jobs on `macos-latest` (Apple Silicon).
- **Apple Silicon detection** in `sf info` output.
- **Optional dependency groups**: `qpu` (qiskit-ibm-runtime, braket-sdk), `viz` (matplotlib),
  `chemistry` (pyscf), and `pytest-timeout` in dev dependencies.

### Fixed
- BP+OSD Decoder no longer returns `zeros_like(syndrome)` — now performs actual
  belief propagation with OSD fallback on the Tanner graph.
- Neural Decoder no longer requires a pre-initialized model — supports untrained
  fallback, pre-trained model loading, and training pipelines.
- Benchmark compilation suite uses correct `level=` parameter for `sf.compile()`.
- Cirq bridge handles rotation gates (Rx, Ry, Rz, XPowGate, etc.) with angle extraction.
- CXPowGate and CZPowGate now correctly map to controlled-X/Z.

### Documentation
- Updated CLI reference with 26 commands and de facto CLI features.
- Updated bridge documentation with Cirq and PennyLane integration.
- Added Quick Import Guide to MISSING_MODULES_STATUS.md.
- Updated frontend docs with new CLI and bridge features.

### Benchmarks
- **Benchpress**: 219/221 tests passing
- **GPU (CuPy)**: 5.5× speedup at 18Q, 13.5× at 20Q, 12.0× at 22Q vs CPU statevector
- **CLI**: 26 total commands (20 enhanced + 6 new de facto)
- **Tests**: 609 total passing (552 unit + 26 de facto CLI + 14 ZNE/scheduler + 9 validate + 8 tutorials)

## [0.0.1] — Initial
- 11 backends: statevector, jax, rust, mps, cuda, cuda_mps, stabilizer, clifford, dm, mps_rust, turbo
- 6 Rust crates: sf-ir, sf-compiler, sf-router, sf-pulse, sf-qec, sf-bindings
- 4 QPU providers: IBM, IonQ, Braket, OpenQuantum
- 12 QEC codes with MWPM and Union-Find decoders
- Noise model with 4 channel types and ibm_eagle preset
- JAX-native differentiable circuits
- Algorithms: VQE, QAOA, QSVM, QRL, QBM, QuantumGPT
- 7 infrastructure modules: config, security, serialization, telemetry, data, experiment, pulse
- CLI with 12 commands and golden banner
- Web dashboard and documentation site
- GitHub Actions CI (Rust + Python + lint) and release pipeline (wheels + PyPI + Docker)
