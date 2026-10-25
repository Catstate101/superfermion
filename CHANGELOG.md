# Changelog

All notable changes to the Superfermion framework.

## [0.1.17] — 2026-10-25

### Fixed
- **Source installs of 0.1.16** — the sdist shipped without `crates/sf-gpu`, a
  crate reachable only through the optional `gpu` path dependency, while
  `crates/sf-ir/Cargo.toml` still pointed at it. Every `pip install` from source
  died in `failed to load manifest for dependency sf-gpu`. The crate is now in
  the sdist, and a packaging check fails CI if any workspace member, path
  dependency or `include_str!` asset is missing from the artifact.

### Changed
- **PyO3 and numpy 0.23 → 0.30** — adds CPython 3.14/3.15 support; requires
  Rust 1.85+.
- **Stable-ABI wheels** — one `cp310-abi3` wheel per platform replaces the
  per-interpreter matrix. A single binary installs on CPython 3.10 and later,
  including interpreters released after the wheel was built.
- **Wheel coverage** — Linux x86_64/aarch64 (glibc 2.17+ and musl 1.2), macOS
  universal2, Windows amd64/arm64.

## [0.1.16] — 2026-10-24

### Fixed
- **Grover string convention** — `GroverOracle`/`grover_search` now treat
  marked strings and `top_bitstring` as counts keys (q0 last), so
  non-palindromic marks round-trip; the sampled branch traces out Toffoli
  ancillas.
- **QAOA cut metadata** — `QAOA._cut_value` now reverses the counts key before
  indexing the edge list, fixing `metadata["max_cut_value"]` on asymmetric
  graphs (a 4-node star reports 3, not 1).
- **Opaque unitaries above three qubits (D1)** — `Circuit.to_ir()` returns a
  compatibility DAG for circuits that contain a `unitary()` on more than three
  qubits; the statevector path applies the matrix instead of silently dropping
  it, and the MPS / density-matrix / stabilizer paths raise a directed error.
- **UNITARY matrices in serialization (D2)** — `to_json` / `from_json` and the
  `.sfc` format carry a `matrix` field next to `params`, including the
  `use_rust_storage=True` path; a `UNITARY` record without a matrix is rejected
  instead of reloading into a circuit that cannot execute.
- **Readout noise on pure-state methods (D3)** — readout-only `NoiseModel`s
  are applied to sampled counts on `statevector` and `mps` as well as
  `density_matrix`; gate noise still warns on the pure-state paths.
- **Stabilizer `shots=0` probabilities (D4)** — returns the exact `|c|^2`
  table (statevector fallback, capped at 20 qubits) instead of `{}`.
- **MPS output order after long-range gates (D10)** — `MPSState.to_statevector`
  and `sf.run(method="mps")` report qubit order; the physical-site view stays
  available as `to_statevector_sites`.
- **`sf.cat` / `sf.prepare` on the D1 compatibility path** — a circuit whose
  opaque unitary spans more than three qubits now raises a directed
  `ValueError` naming the API, instead of the PyO3 `TypeError` about
  `QuantumDAG` that the proxy produced.
- **Version string in the D1 fallback error** — the >3-qubit unitary error on
  the MPS, density-matrix and stabilizer paths named 0.1.15 unconditionally;
  it now reports the version of the package that raised it.

### Tests
- Regression coverage for both conventions in `tests/domain/test_grover.py`
  and `tests/domain/test_qaoa_minimize.py`.
- 26 regression tests in `tests/unit/test_v015_d1_d10_fixes.py` cover D1-D4 and
  D10, including the `use_rust_storage` serialization path; the stabilizer
  expectation in `tests/domain/test_sup_regression.py` was updated to the
  corrected D4 behaviour.
- 8 tests in `tests/unit/test_v016_cat_prepare_guard.py` pin the cat/prepare
  guard and the paths it must not disturb.

## [0.1.0] — 2026-08-10

### Added
- **Rust simulation core** — in-place statevector simulation with Rayon
  parallelism, MPS tensor network, stabilizer tableau, density matrix with
  Kraus channels. PyO3 bindings via `_sf_core`.
- **Adjoint differentiation** — memory-efficient Rust implementation
  (O(2^n) memory, 2M gate applications). Up to 200x faster than
  parameter-shift for deep circuits.
- **4 simulation methods** — statevector (CPU/GPU), MPS, stabilizer,
  density matrix, all accessible via `sf.run(device=, method=)`.
- **GPU simulation** — CUDA statevector via cudarc (sm_75+).
- **Hamiltonian expectation values in Rust** — `hamiltonian_expval()`
  computes Pauli expectation values without Python overhead.
- **Bridge functions** — `from_qiskit`, `to_qiskit`, `from_cirq`,
  `to_cirq`, `from_pennylane`, `to_pennylane` for cross-framework interop.
- **BP+OSD Decoder** — belief propagation with ordered statistics decoding
  fallback for quantum error correction.
- **Neural Decoder** — model registry with `load_pretrained()`, `train()`,
  and built-in pre-trained models for repetition and surface codes.
- **10 QEC codes** — Repetition, Shor, Steane, Bacon-Shor, Surface (2D),
  Toric, Color, Honeycomb, Hypercube, generic CSS.
- **Hardware compilation** — gate decomposition, rotation merging, SABRE
  routing, Pauli twirling targeting IBM, Rigetti, IonQ, IQM.
- **Chemistry module** — Jordan-Wigner and Bravyi-Kitaev transformations,
  UCCSD ansatz, cached H2 STO-3G Hamiltonian, PySCF bridge.
- **VQE and QAOA** — with configurable gradient methods (adjoint default
  for shots=0, parameter-shift fallback for shot-based).
- **Quantum algorithms** — Grover, QPE, HHL, Amplitude Estimation.
- **ML integration** — `QuantumLayer` (Flax), `TorchQuantumLayer`,
  `TFQuantumLayer` for hybrid quantum-classical models.
- **5 gradient methods** — adjoint, parameter-shift, finite difference,
  SPSA, quantum natural gradient.
- **QPU providers** — IBM Quantum, IonQ, AWS Braket, OpenQuantum device
  adapters with credential management.
- **Experiment tracking** — `experiment()` context manager with
  `LocalTracker` for JSON persistence.
- **Optional dependency groups** — `dev`, `gpu`, `qpu`, `benchmarks`,
  `chemistry`, `viz`, `ml`, `qml`, `docs`, `all`.
- **SECURITY.md** — vulnerability reporting policy.

### Fixed
- BP+OSD Decoder performs actual belief propagation (was returning zeros).
- Neural Decoder supports untrained fallback and pre-trained model loading.
- Cirq bridge handles rotation gates with correct angle extraction.
- H2 Hamiltonian coefficients corrected (signs, nuclear repulsion).
- UCCSD ansatz targets correct {|01>, |10>} subspace for 2-qubit H2.

## [0.0.1] — Initial
- Initial proof-of-concept release.
