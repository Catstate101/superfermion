# Changelog

All notable changes to the Superfermion framework.

## [0.1.22] — 2026-10-30

### Added
- **Chemistry toolkit** — the 17 modules that were in the tree but in no release
  join the five published in 0.1.21: active-space selection (MP2 natural
  orbitals, AVAS, APC, AEGISS, orbital entropy, orbital reduction), mappings and
  symmetry (Jordan-Wigner, Bravyi-Kitaev, symmetry-conserving BK, Z2 tapering,
  point-group sectors), solvers (MP2, SQD/QSCI, excited states and QSE,
  seniority, DMET), determinant-subspace tools, Pauli interop with Qiskit, and
  FCIDUMP/HDF5 ingest.
- **`pyscf_bridge` coverage** — restricted and unrestricted orbitals, integrals
  through `pyscf.ao2mo`, folding of the inactive spin orbitals, and reference
  occupations for a chosen active space.

### Fixed
- **OpenQASM 2 parser** — `parse_qasm2` skipped any `qreg`/`qubit` declaration
  it could not parse and left the DAG at its default width, so later gates were
  applied to the wrong qubits without an error. Each declaration now either
  sizes the DAG or returns an error naming the offending line.
- **State-dimension guard** — `n_qubits` was validated with `1 << n`, which
  wraps to 1 for `n >= usize::BITS` in a release build and panics in a debug
  one, so a one-amplitude array was accepted as a 64-qubit state. `checked_dim()`
  returns `None` instead of relying on the shift.
- **Logical ops** — `DefaultLogicalOps` returned the caller's DAG unchanged for
  logical X and Z, which reads as success while the operation is missing. Both
  now panic via `unimplemented!()` until the insertion is implemented.
- **QASM fallback reason** — the reason a native QASM parse fell back to the
  pure-Python one is now thread-local and exposed through module `__getattr__`,
  so two threads cannot read each other's reason and the documented
  `bridge._RUST_QASM_FALLBACK_REASON` spelling still works.

### Changed
- **Tests** — 174 new chemistry tests (parity against analytic references,
  methods, SCBK, defect regressions), per-thread QASM fallback assertions, and
  FFI errors checked to arrive in Python as catchable exceptions.
- **CI** — the wheel lane runs unfiltered: 21 tests deselected by name and an
  `--ignore` that hid a collection `ImportError` are both gone. pytest is capped
  below 9 because 9 removed `Function._nodeid`, which pytest-benchmark 5.3 reads
  while setting up its fixture. New `interop` and `chemistry-platforms` lanes
  install the frameworks and run the suite for real. `cirq-core` and `flax` are
  gated off the 3.15 wheel lane: `tensorstore` has no 3.15 wheel, so the install
  died in a Bazel source build before pytest ran.
- **Docs** — chemistry toolkit, interop and architecture pages.

## [0.1.21] — 2026-10-27

### Fixed
- **Benzene/6-31G import** — `_compute_pyscf` replaced the `O(nao^8)`
  `np.einsum("pi,qj,rk,sl,pqrs->ijkl", ...)` — nao = 66 for benzene/6-31G, which
  never finished — with `pyscf.ao2mo.full(compact=False)` and a `C^H . h . C` MO
  transform. `from_coeffs` now scans the non-zeros with `np.argwhere` instead of
  four nested Python loops. The benzene import went from not finishing (>120 s,
  killed) to 4.86 s; the (6, 6) Hamiltonian takes 0.10 s.
- **Active-space guard** — `to_hamiltonian` warns when mapping the full space
  needs more than 24 qubits, naming `active_space` as the fix. The operator is
  still returned, which is the documented behaviour.
- **OpenQASM 2 endianness** — the Rust parser reversed every qubit label to
  `(n-1-i)`, contradicting the little-endian pass-through used by `to_qasm` and
  `from_qiskit`, so `to_qasm -> from_qasm` was not an involution for asymmetric
  circuits. `q[i]` now passes through unchanged at all four sites (gates,
  reset/barrier, measure qubit, measure cbit), with a Rust regression test.
- **Bridge exporters** — `to_qasm` raises on unmappable gates instead of
  silently dropping them (barriers excepted); `to_ionq`, `to_cirq`,
  `to_pennylane` and `_to_braket` call `_ensure_gates()` so Rust-storage
  circuits export; the pure-Python QASM fallback is little-endian; `PL_MAP`
  gains `SX`, `PhaseShift`, `CY`, `ISWAP` and `CSWAP`.

### Changed
- **Bridge tests** — endianness round-trips through both parsers, the `to_qasm`
  drop guard, Rust-storage exporters, Cirq's unitary convention, and the new
  PennyLane ops. The xfail that stood in for the Rust QASM path is now a
  permanent passing lock.

## [0.1.20] — 2026-10-26

### Added
- **Update notice** — importing the package checks PyPI at most once a day and
  prints one line to stderr when a newer release exists. The request runs in a
  daemon thread with `urllib` imported inside the worker, the answer is cached
  for 24 h, and the check is silent offline, in CI, under pytest, and whenever
  `SF_NO_UPDATE_CHECK`, `NO_UPDATE_NOTIFIER` or `SF_OFFLINE` is set. Import cost
  is unchanged (+5 ms warm, +0.8 ms cold) and returns in 0.0 ms against an
  endpoint that never answers.
- **Install guidance** — docs and README state that pip never upgrades an
  existing install on its own, and that `pip install -U superfermion` is the way
  forward.

## [0.1.19] — 2026-10-26

### Changed
- **Dependencies** — `rand` 0.8.5 -> 0.8.6; website `next` 16.2.11 -> 16.4.3,
  `sharp` 0.34.5 -> 0.35.5, `postcss` 8.4.31 -> 8.5.31, `nanoid`
  3.3.16 -> 3.3.20, `source-map-js` 1.2.1 -> 1.2.2, `zod` 4.4.3 -> 4.6.7,
  `undici` 7.28.0 -> 7.30.1. The five transitive pins live in `pnpm.overrides`
  inside `pnpm-workspace.yaml`, which is where pnpm 11 reads them.
- **Lint** — the workspace is clean under `cargo clippy --all-targets`, and CI
  now runs it with `-D warnings` so test and example code cannot drift back.
  Three lints in the library's own tests were real and are fixed; the four
  measurement probes keep explicit index loops on purpose and carry a
  documented file-level allow instead.

### Fixed
- **`discarded_weight` docstring** — the head clause called it a fraction, which
  contradicts the body two lines down and the measured 1.79 at bond 16. It is a
  sum, and a regression test now pins the first line.

## [0.1.18] — 2026-10-26

### Fixed
- **MPS bond cap** — `blockdiag_exact_split` kept a prefix of `C`'s rows, so when
  a gate concentrated the state in the lower block and `2 * r_q` exceeded the
  cap, the rows carrying the mass were dropped while the rank check still
  passed. Measured on the 16-qubit, 12-layer brickwork at bond 16, it was
  reported as ordinary truncation: `F(16) = 0.0032` against `F(15) = 0.7889`, a
  larger cap returning a worse state. Rows are now chosen by measured weight,
  `F(16)` is 0.8328, and the curve is monotone through bond 64.
- **`mps_todense` argument check** — passing an IR, the natural mistake after
  `evolve_mps` hands back a state, failed deep inside the call with an opaque
  attribute error. It now raises a `TypeError` naming the type it received and
  the call that does work.

### Changed
- **MPS diagnostics documented** — `discarded_weight` is a sum and can exceed 1,
  `fidelity_lower_bound` is clamped so `0.0` only means `sum(eps) >= 1`,
  `truncation_events` counts steps rather than severity, and the densified
  vector is not renormalised. Each is stated at the binding with the measured
  numbers, and a new reference page covers the four quantities, the
  survivor-norm convention, the policy switch and a bond ladder.
- **Release CI** — the publish job now requests `contents: read`. Job-level
  permissions replace the defaults rather than adding to them, so the v0.1.18
  upload failed at checkout while the other 29 jobs were green.

## [0.1.17] — 2026-10-25

### Fixed
- **Source installs of 0.1.16** — the sdist shipped without `crates/sf-gpu`, a
  crate reachable only through the optional `gpu` path dependency, while
  `crates/sf-ir/Cargo.toml` still pointed at it. Every `pip install` from source
  died in `failed to load manifest for dependency sf-gpu`. The crate is now in
  the sdist, and a packaging check fails CI if any workspace member, path
  dependency or `include_str!` asset is missing from the artifact.
- **`superfermion.__version__`** — the 0.1.17 tree still reported `0.1.16` at
  runtime, so `pip show` and `sf.__version__` would have disagreed. A check now
  compares `pyproject.toml`, `Cargo.toml`, `__init__.py` and the release tag,
  and the wheel smoke test asserts the installed package reports the version
  its metadata claims.

### Changed
- **PyO3 and numpy 0.23 → 0.30** — adds CPython 3.14/3.15 support; requires
  Rust 1.85+.
- **Stable-ABI wheels** — one `cp310-abi3` wheel per platform replaces the
  per-interpreter matrix. A single binary installs on CPython 3.10 and later,
  including interpreters released after the wheel was built.
- **Wheel coverage** — Linux x86_64/aarch64 (glibc 2.28+ and musl 1.2), macOS
  universal2, Windows amd64/arm64. Hosts older than glibc 2.28 (CentOS 7,
  Ubuntu 18.04) build from the sdist, which now works.

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
