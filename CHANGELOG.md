# Changelog

All notable changes to the Superfermion framework.

## [0.1.26] - 2026-11-06

Correctness work across circuit sampling, `sf.cat`, chemistry (FCIDUMP notation, AVAS, AEGISS,
XYZ), QEC decoding, the MPS bond cap, and the licensing and documentation corrections that
follow from them. The sampler change is the one that moves numbers: counts for a fixed seed
differ from 0.1.25. Everything below is covered by `tests/domain/test_v0125_fixes.py`, the
`sf-cat` Rust tests, `tests/unit/test_cat_density_matrix.py`, and
`tests/unit/test_cat_vs_qiskit.py` (which compares every `cat` lane against Qiskit and skips
when Qiskit is absent).

### Fixed

- **`sf.cat` density matrix**: the `mode` docstring said `density_matrix` "is not implemented and
  raises `ValueError`". It is implemented and exact — `simulate()` returns a `rho` matching
  Qiskit's `DensityMatrix` bit for bit (max abs diff `0.0` on a 4-qubit circuit) — and it is the
  only lane that can carry a `Reset` after use or an unread mid-circuit measurement. The
  docstring now describes it, and names the one refusal that is real: `sample()` on a
  density-matrix handle.
- **Density-matrix read-out**: `State.numpy()` flattens row-major, but the helper in
  `tests/unit/test_cat_density_matrix.py` reshaped it F-order, which returns `rho.T`. Every state
  in that file was real, where the two coincide, so a transposed `rho` passed all twelve tests. The
  helper is C-order now, and a new test pins a complex off-diagonal that a transpose cannot fake.
- **`cat.sample()` refusal**: the error said `plan is not a stabilizer plan`, which does not say
  which mode the handle holds. It now names it (`compiled in 'density_matrix' mode`).
- **Seeding**: adjacent integer seeds produced correlated shot streams, because Monte Carlo
  paths advanced the generator with `seed.wrapping_add(index)`. All five sites (`sf-ir` DAG
  sampling ×2, MPS, stabilizer, `sf-cat` Pauli-frame sampling) now mix through splitmix64
  (`mix_seed`), so neighbouring seeds share no structure. **Counts for a given seed therefore
  differ from 0.1.25** — a golden histogram pinned to one seed needs re-recording; a
  seed-averaged result does not. `sf-cat` is pinned by `adjacent_seeds_do_not_share_blocks`, which
  compares whole 64-shot blocks of packed keys (exact, not a correlation threshold): under the old
  code the second block of seed `s` was byte-identical to the first block of seed `s + 1`.
- **FCIDUMP**: the default integral notation is now detected instead of assumed. Explicit
  `NOTATION=` wins, then a PySCF header fingerprint, then the standard. Reading a chemist-notation
  file with the previous default silently mis-assigned every two-electron integral. A
  physicist-labelled file written with `annotate=False` no longer carries PySCF's writer
  fingerprint, so it re-reads as physicist instead of as a transposed chemist file.
- **BP+OSD**: `osd_order` is honoured. It was accepted and then ignored, so OSD ran with a
  fixed order regardless of the argument; the search now enumerates the free variables and is
  exhaustive once `osd_order >= n_qubits - rank(H)` *and* the coset fits the
  `_OSD_MAX_CANDIDATES` budget (`n_free <= 17`). Past that budget the result is the lightest
  pattern found, not the lightest that exists.
- **AVAS**: the active-space size is the requested count, projected onto PySCF's `minao`
  minimal basis rather than the working basis, which over-returned orbitals.
- **XYZ**: parsing accepts blank lines and trailing comments and raises a located error instead
  of failing obscurely; SCF reports its converged flag truthfully instead of always `True`.
- **AEGISS**: the entropy refinement no longer raises when every orbital falls below the
  threshold. It warns and returns the unrefined AVAS space with `refinement_applied=False`,
  which is what the default threshold did to water with the O target.
- **MPS `bond_dim`**: values below 1 are refused with a `ValueError`. `bond_dim=0` used to be
  accepted and sample the uniform distribution — the QR cap becomes `min(0, rank) = 0` — and a
  negative value surfaced as pyo3's `OverflowError` instead of naming the argument.

### Added

- **LICENSE** is now the verbatim Apache-2.0 text with the copyright lines filled in, the same
  shape Qiskit ships; the previous file was a paraphrase, which is why GitHub reported
  `NOASSERTION`. A `NOTICE` file carries the section 4(d) attribution, and both are packaged
  through `license-files`.
- **`scripts/release_notes.py`** renders a GitHub Release body from this changelog, falling back
  to a labelled commit list for versions the changelog predates.

### Docs

- **README / getting-started**: the wheel matrix now lists the tags PyPI actually publishes, and
  a new section states what a `pip install` does not provide — CUDA (the `gpu` Cargo feature is
  off by default and is not in any wheel), the crates.io crates (workspace members, unpublished),
  and free-threaded CPython (no limited API before 3.15, so 3.14t installs from source).
- **Contributor path**: `CONTRIBUTING.md` pointed at `github.com/superfermion/superfermion` (not
  this repository), required Rust 1.75 (the README and CI need 1.85), told contributors to run
  maturin inside `crates/sf-bindings` — which drops the root `pyproject.toml` that declares
  `module-name`, `manifest-path` and `python-source` — and ended with a hand `cp` of the built
  library into `superfermion/`, the exact move that lets a stale ABI-tagged module win the import.
  It now uses the wheel CI builds, prints the extension path to verify which engine loaded, and
  links `AGENTS.md` instead of a `docs/conventions.md` that does not exist. `AGENTS.md` carried the
  same broken build block.
- **Docs site**: new [Contributing page](https://superfermion.com/docs/contributing) carrying the
  setup, the six gates CI runs, the import-shadowing trap above, and what a PR is expected to
  contain; `Pull requests` is in the site header, and the README gained a Contributing section
  linking the guide, the page, the compare view and the issue tracker.

## [0.1.25] - 2026-11-05

Correctness work across QEC, error mitigation, gradients, chemistry, algorithms, the
compiler and the GPU lane, plus the documentation corrections that follow from it.
Every item below is pinned by a test; the additions are described by what they
change, and each test name states the behaviour it locks.

### Fixed

- **QEC**: codes gained `stabilizers()` / `gauge_operators()` and a real `encoder()`, built on
  new GF(2) primitives (`_gf2_rref`, `_gf2_kernel`); decoders gained minimum-weight search and
  a dependency check that reports the missing neural extra instead of failing deep inside
  import; `manager` gained `run_logical_lifecycle` and a qubit-limit guard.
- **Error mitigation**: zero-noise extrapolation now builds its folded expectation values from
  the circuit it is handed (`_folded_expectations`, `_amplification`, `_checked_scale`) and
  warns when extrapolation cannot change anything, instead of returning the unmitigated value
  silently; the noisy expectation path is separable from the ideal one.
- **Gradients**: adjoint observables are converted through `_observable_to_rust_terms`;
  parameter-shift normalises its observable argument; `qng` returns the step in the caller's
  parameter order; `grad_checks` resolves DAG parameter names rather than relying on labels.
- **Chemistry**: `uccsd_ansatz`, `excitation_pairs`, `hardware_efficient_ansatz` and the
  single/double excitation gates are exposed; `library` resolves names through
  `require`/`require_by_name`/`lookup` and raises on an unknown name; `interop` converts
  operators to and from Qiskit; `pyscf_bridge` derives its formula from the atoms.
- **Algorithms**: QPE builds controlled single-qubit gates from the gate matrix, HHL applies its
  bit reversal explicitly, and amplitude estimation builds the controlled embedding — each was
  previously assembled from an assumption about the input.
- **Results / runner / observables**: `results` gained documented gradient fallbacks and
  order-stable outcome indexing; `runner` distinguishes "not passed" from `None`
  (`_Unset`); MPS truncation is reported through `truncation_report()` with the four documented
  fields, and asking for the report does not change the value.
- **Sampling**: an omitted `seed` still defaults to the historical `42`, but `seed=None` no longer
  raises — it draws a fresh 32-bit seed and records it in `result.metadata["seed"]`, so unseeded
  repeats finally differ and a drawn run stays replayable. Previously every method re-seeded to
  `42` on every call, which made error bars from repeated runs meaningless.
- **Memory guard**: an exact state larger than physical RAM used to abort the interpreter inside
  the Rust allocator (exit `0xC0000409`, no `MemoryError`). The size is now estimated first and
  refused with a catchable `MemoryError` naming the arithmetic and the alternatives; override
  per run with `max_state_gib=`, globally with `SF_STATE_MEMORY_LIMIT_GIB` (`0` disables).
- **Classical register order**: `measure(q, cbit=c)` said the classical register was the output,
  but counts were qubit-keyed. They are relabelled when the cbit targets permute the qubits, so
  the string width and every other bit are unchanged.
- **Compiler**: native lowering moved into `compiler/layout.py` and
  `compiler/native_lowering.py`, with `specs` exposing `is_native` / `is_emittable` /
  `degree_histogram`; `manager` gained a Rust compile path.
- **Circuit**: added the `ch` (controlled-Hadamard) gate.
- **Rust**: `sf-ir` gained a batched MPS Pauli-expectation path
  (`simulate_mps_pauli_expval_batch_report`) exposed through `sf-bindings`; `sf-gpu` now
  contains the nvrtc initialisation panic, so `gpu_available()` returns `False` instead of
  raising `pyo3_runtime.PanicException`, and `gpu_diagnose()` reports the loader detail.

### Docs

- The execution guide gained a reproducibility section (when `seed` is fixed, when it is drawn,
  and how to replay) and the `MemoryError` precondition; the `sf.run` parameter table lists `seed`
  and `max_state_gib`.
- The GPU notes no longer claim an `sm_75+` gate: kernels are JIT-compiled for the device's own
  compute capability and were verified on `sm_61`, where the GPU state is bit-identical to the
  CPU state. The nvrtc filename gap, the graceful-degradation contract and the memory-bound
  qubit ceiling (16 bytes per amplitude) are now stated where the old text claimed "~30 qubits
  on GPU".

### Verification

- Full suite on this tree: **1582 passed, 0 skipped, 0 failed** (287.00 s).
- `cargo fmt --check` clean; `clippy -p sf-gpu --all-features` clean.
- GPU lane: `||psi_cpu - psi_gpu||^2 = 0` at 12, 16 and 20 qubits; seeded counts identical
  between lanes; against Qiskit Aer the CPU lane holds fidelity >= 0.999999999999988 and
  `max|d<Z>| <= 2.22e-16` over three circuit families.

## [0.1.24] — 2026-10-31

Gradient correctness. Five defects that returned **plausible wrong numbers** —
not exceptions — on 0.1.22 and 0.1.23, all reproducible on the PyPI wheel and on
a source build. Each is now pinned by a finite-difference-anchored test, and the
class of bug that produced them is blocked by an exhaustiveness test, a rule
table that raises instead of guessing, and a gate every install lane runs.

The same release closes an audit of `sf.cat`, the compile-once handle: nine
findings — one silent wrong answer, one panic that escaped `except Exception`,
two false refusals and five message defects. Four of them change what the
library accepts or refuses on purpose. The last unimplemented lane,
`density_matrix`, is implemented here; it is the only one that can hold a
`Reset` after use.

### Fixed
- **Adjoint gradients were 0.0 for `p`, `u`/`u3`, `cp`, `cu`** (and `r1`,
  `crx`, `crz` in the IR). The Rust kernel's generator table stopped at
  `Rx/Ry/Rz/Rxx/Ryy/Rzz` and had no error branch, so a parameter used only in
  one of the other seven gates fell through the accumulation guard. Reported
  on the reviewer's 2-qubit circuit: `p(a)` returned 0.0 where `rz(a)` — which
  differs from it only by a global phase — returned +0.038875. `u`/`cp`/`cu`
  now match central finite differences to 1e-11, and the whole table is
  verified against an independent numerical derivative of `to_matrix()`.
- **Adjoint gradients were 0.0 for an already-bound circuit.** `bind()` folds
  the symbols into constants, so the DAG no longer knows the names and
  `grad_dict.get(name, 0.0)` turned "not a parameter" into a plausible zero.
  It now raises `ValueError` naming the missing symbols and pointing at
  `circuit.to_ir()`; the same guard covers `res.grad()` and `State.grad()`.
- **Parameter-shift returned half the gradient when one name drove several
  gates.** `ry(t); ry(t)` at t=0.35 gave 0.0 instead of −1.2884353744: the
  shift used `bind()`, which is name-global, so both occurrences moved
  together. Shifts are now applied per occurrence through the new
  `Circuit.bind_occurrence`, and the contributions are summed.
- **Parameter-shift was wrong for controlled rotations.** `cu`'s θ generator
  has spectrum {0, ±½}, which the single π/2 shift cannot see; the new rule
  adds a π shift. Reported value −0.2424653649, true −0.1714489037 — now
  within 1e-9 of finite differences, along with φ, λ and γ.
- **Parameter-shift rejected `str` observables** with
  `AttributeError: 'str' object has no attribute '_fast_expval'`. `str`,
  `dict`, list-of-`str`, `SparsePauliOp`, `Hamiltonian` and `PauliString` are
  now all coerced through the public `expval`, in both `parameter_shift_grad`
  and `finite_diff_grad`.
- **A gate outside the shift-rule table now raises**
  `ValueError("parameter-shift rule unavailable for gate …")` instead of
  returning a number derived from the wrong spectrum.
- **`from_qasm3` silently dropped seven gate names.** Its dispatch table had
  no entry for `r1`, `cp`/`cr1`, `cu`/`cu3`, `crx`, `cry`, `crz` or `u`/`u3`,
  and an unmapped gate hit a bare `continue`. `from_qasm3(to_qasm3(c))` therefore
  returned a circuit with the right qubit count and **missing operations** —
  the same wrong-answer class as the gradient defects, in serialization. A
  two-gate probe came back with one gate and a statevector off by 0.28.
  Unknown statements now raise, naming the statement and pointing at
  `strict=False`; the escape hatch keeps the old skip behaviour for callers who
  know what they are dropping. Round-trips are now bit-exact (0.00e+00) for
  every gate the writer emits, including `cu` whose fourth parameter
  (`gamma`) follows the qubits in Python but precedes them in QASM. Classical
  feed-forward (`c[k] = measure q[j]` / `if (c[k] == v) g;`) is parsed and
  re-emitted text-identically, and a whole-register `measure q;` is expanded.
- **`Circuit.draw()` raised `UnicodeEncodeError` on a cp1252 console.** The
  diagram used `─`, `●`, `⊕` and `│` unconditionally, so `print(circuit.draw())`
  died with `'charmap' codec can't encode character '\u2500'` on a default
  Windows code page and under `PYTHONIOENCODING=ascii`. `draw()` now checks what
  the target stream can encode and falls back to `-`, `o`, `+`, `|`.
- **`bridge.from_qasm` rejected its own QASM 3 output.** The function documents
  OpenQASM 2.0/3.0, but the native parser does not read `bit[n] c;` and the
  pure-Python fallback is QASM 2 only, so `from_qasm(c.to_qasm3())` raised
  `ValueError: Could not parse QASM statement: 'bit[2] c;'`. QASM 3 input the
  native parser rejects is now routed to `from_qasm3`.
- **`sf.cat(mode="adjoint")` returned a gradient that does not exist.** On
  `ry(θ); measure(0); rz(φ)` with observable X the readout leaves the qubit
  with no coherence, so `<X> == 0` for every angle and *both* partial
  derivatives are exactly 0. `Mode::Adj` compiled the measurement away and
  returned the unitary-only `cos θ · cos φ` — 0.7307 at θ=0.7, φ=0.3. Every
  lane now runs the same dynamic-circuit guard before compiling, with its own
  name in the message; a terminal measurement stays legal, because it cannot
  change a unitary or the expectation of a Pauli observable.
- **`measure_all()` with `n_cbits < n_qubits` panicked inside Rust.** The
  `PanicException` derives from `BaseException`, so `except Exception` did not
  catch it; it fired from every entry point, `sf.run(...,
  method="statevector")` included, and printed a Rust backtrace instead of a
  message. `c_if()` already bounds-checked its classical bit — `measure()` did
  not. It does now: `measure() classical bit 1 out of range (circuit has 1
  cbits).` with the bit, the width and a fix. Five regression tests, including
  the `to_ir()` call where the panic used to happen.
- **`statevector` refused every measurement, including a terminal one**, while
  `stabilizer`/`clifford` accepted the same circuit, so the answer depended on
  which lane you asked for. A readout that nothing follows is now dropped (the
  returned state is the state *before* it, which is the number
  `Aer.save_statevector` gives when the save sits there), and a readout with
  any gate after it is still refused, even a gate on another qubit. That is
  deliberately stricter than the tableau rule: a later gate leaves a
  distribution's marginal alone, and does not leave a state alone.
- **A `Reset` on an untouched qubit was refused** by the adjoint and
  statevector lanes, so `reset(0); h(0); cx(0,1)` — an ordinary hardware-style
  preamble — could be neither simulated nor differentiated, while both tableau
  lanes accepted it. A leading reset is the identity on `|0>` and is now
  skipped: the adjoint gradient with and without it is bit-identical
  (`assert_eq!` on the returned state, not a tolerance). A reset *after* use
  stays refused in both lanes, and that is a deliberate divergence from
  Qiskit — `Statevector.from_instruction` returns `|00>` there, a
  post-selection on one branch, where the reset channel gives
  `diag(.5, 0, .5, 0)` and no single pure vector can hold it.
- **Refusal messages named the wrong lane.** The adjoint gate-set limit came
  from `AdjProgram`, one code path over from the guard, so it said `cat adj:
  unsupported gate H (only RY/RZ ...)` while every neighbouring refusal said
  `cat: <lane> plan:`; the dynamic guard said `stabilizer/clifford` to adjoint
  callers; two explanations hard-coded "the Pauli-frame sampler"; and four
  printed messages carried an em dash, which a cp437 console renders as `ù`.
  Each lane now names itself, and the printed strings are ASCII.

### Added
- `Circuit.r1(φ,q)`, `Circuit.crx(θ,c,t)`, `Circuit.cry(θ,c,t)` and
  `Circuit.crz(θ,c,t)`. Those gates were already in both rule tables (adjoint
  and parameter-shift) but had no Python builder, so that coverage could only
  be reached from a hand-written IR. `r1` maps to its own opcode and is the
  same matrix as `p`; the IR has no `CRy` opcode, so `cry(θ)` emits the exact
  equivalent `cu(θ, 0, 0)` (`U3(θ,0,0) == Ry(θ)`, and a controlled block keeps
  the phase). Checked against Qiskit's `Statevector` (0.0 on ⟨O⟩), PennyLane
  (≤4.4e-16 on ⟨O⟩, ≤8.3e-17 on the gradient) and finite differences, and
  pinned by four new tests in `test_gradient_regressions.py`.
- `Circuit.param_occurrences(name)` and `Circuit.bind_occurrence(values, name,
  k, value=…)` — bind one occurrence of a shared symbol without touching the
  others.
- `tests/domain/test_gradient_regressions.py` — ten finite-difference-anchored
  tests covering every defect above, the three public gradient entry points,
  and VQE on the package's own `u_ansatz` (which used to stall at 0.38165590
  with a true ground state of −1.0).
- `tests/domain/test_engine_provenance.py` — every run records the interpreter,
  package path, engine path and engine sha256; `SF_EXPECT_ENGINE_ROOT` turns
  "which install produced this?" into an assertion. `tests/conftest.py` prints
  the same identity in the pytest header.
- `scripts/grad_gate.py` — one file, numpy only, about twenty seconds: verifies
  all five defects plus the VQE path against whichever engine the current
  interpreter can import, and exits non-zero on any failure. Runs in CI on the
  editable installs, the abi3 wheel, an sdist-built install and free-threaded
  3.14t, and in the release workflow before and after the wheel matrix.
- Rust: `test_every_parametric_gate_has_generator`,
  `test_closed_form_covers_every_parametric_slot`,
  `test_numerical_fallback_matches_closed_form` and
  `test_unknown_parametric_gate_is_an_error` in `crates/sf-ir/src/adjoint.rs`.
  A new parametric gate that is not added to the table now fails the build
  rather than returning zeros.
- macOS row in the Python matrix, and `SF_EXPECT_ENGINE_ROOT` on every row.
- `tests/domain/test_gate_unitary_oracle.py` — 128 dependency-free assertions
  that re-derive every gate's matrix from its definition (Pauli exponentials,
  controlled blocks, basis permutations) and compare it element for element
  with `Circuit.to_unitary()`, for all 37 builders and every parameter slot.
  No Qiskit/PennyLane/qulacs needed, so the oracle runs on every install lane.
  It also pins the documented relationship between the two orderings: a
  `GateRecord` matrix is textbook (`qubits[0]` most significant) and
  bit-reversing it reproduces the little-endian `Circuit.to_unitary()`.
- `tests/domain/test_qasm3_roundtrip.py` — 26 tests over the writer's whole
  gate set, the `cu` gamma ordering, feed-forward, the strict/lenient contract
  and the QASM 2 ↔ QASM 3 importer split.
- `Circuit.draw(ascii_only=…)` — force the plain-ASCII rendering (or force the
  Unicode one); the default detects the stream's codec.
- **`sf.cat(mode="density_matrix")`, alias `"dm"`** — the last unimplemented
  lane, and the only one that can answer `h(0); cx(0,1); reset(0)`. `Reset` is
  the channel `rho -> Tr_q(rho) ⊗ |0><0|` (Kraus `{P0, |0><1|}`) and a
  non-selective readout is `P0 rho P0 + P1 rho P1` — a dephasing, not a
  collapse — both applied through `sf-ir`'s `apply_1q_superop`. Feed-forward
  stays refused: the lane returns one `rho`, not a mixture over
  branch-conditioned states, so a conditioned gate has no single matrix here.
  `CatPlan` carries `dm: Option<DmProgram>`, and `Exec` a `Rho(Vec<Complex64>)`.
- The density-matrix lane also applies **gates wider than two qubits**.
  `DensityMatrixState::apply_unitary` panics above two, so `DmProgram`
  transforms the ket axis by `U` and the bra axis by `U†` itself; Toffoli,
  CSWAP and opaque three-qubit unitaries work through `sf.cat` and match the
  statevector lane to 2.8e-17. `sf.run(..., method="density_matrix")` is
  unchanged and still panics on those, which is why the probe that exposed the
  gap now passes in one lane and not the other.
- `tests/unit/test_cat_density_matrix.py` — 12 tests: the exact mixed state
  after a reset (`diag(.5, 0, .5, 0)`, purity 0.5, von Neumann entropy ln 2),
  the same bits through `sf.run(method="density_matrix")` at
  `max |diff| = 0.0`, parameter rebinding through a reset, feed-forward
  refusal, a three-qubit gate against the statevector lane, and a Qiskit
  `DensityMatrix.from_instruction` cross-check behind `importorskip`.
- `tests/unit/test_cat_dynamic_guard.py` — 17 tests over the matrix of lanes
  against dynamic shapes, including every shape each lane must still refuse.

### Performance
- The stabilizer/clifford plan builder was rebuilt around a column-major
  tableau (`crates/sf-cat/src/colbuild.rs`), with a per-word column cache for
  the AG measurement pass and a fused row update in place of three passes over
  the same row. Cold A/B of the previous and the current binary in one window,
  per-call medians:

  ```text
  ghz:64          0.2099 -> 0.2084 ms   1.01x
  ghz:256         1.0060 -> 0.9819 ms   1.02x
  brick:256      11.6379 -> 11.6888 ms  1.00x
  brick:1024     53.9616 -> 53.8499 ms  1.00x
  randcliff:1024 122.592 -> 98.531 ms   1.24x
  ```

  The gain lands where the change points: at `randcliff:1024` the plan build
  itself went 85.7 -> 6.2 ms. The other four rungs are inside their own spread
  and are reported as unchanged, not as wins.
- `SF_CAT_THREADS` (or the workspace-wide `SF_THREADS`) caps this crate's
  rayon pool without touching the host application's global one, so batch
  hosts running several Python workers per machine stop paying for
  `workers x cores` runnable threads. Read once, at first use.

### Notes
- The RY/RZ adjoint results are bit-identical to 0.1.23 (same generator
  constants, same fused ≤2-Pauli accumulation, same cache), but the two arms
  were rewritten from a single-entry table lookup to explicit generator
  vectors to make room for the seven new gates, so the binary hash moved.
  Timings were re-measured on the new build; the drift is not attributable to
  this change, since the untouched Qulacs lane slowed by as much.
- Parameter-shift now costs two simulations per *occurrence* rather than per
  name; a name reused m times costs m times as much. Statevector users are
  unaffected (adjoint remains the default and is cheaper).
- No adjoint support is added for density-matrix, MPS or stabilizer states;
  they keep raising `MethodError` and using parameter-shift. The new `sf.cat`
  density-matrix lane is an execution lane, not a gradient one.
- Four of the `sf.cat` items above are deliberate behaviour changes, not
  silent relaxations: `adjoint` now *refuses* mid-circuit readouts,
  feed-forward and resets after use (it used to return the unitary-only
  gradient), and `statevector` now *accepts* a trailing readout and a leading
  reset. Three pre-existing tests encoded the old behaviour and were repointed
  at the refusal that must survive, not deleted.
- `SF_CAT_THREADS` is a bound on thread count, not a throughput feature. On 8
  logical CPUs, capping to one thread made `n=1024` across 8 concurrent
  processes 9% faster and `n=4096` 8% *slower*; the direction flips with
  circuit size, and aggregate CPU per circuit rises either way.
- The stored `ghz:64` number moved 0.20 -> 0.23 ms, which is a 15% step inside
  a 62.9% seven-run spread (min 0.1124, median 0.1299, max 0.1940 ms). It is
  not attributable to this release, and the same-window A/B of the two
  binaries gave 1.01x.
- The density-matrix lane exists for correctness, not speed. The feasibility
  probe measured a statistical tie with the routed Aer path, and nothing here
  changes that: it is not the fast path for a plain unitary circuit.
- The Python modules this release touches now pass the package's *own*
  declared ruff rule set (`E,F,I,N,W,UP`), not only the `F`-only gate CI runs:
  157 diagnostics went to 0 across `superfermion/__init__.py`, `bridge/`,
  `circuit.py`, `gradient/adjoint.py`, `gradient/parameter_shift.py` and
  `qasm_roundtrip.py`. That was 16 `typing` generics moved to builtins, ten
  local gate tables renamed to lowercase, ~100 blank lines that carried
  indentation, and four over-long lines wrapped. Two imports keep an explicit
  `noqa` because they are a PennyLane availability probe and a public
  re-export, and `superfermion/__init__.py` keeps `E402` as a per-file ignore —
  its imports have to follow the `State` alias or they reintroduce a circular
  import. The full suite is unchanged (1,112 passed, 2 skipped).
- Not measured: the modes benchmark notebook was not re-run (only `ghz:64`
  was refreshed), contention was measured on `brick` only, and every number
  above is one machine with 8 logical CPUs.

## [0.1.23] — 2026-10-31

### Fixed
- **Braket result order** — Braket keys every bitstring with the first measured
  qubit leftmost, SF reads q0 from the right (see `results.py`). Counts are now
  flipped into SF order, and untouched qubits — which Braket omits from the key
  entirely — are scattered back onto their real positions. Measured through
  `BraketDeviceExecutor.execute` against `LocalSimulator`: a 3-qubit circuit
  with q0 at 3% and q2 at 30% delivered `{q0: 0.0, q1: 0.2989, q2: 0.0289}`
  before the fix and `{q0: 0.0299, q1: 0.0, q2: 0.2971}` after (`<Z0>` +0.940
  vs +0.939 exact, `<Z2>` +0.406 vs +0.409; pre-fix those read +0.402 and
  +1.000).
- **Braket `RESET`** — `RESET` was in the skip list, so a circuit that reset a
  qubit ran with the reset missing. This SDK has no reset instruction, so
  `_to_braket` now refuses instead of silently dropping the operation.
- **IBM partial measurement** — `_ensure_measurements` only added
  `measure_all()` when *no* measurement existed, so `measure(0)` produced
  full-width keys with the untouched qubits reading 0 (verified on released
  0.1.22 against `FakeSherbrooke`). Circuits with an incomplete measurement are
  now completed on a copy before transpilation.
- **IonQ counts** — the adapter assigned the whole `/results` document to
  `RunResult.counts`, so `expectation()` raised `TypeError: unsupported operand
  type(s) for +: 'int' and 'str'`. `_extract_ionq_counts` unwraps `histogram`
  (or scales `probabilities` by the submitted shots) and converts IonQ's
  integer keys, which put qubit 0 in the most significant bit, into SF order.
- **Device width** — `runner` fetched `capabilities()` and discarded it, so a
  5-qubit circuit dispatched to a 2-qubit device and returned `{'00000': 8}`.
  `max_qubits` is now enforced for hardware executors; simulators and devices
  reporting 0 are left alone.
- **IBM capabilities** — `max_qubits` was hardcoded to 127, under-reporting
  156-qubit Herons. It now reads `backend.num_qubits` and keeps 127 as the
  offline fallback.
- **IBM job timeout** — a `timeout` kwarg was swallowed by `**kwargs`, so a job
  stuck behind a backlog blocked forever. It is forwarded to `job.result()`
  when the runtime accepts it, and an expired wait now re-raises the runtime's
  `RuntimeJobMaxTimeoutError` with the job id attached — the job keeps running
  server-side, so the handle is what lets the caller poll or cancel it.

### Added
- **Braket gate coverage** — `SDG`, `TDG`, `SX`, `P`, `CP`, `RXX`, `RYY`,
  `RZZ` map to `si`/`ti`/`v`/`phaseshift`/`cphaseshift`/`xx`/`yy`/`zz`, and `U`
  is emitted as `rz`-`ry`-`rz` (`U3 = e^{-i(phi+lam)/2} RZ(phi) RY(theta)
  RZ(lam)`). Each was checked against `LocalSimulator`: worst TVD 0.0058
  against a 0.0500 five-sigma floor.
- **OpenQASM 2 output** — `CY`, `CCX`, `CSWAP`, `P`, `CP`, `U`, `SX` and
  `RESET` emit qelib1 statements, and `ISWAP`, `RXX`, `RYY`, `RZZ` expand into
  exact `u3`/`cx`/`u1`/`s` sequences. Every snippet was round-tripped through
  Qiskit's OpenQASM 2.0 parser and compared against the gate unitary (max
  deviation 1e-11).
- **IonQ JSON coverage** — `CY`, `RXX`, `RYY`, `RZZ` map directly and `SX`
  emits `v` (IonQ's square-root-of-not; `sx` is not an API gate name). `P`, `U`
  and `CP` expand into `rz`/`ry`/`zz` sequences, `CCX` into the documented
  multi-controlled `not` form, and `CSWAP`/`ISWAP` into exact `cnot`/`not` and
  `swap`/`s`/`cz` decompositions — IonQ v0.3 documents no `ccnot`, `cswap` or
  `iswap` instruction. `CY` now emits `control`/`target` rather than `targets`.
  Parity checked by interpreting the emitted JSON back into a unitary (13
  gates, max deviation 2e-16).
- **Tests** — 31 new regression tests across the provider adapters, the QASM
  and IonQ converters, IBM measurement completion, IBM timeout handling and the
  runner width guard.

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
