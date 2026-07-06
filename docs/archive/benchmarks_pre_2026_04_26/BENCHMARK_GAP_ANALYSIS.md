# Benchmark gap analysis — why SF doesn't match Qiskit / PennyLane on some rows

Date: 2026-04-18  •  Scope: `QML_INDUSTRY_BENCHMARK.md`

## The three measurable gaps

| Row | SF | Qiskit | PennyLane | Gap size |
|-----|------|---------|-----------|---------|
| **T1 measurement max \|err\|** | 3.59×10⁻⁸ | 2.22×10⁻¹⁶ | 7.77×10⁻¹⁶ | ≈10⁸× looser |
| **T1 measurement latency** | 250 ms | 52 ms | 96 ms | 5× / 2.5× slower |
| **T2 VQE \|ΔE\|** | 6.05×10⁻⁷ Ha | 3.02×10⁻⁸ Ha | 3.02×10⁻⁸ Ha | 20× looser |
| **T3 Moons test acc** | 76.0% | n/a | 80.0% | 4 pp |
| **T3 BCW test acc** | 67.1% | n/a | 78.3% | 11 pp |

Everywhere else SF wins by 16–192× on latency and 3–25× on memory — so these five rows are the only diagnoses worth doing.

## Root causes

### RC-1 — JAX backend is hard-locked to `complex64`
File: `superfermion/backends/jax_sim.py`. Every gate matrix, every initial state, every buffer uses `dtype=jnp.complex64` — **28 call-sites** in that one file. Confirmed:
- NumPy `statevector` backend: correctly uses `complex128`.
- NumPy `simulator` backend: correctly uses `complex128`.
- `density_matrix` backend: correctly uses `complex128`.
- `qml.gradient.{adjoint, parameter_shift, core}`: correctly use `complex128`.
- **`jax_sim.py`: uses `complex64` everywhere** — this is the sole source of the precision gap.

When our benchmark calls `sf.qml.circuit_to_jax(..., backend="jax")`, the statevector lives in complex64 → amplitudes carry ≈10⁻⁷ noise → observables inherit that → VQE gradient-norm threshold trips at 10⁻⁶ instead of 10⁻¹⁵. This also leaks into T3 (Moons / BCW): Adam's updates are polluted by complex64 rounding, so training plateaus earlier than PennyLane's complex128 equivalent.

**Impact if fixed**: T1 max|err| → ≈10⁻¹⁵; T2 |ΔE| → ≈10⁻⁸; T3 accuracy gap likely closes by ~2–4 pp.

### RC-2 — T1 measures cold-start latency, not steady-state
T1 is a one-shot forward pass. SF's 250 ms is dominated by:
- Module load for `superfermion.backends.*` (lazy-imported backend registry).
- ClusterManager `jax.devices()` scan.
- First JAX trace of the expectation path.

Qiskit `StatevectorEstimator` has a tiny, single-purpose init; PennyLane `default.qubit` lazy-traces per qnode. Neither has SF's service-oriented registry overhead.

**Impact if fixed** (warmup + steady-state): SF's per-measurement cost should drop to < 5 ms, matching or beating the others. This is not a correctness fix — it's benchmarking hygiene.

### RC-3 — T3 uses a fixed epoch count, not a fixed wall-clock budget
Current T3 runs 25 Adam epochs for both SF and PennyLane. SF finishes Iris in 10 s; PennyLane takes 241 s. Giving the slower framework the same epoch count is fair by one definition (same optimiser steps) but unfair by another (same time budget). On BCW, PennyLane's 25 epochs take 1170 s — SF could run **~4800 epochs** in that same wall-clock and almost certainly close the accuracy gap.

**Impact if fixed** (wall-clock-budget training): SF likely matches PennyLane on Moons and BCW accuracy while still being faster per epoch.

### RC-4 — We only benchmarked 2 of SF's 12 registered backends
SF has named backends: `statevector`, `jax`, `rust`, `mps`, `singularity`, `density_matrix`, `cuda`, `cuda_mps`, `jax_mps`, `cluster`, `cupy_sim`, `supremacy`, `dwave`. The current sweep only tested `statevector` + `jax` for SF. On CPU we should also cover `rust`, `mps`, `singularity`, `density_matrix`. Known-broken per user memory: `jax_mps`, `supremacy`. GPU: `cuda`, `cuda_mps`, `cupy_sim`. Distributed: `cluster`. Stub: `dwave`. Registered on this box (per `sf.list_backends()`): `statevector`, `jax`, `cuda`, `mps`, `singularity`.

**Impact if fixed**: for users evaluating SF, they see a per-backend table — including a clear statement of which backends are production-ready at which qubit counts.

## Fix plan

### F1 — Complex128 toggle in `jax_sim.py`  (~45 min)
Add a module-level `_DTYPE` constant that resolves at import time:
```python
from jax import config
_DTYPE = jnp.complex128 if config.jax_enable_x64 else jnp.complex64
```
Replace all 28 hard-coded `jnp.complex64` with `_DTYPE`. Add a `set_dtype(dtype)` helper so the benchmark can flip it without touching env vars. No API break — default behaviour unchanged.

Verification: after `jax.config.update("jax_enable_x64", True)` and `set_dtype(jnp.complex128)`, `sf.qml.circuit_to_jax(..., backend="jax")` must produce complex128 statevectors — checked by probing `sv.dtype`.

### F2 — Benchmark warmup pass for T1  (~15 min)
In `sf_measurement_task`, `qiskit_measurement_task`, `pl_measurement_task`: call the full measurement pipeline once (discarded), then measure. Report both cold + warm times explicitly.

### F3 — Wall-clock-budget T3 variant  (~30 min)
New task T3b: same setup, but train until wall-clock ≥ `BUDGET_SECONDS` (set to the slowest framework's fixed-epoch total). Every framework gets the same total seconds; report final accuracy + number of epochs completed.

### F4 — Per-backend sweep for SF  (~45 min)
For T1 and T2, add a `variant` loop over `["statevector", "jax", "rust", "mps", "singularity", "density_matrix"]`. Skip any that aren't registered on this box. Record per-backend latency, memory, accuracy. Produce a matrix table in the report.

### F5 — Regenerate report  (~10 min)
Extend `write_report()` to show:
- Cold vs warm T1 latency side-by-side.
- Per-backend SF table for T1 + T2.
- Before/after complex128 VQE error.
- Wall-clock-budget T3 accuracy.

## Exit criteria

- SF's T2 |ΔE| ≤ 5×10⁻⁸ Ha (matching Qiskit/PL to within 2×).
- SF's T1 warm latency ≤ Qiskit's warm latency (or at most 1.5×).
- SF's T3 BCW accuracy ≥ 75% within a matched wall-clock budget.
- Backend matrix printed for 4–5 SF backends.
- Every table in the final `QML_INDUSTRY_BENCHMARK.md` is reproducible from one command.
