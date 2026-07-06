# Archive — Benchmarks pre-2026-04-26

These benchmark reports were generated **before** the 2026-04-26 series of
performance fixes (Fix #1 through Fix #10) and the 2026-04-27/28 follow-up
work on adjoint differentiation and the QR-based MPS sweep.

**They under-report SuperFermion's current performance by 5×–30×** on most
cells.  They are kept here for historical / regression reference only.

For current benchmarks, see:

- `docs/benchmarks.md` — the canonical, current scoreboard
- `BENCH_ALL_BACKENDS.md` (project root) — most recent full backend sweep
- `BENCH_QML.json` (project root) — most recent QML / gradient bench
- `BENCH_INDUSTRY.json` (project root) — most recent industry-standard bench
- `bench_publication.py` (project root) — reproducible publication-grade bench

## Index of archived files

| File | Era | What it covered |
|---|---|---|
| `BENCHMARK_REPORT.md` | 2026-03-29 | early sf-vs-qiskit |
| `BENCHMARK_SCALING_PROOF.md` | 2026-04-19 | sf scaling at n=10–22 |
| `BENCHMARK_SCALING_PROOF_V2.md` | 2026-04-19 | revised version |
| `BENCHMARK_EXPLANATION.md` | 2026-04-07 | bench methodology |
| `BENCHMARK_GAP_ANALYSIS.md` | 2026-04-18 | gap analysis driving Fix #1–#5 |
| `CANONICAL_BENCHMARK.md` | 2026-04-19 | canonical run |
| `COMPREHENSIVE_ACCURACY_V2_REPORT.md` | 2026-04-09 | accuracy verification |
| `EXACT_FAILURE_ANALYSIS.md` | 2026-04-07 | failure-mode analysis |
| `INDUSTRIAL_BENCHMARK_FULL_REPORT.md` | various | industrial workload sweep |
| `INDUSTRY_BENCHMARK_REPORT.md` | various | industry comparison |
| `MPS_TN_BENCHMARK.md` | various | MPS / tensor-network bench |
| `QML_INDUSTRY_BENCHMARK.md` | various | QML industry comparison |
| `SCIENTIFIC_ACCURACY_FINAL_REPORT.md` | various | accuracy final |
| `comprehensive_mps_showdown_report.md` | various | MPS showdown |
| `ina_industry_benchmark_report.md` | various | industry bench (typo) |
| `scaling_supremacy_report.md` | various | scaling sweep |
| `scientific_accuracy_report.md` | various | accuracy report |
| `sf_industry_benchmark.md` | various | sf industry bench |
| `ultimate_industry_benchmark_report.md` | various | "ultimate" bench |

If you cite SF in a paper, cite `docs/benchmarks.md` (current data) plus
the publication-grade `bench_publication.py` JSON output.
