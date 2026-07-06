"""
MIT License - CERN Quantum Analysis Framework Files

This directory contains a complete end-to-end quantum analysis system
for CERN particle collision data using SuperFermion with PennyLane validation.

QUICK START GUIDE
=================

1. Ensure dependencies:
   pip install superfermion pennylane jax jaxlib numpy scipy

2. Run the pipeline:
   python cern_quantum_pipeline.py --dataset higgs --n-events 500

3. Check reports in /reports directory:
   - higgs_analysis_report.md (summary)
   - higgs_results.json (detailed results)
   - higgs_analysis_report.html (visualization)

FILES OVERVIEW
==============

├── cern_data_pipeline.py
│   └─ Data ingestion, preprocessing, statistics
│      • CERNOpenDataDownloader: Fetch from CERN portal
│      • CollisionEventPreprocessor: Generate realistic events
│      • CERNDataStats: Compute mass spectra
│
├── superfermion_quantum_circuits.py
│   └─ Quantum circuit design & execution
│      • SuperFermionCollisionAnalyzer: Circuit builder
│      • BatchQuantumAnalyzer: Batch processing
│      • QuantumAnalysisResult: Output dataclass
│
├── pennylane_validator.py
│   └─ Classical validation using PennyLane
│      • PennyLaneValidationEngine: Single device validator
│      • MultiDeviceValidator: Cross-device consistency
│      • GroundTruthResult: Validation output
│
├── comparison_framework.py
│   └─ Fidelity metrics between SuperFermion & PennyLane
│      • QuantumFidelityCompute: State comparison
│      • DistributionComparison: KL, Hellinger, Wasserstein
│      • ComparisonFramework: Batch analysis
│
├── scientific_report_engine.py
│   └─ Report generation
│      • ScientificReport: Markdown/JSON/HTML
│      • PipelineOrchestrator: Full pipeline execution
│
├── cern_quantum_pipeline.py
│   └─ CLI entry point
│      • Main orchestrator with argument parsing
│
└── CERN_QUANTUM_ANALYSIS_README.md
    └─ Comprehensive documentation

EXAMPLE USAGE
=============

Basic analysis:
    python cern_quantum_pipeline.py

Custom settings:
    python cern_quantum_pipeline.py \\
        --dataset z_mumu \\
        --n-events 1000 \\
        --n-qubits 14 \\
        --output-dir my_results

With verbose logging:
    python cern_quantum_pipeline.py --verbose

DATASET TYPES
=============

1. higgs    → Higgs→γγ, M=125 GeV (5% signal)
2. z_mumu   → Z→μμ, M=91 GeV (70% signal)
3. mixed    → Combined signal+background

COMPUTATIONAL REQUIREMENTS
===========================

Small (n_events=100, n_qubits=8):
  Time: ~2 minutes
  Memory: ~500 MB

Medium (n_events=500, n_qubits=12):
  Time: ~10 minutes
  Memory: ~1.5 GB

Large (n_events=1000, n_qubits=14):
  Time: ~30 minutes
  Memory: ~3 GB

GPU NOT REQUIRED (but can accelerate with JAX GPU backend)

INTERPRETING RESULTS
====================

Markdown Report (higgs_analysis_report.md):
  ✓ Executive summary with event counts
  ✓ Invariant mass spectrum statistics
  ✓ SuperFermion quantum results
  ✓ PennyLane validation metrics
  ✓ Event classification table

JSON Results (higgs_results.json):
  ✓ Programmatic access to all metrics
  ✓ Event-by-event predictions
  ✓ Comparison scores
  ✓ Confidence levels

Key Metrics to Check:
  1. Mean Anomaly Score: Should be ~0.5 for mixed events
  2. Mass Error: Should be <5 GeV for trained circuits
  3. Confidence Level: >0.8 indicates reliable results
  4. High Confidence Fraction: >80% indicates good agreement

TROUBLESHOOTING
===============

Issue: ImportError for superfermion
  Fix: pip install superfermion --upgrade

Issue: PennyLane device not available
  Fix: Use default device (automatic fallback)

Issue: Out of memory
  Fix: Reduce --n-qubits to 8 or --n-events to 100

Issue: Slow execution
  Fix: Reduce batch size or use fewer qubits

EXTENDING THE FRAMEWORK
=======================

To add a new physics process:

1. Edit cern_data_pipeline.py:
   - Add generator method generate_synthetic_XXX()
   - Include realistic kinematics + mass spectrum

2. Edit superfermion_quantum_circuits.py:
   - Optionally customize feature encoding

3. Run pipeline with new dataset

To implement real CERN data:

1. Download ROOT files from opendata.cern.ch
2. Add ROOT parser (requires uproot package)
3. Convert ROOT events → CollisionEvent objects
4. Run through existing pipeline

To validate on real quantum hardware:

1. Replace PennyLane validator with hardware provider
2. Add error mitigation (QAOA-style techniques)
3. Adjust circuit depth for device constraints

SCIENCE REFERENCES
==================

Quantum Machine Learning for Particle Physics:
  - Havlícek et al. (2019): "Feature maps for quantum ML"
  - Schuld & Killoran (2022): "Quantum ML on quantum computers"

Fidelity Measures:
  - Schumacher (1996): "Von Neumann Entropy"
  - Jozsa (1994): "Fidelity for quantum channels"

LHC Physics:
  - Higgs discovery: Aad et al., PRL 112 (2014)
  - Standard Model: Olive et al., PDG 2014

SUPPORT
=======

For documentation: See CERN_QUANTUM_ANALYSIS_README.md
For examples: Run with --verbose flag to see executed code
For issues: Check exception messages in pipeline output

Last Updated: 2026-04-06
Version: 1.0 Production
"""

print(__doc__)
