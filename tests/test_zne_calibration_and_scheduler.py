"""
Integration tests for:
- Hardware noise model (ZNE → calibration wiring)
- Cloud Job Scheduler
- Verification that sf.train() / sf.Pipeline / Rust SABRE are present
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.curdir))

import jax
import jax.numpy as jnp
import numpy as np

import superfermion as sf


# ═══════════════════════════════════════════════════════════════════════
# Test 1: ZNE → Calibration wiring
# ═══════════════════════════════════════════════════════════════════════

def test_calibration_extract_noise_params():
    """CalibrationSet.extract_noise_params() computes plausible noise params."""
    from superfermion.pulse.calibration import CalibrationSet

    cals = CalibrationSet("ibm_test", dt=0.222)
    cals.add_default_single_qubit(0)
    cals.add_default_single_qubit(1)
    cals.add_default_two_qubit(0, 1)

    params = cals.extract_noise_params()

    assert "avg_1q_fidelity" in params
    assert "avg_2q_fidelity" in params
    assert "depolarizing_1q" in params
    assert "depolarizing_2q" in params
    assert "readout_error" in params
    assert "noise_factors" in params
    assert params["noise_factors"] == [1, 2, 3]

    # Fidelity should be high (default calibrations are near-perfect)
    assert params["avg_1q_fidelity"] > 0.99
    assert params["avg_2q_fidelity"] > 0.99
    # Noise params should be small
    assert params["depolarizing_1q"] < 0.01
    assert params["depolarizing_2q"] < 0.01

    print(f"  [PASS] extract_noise_params: 1Q_fid={params['avg_1q_fidelity']:.6f}, "
          f"2Q_fid={params['avg_2q_fidelity']:.6f}, "
          f"dp_1q={params['depolarizing_1q']:.6f}")


def test_calibration_to_noise_model():
    """CalibrationSet.to_noise_model() returns a valid NoiseModel."""
    from superfermion.pulse.calibration import CalibrationSet
    from superfermion.noise import NoiseModel

    cals = CalibrationSet("ibm_test", dt=0.222)
    cals.add_default_single_qubit(0)
    cals.add_default_single_qubit(1)
    cals.add_default_two_qubit(0, 1)

    nm = cals.to_noise_model()
    assert isinstance(nm, NoiseModel)
    assert len(nm.single_qubit_channels) >= 1  # At least depolarizing
    assert nm.readout_error > 0  # Non-zero readout error

    print(f"  [PASS] to_noise_model: {nm}")


def test_zne_with_calibration():
    """zne_with_calibration() runs ZNE driven by CalibrationSet data."""
    from superfermion.pulse.calibration import CalibrationSet
    from superfermion.mitigation import zne_with_calibration

    cals = CalibrationSet("ibm_test", dt=0.222)
    cals.add_default_single_qubit(0)
    cals.add_default_single_qubit(1)

    c = sf.Circuit(2)
    c.h(0).cx(0, 1)

    def observable(sv):
        # Expectation of Z on qubit 0: P(|0>) - P(|1>)
        probs = jnp.abs(sv.reshape(-1))**2
        return float(jnp.sum(probs[:2]) - jnp.sum(probs[2:]))

    result = zne_with_calibration(c, observable, cals, scale_factors=[1, 2, 3])

    assert "zne_value" in result
    assert "raw_values" in result
    assert "noise_params" in result
    assert len(result["raw_values"]) == 3
    assert "avg_1q_fidelity" in result["noise_params"]

    print(f"  [PASS] zne_with_calibration: ZNE={result['zne_value']:.6f}, "
          f"raw={[f'{v:.4f}' for v in result['raw_values']]}")


def test_zne_with_noise_model():
    """zne_with_calibration() works with a NoiseModel directly."""
    from superfermion.noise import ibm_eagle_noise
    from superfermion.mitigation import zne_with_calibration

    nm = ibm_eagle_noise()

    c = sf.Circuit(1)
    c.h(0)

    def observable(sv):
        return float(jnp.real(jnp.abs(sv[0])**2 - jnp.abs(sv[1])**2))

    result = zne_with_calibration(c, observable, nm, scale_factors=[1, 2, 3])

    assert "zne_value" in result
    assert result["noise_params"]["readout_error"] == 0.01
    print(f"  [PASS] zne_with_noise_model: ZNE={result['zne_value']:.6f}")


def test_calibration_based_noise_model():
    """calibration_based_noise_model() builds NoiseModel from backend name."""
    from superfermion.mitigation import calibration_based_noise_model
    from superfermion.noise import NoiseModel

    nm = calibration_based_noise_model("ibm_brisbane")
    assert isinstance(nm, NoiseModel)
    assert len(nm.single_qubit_channels) > 0
    assert nm.readout_error > 0
    print(f"  [PASS] calibration_based_noise_model: {nm}")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Cloud Job Scheduler
# ═══════════════════════════════════════════════════════════════════════

def test_scheduler_imports():
    """All scheduler classes are importable from runtime."""
    from superfermion.runtime.scheduler import (
        CloudScheduler, SchedulerJob, JobPriority,
        SchedulingPolicy, BackendRegistration, BatchResult,
    )

    # Enums have expected values
    assert JobPriority.CRITICAL.value == 5
    assert JobPriority.LOW.value == 2
    assert SchedulingPolicy.PRIORITY.value == "priority"
    assert SchedulingPolicy.COST_AWARE.value == "cost_aware"
    print("  [PASS] Scheduler imports & enums")


def test_scheduler_create():
    """CloudScheduler can be created and started."""
    from superfermion.runtime.scheduler import CloudScheduler

    scheduler = CloudScheduler(max_workers=2)
    scheduler.register_backend("jax", provider="local")
    scheduler.register_backend("statevector", provider="local")

    assert "jax" in scheduler.list_backends()
    assert "statevector" in scheduler.list_backends()

    scheduler.start()
    assert scheduler._running.is_set()
    scheduler.stop()
    print("  [PASS] Scheduler create/start/stop")


def test_scheduler_submit_and_wait():
    """Submit a job and wait for result."""
    from superfermion.runtime.scheduler import CloudScheduler, JobPriority

    scheduler = CloudScheduler(max_workers=4)
    scheduler.register_backend("jax", provider="local")

    c = sf.Circuit(2)
    c.h(0).cx(0, 1)

    job_id = scheduler.submit(c, backend="jax", priority=JobPriority.HIGH)
    assert job_id is not None
    assert len(job_id) == 36  # UUID

    result = scheduler.wait_for(job_id, timeout=30)
    assert result is not None
    assert result.counts is not None

    # Check job status
    status = scheduler.status(job_id)
    assert status.value == "completed"

    scheduler.stop()
    print(f"  [PASS] Scheduler submit+wait: {job_id[:8]}... "
          f"counts={result.counts}")


def test_scheduler_batch():
    """Submit a batch of circuits."""
    from superfermion.runtime.scheduler import CloudScheduler, JobPriority

    scheduler = CloudScheduler(max_workers=4)
    scheduler.register_backend("jax", provider="local")

    circuits = []
    for i in range(3):
        c = sf.Circuit(1)
        c.h(0)
        circuits.append(c)

    batch = scheduler.submit_batch(circuits, backend="jax", priority=JobPriority.BATCH)
    assert len(batch.job_ids) == 3

    for jid in batch.job_ids:
        result = scheduler.wait_for(jid, timeout=30)
        assert result is not None

    scheduler.stop()
    print(f"  [PASS] Scheduler batch: {len(batch.job_ids)} jobs completed")


def test_scheduler_metrics():
    """Metrics correctly track job states."""
    from superfermion.runtime.scheduler import CloudScheduler

    scheduler = CloudScheduler(max_workers=4)
    scheduler.register_backend("jax", provider="local")

    c = sf.Circuit(1)
    c.h(0)

    jid1 = scheduler.submit(c, backend="jax")
    jid2 = scheduler.submit(c, backend="jax")

    scheduler.wait_for(jid1, timeout=30)
    scheduler.wait_for(jid2, timeout=30)

    m = scheduler.metrics()
    assert m["total_jobs"] == 2
    assert m["completed"] == 2
    assert m["failed"] == 0
    assert m["avg_wait_ms"] >= 0

    scheduler.stop()
    print(f"  [PASS] Scheduler metrics: completed={m['completed']}, "
          f"avg_wait={m['avg_wait_ms']:.1f}ms")


def test_scheduler_dependencies():
    """Jobs with dependencies wait for predecessors."""
    from superfermion.runtime.scheduler import CloudScheduler, JobPriority

    scheduler = CloudScheduler(max_workers=4)
    scheduler.register_backend("jax", provider="local")

    c = sf.Circuit(1)
    c.h(0)

    # Submit job A
    jid_a = scheduler.submit(c, backend="jax", priority=JobPriority.HIGH)
    result_a = scheduler.wait_for(jid_a, timeout=30)
    assert result_a is not None

    # Submit job B that depends on A
    jid_b = scheduler.submit(
        c, backend="jax",
        dependencies=[jid_a],
        priority=JobPriority.NORMAL,
    )

    result_b = scheduler.wait_for(jid_b, timeout=30)
    assert result_b is not None

    # Both completed
    assert scheduler.status(jid_a).value == "completed"
    assert scheduler.status(jid_b).value == "completed"

    scheduler.stop()
    print(f"  [PASS] Scheduler dependencies: A={jid_a[:8]} -> B={jid_b[:8]}")


def test_scheduler_context_manager():
    """CloudScheduler works as context manager."""
    from superfermion.runtime.scheduler import CloudScheduler

    with CloudScheduler(max_workers=2) as scheduler:
        scheduler.register_backend("jax", provider="local")
        c = sf.Circuit(1)
        c.h(0)
        jid = scheduler.submit(c, backend="jax")
        result = scheduler.wait_for(jid, timeout=30)
        assert result is not None

    print("  [PASS] Scheduler context manager")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Verify existing items (sf.train, sf.Pipeline, Rust SABRE)
# ═══════════════════════════════════════════════════════════════════════

def test_sf_train_present():
    """sf.train() function exists and is importable."""
    from superfermion.train import train, TrainState
    assert callable(train)
    assert TrainState is not None
    print("  [PASS] sf.train() is present")


def test_sf_pipeline_present():
    """sf.Pipeline class exists and is functional."""
    from superfermion.pipeline import Pipeline, make_pipeline

    X = np.random.randn(5, 3)

    # Functional pipeline
    pipe = make_pipeline(
        lambda x: x * 2.0,
        lambda x: x + 1.0,
    )
    out = pipe.execute(X)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out, X * 2.0 + 1.0)

    # Named pipeline
    pipe2 = Pipeline([
        ("scale", lambda x: x * 3.0),
        ("shift", lambda x: x - 1.0),
    ])
    out2 = pipe2.execute(X)
    assert out2.shape == (5, 3)

    print("  [PASS] sf.Pipeline is present and functional")


def test_rust_sabre_present():
    """Rust SABRE router crate exists."""
    import os
    # Workspace root is parent of tests/
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    crate_path = os.path.join(workspace_root, "crates", "sf-router")
    assert os.path.isdir(crate_path), "sf-router crate directory missing"

    # Check key files
    for fname in ["Cargo.toml", "src", "src/lib.rs", "src/sabre.rs",
                   "src/topology.rs", "src/layout.rs", "src/token_swap.rs"]:
        fpath = os.path.join(crate_path, fname)
        assert os.path.exists(fpath), f"Missing: {fpath}"

    # Check Cargo.toml content
    with open(os.path.join(crate_path, "Cargo.toml"), "r") as f:
        cargo_content = f.read()
    assert 'name = "sf-router"' in cargo_content
    assert 'sf-ir' in cargo_content

    print("  [PASS] Rust SABRE router crate is present")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Testing: ZNE → Calibration Wiring + Cloud Scheduler")
    print("=" * 60)

    # ── ZNE-Calibration tests ──────────────────────────────────────
    print("\n[ZNE → Calibration]")
    try:
        test_calibration_extract_noise_params()
    except Exception as e:
        print(f"  [FAIL] extract_noise_params: {e}")
        import traceback; traceback.print_exc()

    try:
        test_calibration_to_noise_model()
    except Exception as e:
        print(f"  [FAIL] to_noise_model: {e}")
        import traceback; traceback.print_exc()

    try:
        test_zne_with_calibration()
    except Exception as e:
        print(f"  [FAIL] zne_with_calibration: {e}")
        import traceback; traceback.print_exc()

    try:
        test_zne_with_noise_model()
    except Exception as e:
        print(f"  [FAIL] zne_with_noise_model: {e}")
        import traceback; traceback.print_exc()

    try:
        test_calibration_based_noise_model()
    except Exception as e:
        print(f"  [FAIL] calibration_based_noise_model: {e}")
        import traceback; traceback.print_exc()

    # ── Scheduler tests ────────────────────────────────────────────
    print("\n[Cloud Job Scheduler]")
    try:
        test_scheduler_imports()
    except Exception as e:
        print(f"  [FAIL] imports: {e}")
        import traceback; traceback.print_exc()

    try:
        test_scheduler_create()
    except Exception as e:
        print(f"  [FAIL] create: {e}")
        import traceback; traceback.print_exc()

    try:
        test_scheduler_submit_and_wait()
    except Exception as e:
        print(f"  [FAIL] submit+wait: {e}")
        import traceback; traceback.print_exc()

    try:
        test_scheduler_batch()
    except Exception as e:
        print(f"  [FAIL] batch: {e}")
        import traceback; traceback.print_exc()

    try:
        test_scheduler_metrics()
    except Exception as e:
        print(f"  [FAIL] metrics: {e}")
        import traceback; traceback.print_exc()

    try:
        test_scheduler_dependencies()
    except Exception as e:
        print(f"  [FAIL] dependencies: {e}")
        import traceback; traceback.print_exc()

    try:
        test_scheduler_context_manager()
    except Exception as e:
        print(f"  [FAIL] context manager: {e}")
        import traceback; traceback.print_exc()

    # ── Verification: existing items ──────────────────────────────
    print("\n[Verification: existing items]")
    try:
        test_sf_train_present()
    except Exception as e:
        print(f"  [FAIL] sf.train: {e}")

    try:
        test_sf_pipeline_present()
    except Exception as e:
        print(f"  [FAIL] sf.Pipeline: {e}")

    try:
        test_rust_sabre_present()
    except Exception as e:
        print(f"  [FAIL] Rust SABRE: {e}")

    print("\n" + "=" * 60)
    print("All tests complete.")
