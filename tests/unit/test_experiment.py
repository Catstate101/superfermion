"""Unit tests for experiment tracking and TrackerProtocol."""

import json
import threading
from unittest.mock import patch

import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.experiment.context import _get_active_tracker, experiment
from superfermion.experiment.local_tracker import LocalTracker
from superfermion.experiment.protocols import TrackerProtocol
from superfermion.results import RunResult


pytestmark = pytest.mark.unit


class CustomTracker:
    def __init__(self):
        self.events = []

    def on_run_start(self, circuit, device, shots, metadata=None):
        self.events.append(("start", device, shots))

    def on_run_complete(self, result, metadata=None):
        self.events.append(("complete", result))

    def on_run_error(self, error, metadata=None):
        self.events.append(("error", error))


class TestTrackerProtocol:
    def test_custom_tracker_satisfies_protocol(self):
        tracker = CustomTracker()
        assert isinstance(tracker, TrackerProtocol)


class TestExperimentContext:
    def test_no_active_tracker_outside_block(self):
        assert _get_active_tracker() is None

    def test_experiment_sets_and_resets_tracker(self, mock_tracker):
        assert _get_active_tracker() is None
        with sf.experiment("test-run", tracker=mock_tracker) as active:
            assert active is mock_tracker
            assert _get_active_tracker() is mock_tracker
        assert _get_active_tracker() is None

    def test_nested_experiments_restore_outer(self):
        outer = CustomTracker()
        inner = CustomTracker()
        with sf.experiment("outer", tracker=outer):
            assert _get_active_tracker() is outer
            with sf.experiment("inner", tracker=inner):
                assert _get_active_tracker() is inner
            assert _get_active_tracker() is outer
        assert _get_active_tracker() is None


class TestThreadSafety:
    def test_concurrent_experiments_have_independent_trackers(self):
        results = {}
        barrier = threading.Barrier(2)

        def run_experiment(name, tracker):
            with sf.experiment(name, tracker=tracker):
                barrier.wait(timeout=5)
                results[name] = _get_active_tracker()

        t1 = threading.Thread(
            target=run_experiment, args=("exp-a", CustomTracker()),
        )
        t2 = threading.Thread(
            target=run_experiment, args=("exp-b", CustomTracker()),
        )
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["exp-a"] is not results["exp-b"]
        assert isinstance(results["exp-a"], CustomTracker)
        assert isinstance(results["exp-b"], CustomTracker)


class TestLocalTracker:
    def test_lifecycle_writes_json(self, tmp_tracker_dir, bell_circuit):
        tracker = LocalTracker("bell-test", base_dir=str(tmp_tracker_dir))
        result = RunResult(counts={"00": 100}, shots=100)

        tracker.on_run_start(bell_circuit, "cpu", 100)
        tracker.on_run_complete(result)

        run_dir = tmp_tracker_dir / "bell-test"
        assert run_dir.is_dir()
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1

        record = json.loads(files[0].read_text())
        assert record["device"] == "cpu"
        assert record["shots"] == 100
        assert record["n_qubits"] == 2
        assert "completed_at" in record

    def test_on_run_error_persists_failure(self, tmp_tracker_dir, bell_circuit):
        tracker = LocalTracker("fail-test", base_dir=str(tmp_tracker_dir))
        tracker.on_run_start(bell_circuit, "cpu", 50)
        tracker.on_run_error(RuntimeError("sim failed"))

        files = list((tmp_tracker_dir / "fail-test").glob("run_*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record["error_type"] == "RuntimeError"
        assert "sim failed" in record["error"]

    def test_experiment_creates_local_tracker_by_default(self, tmp_tracker_dir):
        from superfermion.experiment import local_tracker

        with patch.object(
            local_tracker,
            "LocalTracker",
            side_effect=lambda name: LocalTracker(name, base_dir=str(tmp_tracker_dir)),
        ):
            with sf.experiment("auto-track") as tracker:
                assert isinstance(tracker, LocalTracker)
                assert tracker.name == "auto-track"
