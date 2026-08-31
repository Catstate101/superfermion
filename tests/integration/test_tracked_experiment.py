"""Integration tests for experiment tracking end-to-end."""

import json

import pytest

import superfermion as sf
from superfermion.devices import DeviceCapabilities
from superfermion.experiment.local_tracker import LocalTracker
from superfermion.experiment.protocols import TrackerProtocol


pytestmark = pytest.mark.integration


class RunTracker:
    """Lightweight tracker matching conftest MockTracker."""

    def __init__(self):
        self.starts = []
        self.completions = []
        self.errors = []

    def on_run_start(self, circuit, device, shots, metadata=None):
        self.starts.append({"circuit": circuit, "device": device, "shots": shots})

    def on_run_complete(self, result, metadata=None):
        self.completions.append({"result": result})

    def on_run_error(self, error, metadata=None):
        self.errors.append(error)


class FailingExecutor:
    """DeviceExecutor that always raises, for error-path integration tests."""

    def execute(self, circuit, shots=1000, **kwargs):
        raise RuntimeError("simulation failed")

    def capabilities(self):
        return DeviceCapabilities(max_qubits=32)


class TestTrackedExperiment:
    def test_experiment_context_tracks_run(self, bell_circuit, mock_tracker):
        with sf.experiment("test", tracker=mock_tracker):
            result = sf.run(bell_circuit, device="cpu", shots=200)

        assert len(mock_tracker.starts) == 1
        assert len(mock_tracker.completions) == 1
        assert len(mock_tracker.errors) == 0

        start = mock_tracker.starts[0]
        assert start["shots"] == 200
        assert start["device"] == "cpu"
        assert start["circuit"].n_qubits == bell_circuit.n_qubits

        completion = mock_tracker.completions[0]
        assert completion["result"].shots == 200
        assert sum(completion["result"].counts.values()) == 200

    def test_multiple_runs_tracked_in_sequence(self, bell_circuit, mock_tracker):
        with sf.experiment("multi-run", tracker=mock_tracker):
            sf.run(bell_circuit, device="cpu", shots=100)
            sf.run(bell_circuit, device="cpu", shots=250)
            sf.run(bell_circuit, device="cpu", shots=150)

        assert len(mock_tracker.starts) == 3
        assert len(mock_tracker.completions) == 3
        assert [s["shots"] for s in mock_tracker.starts] == [100, 250, 150]

    def test_tracker_receives_correct_start_and_complete_payloads(
        self, bell_circuit, mock_tracker,
    ):
        shots = 400
        with sf.experiment("payload-check", tracker=mock_tracker):
            sf.run(bell_circuit, device="cpu", shots=shots)

        start = mock_tracker.starts[0]
        assert start["circuit"].gate_count == bell_circuit.gate_count
        assert start["device"] == "cpu"
        assert start["shots"] == shots

        result = mock_tracker.completions[0]["result"]
        assert result.shots == shots
        assert "00" in result.counts
        assert "11" in result.counts

    def test_error_inside_experiment_calls_on_run_error(self, bell_circuit, mock_tracker):
        with sf.experiment("error-test", tracker=mock_tracker):
            with pytest.raises(RuntimeError, match="simulation failed"):
                sf.run(bell_circuit, device=FailingExecutor(), shots=50)

        assert len(mock_tracker.starts) == 1
        assert len(mock_tracker.completions) == 0
        assert len(mock_tracker.errors) == 1
        assert isinstance(mock_tracker.errors[0], RuntimeError)

    def test_experiment_closes_cleanly_after_error(self, bell_circuit, mock_tracker):
        from superfermion.experiment.context import _get_active_tracker

        with sf.experiment("cleanup", tracker=mock_tracker):
            with pytest.raises(RuntimeError):
                sf.run(bell_circuit, device=FailingExecutor(), shots=10)
            assert _get_active_tracker() is mock_tracker

        assert _get_active_tracker() is None

        result = sf.run(bell_circuit, device="cpu", shots=100)
        assert result.shots == 100

    def test_nested_experiments_route_to_correct_trackers(self, bell_circuit):
        outer = RunTracker()
        inner = RunTracker()

        with sf.experiment("outer", tracker=outer):
            sf.run(bell_circuit, device="cpu", shots=100)
            with sf.experiment("inner", tracker=inner):
                sf.run(bell_circuit, device="cpu", shots=50)
            sf.run(bell_circuit, device="cpu", shots=75)

        assert len(outer.starts) == 2
        assert len(outer.completions) == 2
        assert len(inner.starts) == 1
        assert len(inner.completions) == 1
        assert [s["shots"] for s in outer.starts] == [100, 75]
        assert inner.starts[0]["shots"] == 50

    def test_local_tracker_writes_json_files(self, tmp_tracker_dir, bell_circuit):
        tracker = LocalTracker("disk-test", base_dir=str(tmp_tracker_dir))

        with sf.experiment("disk-test", tracker=tracker):
            sf.run(bell_circuit, device="cpu", shots=300)

        run_dir = tmp_tracker_dir / "disk-test"
        json_files = sorted(run_dir.glob("run_*.json"))
        assert len(json_files) == 1

        record = json.loads(json_files[0].read_text())
        assert record["device"] == "cpu"
        assert record["shots"] == 300
        assert record["n_qubits"] == 2
        assert "completed_at" in record

    def test_local_tracker_persists_error_on_failed_run(
        self, tmp_tracker_dir, bell_circuit,
    ):
        tracker = LocalTracker("fail-disk", base_dir=str(tmp_tracker_dir))

        with sf.experiment("fail-disk", tracker=tracker):
            with pytest.raises(RuntimeError):
                sf.run(bell_circuit, device=FailingExecutor(), shots=10)

        json_files = list((tmp_tracker_dir / "fail-disk").glob("run_*.json"))
        assert len(json_files) == 1
        record = json.loads(json_files[0].read_text())
        assert record["error_type"] == "RuntimeError"

    def test_custom_tracker_receives_all_lifecycle_events(self, bell_circuit):
        class LifecycleTracker:
            def __init__(self):
                self.events = []

            def on_run_start(self, circuit, device, shots, metadata=None):
                self.events.append(("start", device, shots))

            def on_run_complete(self, result, metadata=None):
                self.events.append(("complete", result.shots))

            def on_run_error(self, error, metadata=None):
                self.events.append(("error", type(error).__name__))

        tracker = LifecycleTracker()
        assert isinstance(tracker, TrackerProtocol)

        failing = FailingExecutor()
        with sf.experiment("lifecycle", tracker=tracker):
            sf.run(bell_circuit, device="cpu", shots=100)
            with pytest.raises(RuntimeError):
                sf.run(bell_circuit, device=failing, shots=10)

        assert tracker.events == [
            ("start", "cpu", 100),
            ("complete", 100),
            ("start", repr(failing), 10),
            ("error", "RuntimeError"),
        ]
