"""Unit tests for sf.run() dispatch and lifecycle."""

from unittest.mock import MagicMock, patch

import pytest

import superfermion as sf
from superfermion.devices import DeviceCapabilities, DeviceExecutor
from superfermion.results import RunResult


pytestmark = pytest.mark.unit


class MockExecutor:
    """DeviceExecutor-satisfying stub for runner tests."""

    def __init__(self, result=None, skip_fusion=False):
        self._result = result or RunResult(counts={"0": 1}, shots=1)
        self._skip_fusion = skip_fusion
        self.execute = MagicMock(return_value=self._result)
        self.capabilities = MagicMock(
            return_value=DeviceCapabilities(skip_fusion=skip_fusion),
        )


def _mock_executor(result=None, skip_fusion=False):
    return MockExecutor(result=result, skip_fusion=skip_fusion)


class TestRunDeviceResolution:
    def test_string_device_resolves_via_builtin(self, bell_circuit):
        executor = _mock_executor()
        with patch("superfermion.runner._resolve_device", return_value=executor):
            result = sf.run(bell_circuit, device="cpu", shots=10)
        executor.execute.assert_called_once()
        assert isinstance(result, RunResult)

    def test_device_executor_called_directly(self, bell_circuit):
        executor = _mock_executor(skip_fusion=True)
        result = sf.run(bell_circuit, device=executor, shots=25)
        executor.execute.assert_called_once_with(bell_circuit, shots=25)
        assert isinstance(result, RunResult)


class TestRunParameterValidation:
    def test_unbound_parameters_raise(self, parametric_circuit):
        with pytest.raises(RuntimeError, match="unbound parameter"):
            sf.run(parametric_circuit, device="cpu")


class TestRunTrackerLifecycle:
    def test_explicit_tracker_hooks(self, bell_circuit, mock_tracker):
        executor = _mock_executor()
        with patch("superfermion.runner._resolve_device", return_value=executor):
            sf.run(bell_circuit, device=executor, shots=100, tracker=mock_tracker)

        assert len(mock_tracker.starts) == 1
        assert mock_tracker.starts[0]["shots"] == 100
        assert len(mock_tracker.completions) == 1
        assert mock_tracker.errors == []

    def test_active_experiment_tracker_used(self, bell_circuit, mock_tracker):
        executor = _mock_executor()
        with patch("superfermion.runner._resolve_device", return_value=executor):
            with sf.experiment("run-test", tracker=mock_tracker):
                sf.run(bell_circuit, device=executor, shots=50)

        assert len(mock_tracker.starts) == 1
        assert len(mock_tracker.completions) == 1


class TestRunCompilation:
    def test_target_triggers_compile(self, bell_circuit):
        executor = _mock_executor(skip_fusion=True)
        with patch("superfermion.runner._resolve_device", return_value=executor), \
             patch("superfermion.runner._compile_for_target", return_value=bell_circuit) as mock_compile:
            sf.run(bell_circuit, device=executor, target="ionq_aria")

        mock_compile.assert_called_once_with(bell_circuit, "ionq_aria")
        executor.execute.assert_called_once_with(bell_circuit, shots=1000)


class TestRunParams:
    def test_params_kwarg_binds_circuit(self, parametric_circuit):
        """sf.run(circuit, params={...}) should bind and execute."""
        executor = _mock_executor()
        with patch("superfermion.runner._resolve_device", return_value=executor):
            result = sf.run(parametric_circuit, device=executor, params={"theta": 0.5, "phi": 1.0})
        executor.execute.assert_called_once()
        assert isinstance(result, RunResult)


class TestRunErrorPropagation:
    def test_executor_error_calls_on_run_error_and_reraises(self, bell_circuit, mock_tracker):
        executor = _mock_executor()
        executor.execute.side_effect = RuntimeError("backend crash")
        with patch("superfermion.runner._resolve_device", return_value=executor):
            with pytest.raises(RuntimeError, match="backend crash"):
                sf.run(bell_circuit, device=executor, tracker=mock_tracker)

        assert len(mock_tracker.errors) == 1
        assert len(mock_tracker.completions) == 0
