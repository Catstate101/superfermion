"""Unit tests for DeviceExecutor protocol and device resolution."""

from unittest.mock import MagicMock, patch

import pytest

from superfermion.devices import DeviceCapabilities, DeviceExecutor, _resolve_builtin
from superfermion.results import RunResult


pytestmark = pytest.mark.unit


class CustomExecutor:
    """Minimal DeviceExecutor implementation for protocol checks."""

    def execute(self, circuit, shots=1000, **kwargs):
        return RunResult(counts={"0": shots}, shots=shots)

    def capabilities(self):
        return DeviceCapabilities(max_qubits=8)


class TestDeviceExecutorProtocol:
    def test_custom_class_satisfies_protocol(self):
        executor = CustomExecutor()
        assert isinstance(executor, DeviceExecutor)


class TestDeviceCapabilities:
    def test_defaults(self):
        caps = DeviceCapabilities()
        assert caps.max_qubits == 32
        assert caps.native_gates == ["all"]
        assert caps.coupling_map is None
        assert caps.skip_fusion is False
        assert caps.supports_statevector is True
        assert caps.is_simulator is True

    def test_all_fields_set(self):
        caps = DeviceCapabilities(
            max_qubits=127,
            native_gates=["cx", "rz"],
            coupling_map=[(0, 1), (1, 2)],
            skip_fusion=True,
            supports_statevector=False,
            is_simulator=False,
        )
        assert caps.max_qubits == 127
        assert caps.native_gates == ["cx", "rz"]
        assert caps.coupling_map == [(0, 1), (1, 2)]
        assert caps.skip_fusion is True
        assert caps.supports_statevector is False
        assert caps.is_simulator is False


class TestRustDevice:
    def test_rust_device_statevector(self, bell_circuit):
        from superfermion.devices.rust_device import RustDevice
        device = RustDevice(hardware="cpu", method="statevector")
        assert isinstance(device, DeviceExecutor)
        result = device.execute(bell_circuit, shots=100)
        assert isinstance(result, RunResult)
        assert result.shots == 100
        assert sum(result.counts.values()) == 100
        assert result.state is not None

    def test_rust_device_methods(self, bell_circuit):
        from superfermion.devices.rust_device import RustDevice
        for method in ("statevector", "density_matrix", "mps", "stabilizer"):
            device = RustDevice(hardware="cpu", method=method)
            result = device.execute(bell_circuit, shots=100)
            assert result.state is not None


class TestResolveBuiltin:
    def test_cpu_returns_rust_device(self):
        from superfermion.devices.rust_device import RustDevice
        device = _resolve_builtin("cpu")
        assert isinstance(device, RustDevice)
        assert device._hardware == "cpu"

    def test_cpu_satisfies_protocol(self):
        device = _resolve_builtin("cpu")
        assert isinstance(device, DeviceExecutor)

    def test_gpu_raises_when_unavailable(self):
        from superfermion.devices.rust_device import RustDevice
        try:
            device = _resolve_builtin("gpu")
            assert isinstance(device, RustDevice)
            assert device._hardware == "gpu"
        except RuntimeError:
            pass  # GPU not available is expected in CI/test environments

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown device"):
            _resolve_builtin("totally_unknown_backend_xyz")


class TestIBMDevice:
    def test_callable_returns_executor(self):
        from superfermion.devices.ibm import IBMDevice, IBMDeviceExecutor

        ibm = IBMDevice(token="fake-token")
        with patch.object(ibm, "_ensure_service") as mock_service:
            mock_service.return_value = MagicMock()
            executor = ibm("ibm_fez")
        assert isinstance(executor, IBMDeviceExecutor)

    def test_missing_token_raises(self):
        from superfermion.devices.ibm import IBMDevice

        ibm = IBMDevice()
        with pytest.raises(ValueError, match="requires a token"):
            ibm("ibm_fez")


class TestIonQDevice:
    def test_callable_returns_executor(self):
        from superfermion.devices.ionq import IonQDevice, IonQDeviceExecutor

        ionq = IonQDevice(api_key="fake-key")
        executor = ionq("ionq.aria-1")
        assert isinstance(executor, IonQDeviceExecutor)

    def test_missing_api_key_raises(self):
        from superfermion.devices.ionq import IonQDevice

        ionq = IonQDevice()
        with pytest.raises(ValueError, match="requires an api_key"):
            ionq("ionq.aria-1")


class TestBraketDevice:
    def test_callable_returns_executor(self):
        from superfermion.devices.braket import BraketDevice, BraketDeviceExecutor

        with patch("boto3.Session"):
            braket = BraketDevice(s3_bucket="test-bucket")
        with patch.object(braket, "_resolve_device_arn", return_value="arn:aws:braket:us-east-1::device/qpu/ionq"):
            executor = braket("sv1")
        assert isinstance(executor, BraketDeviceExecutor)

    def test_resolve_device_arn_unknown_raises(self):
        from superfermion.devices.braket import BraketDevice

        with patch("boto3.Session"):
            braket = BraketDevice(s3_bucket="test-bucket")
        with patch.object(braket, "_get_aws_session") as mock_session:
            mock_session.return_value.search_devices.return_value = []
            with pytest.raises(ValueError, match="No Braket device found"):
                braket("nonexistent_qpu_xyz")
