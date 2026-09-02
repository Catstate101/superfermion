"""Integration tests for cloud provider device adapters (fully mocked)."""

from unittest.mock import MagicMock, patch

import pytest

import superfermion as sf
from superfermion.devices import DeviceExecutor


pytestmark = pytest.mark.integration


class TestIBMDeviceAdapter:
    def test_callable_returns_device_executor(self):
        from superfermion.devices.ibm import IBMDevice, IBMDeviceExecutor

        ibm = IBMDevice(token="fake-token")
        with patch.object(ibm, "_ensure_service") as mock_service:
            mock_service.return_value = MagicMock()
            executor = ibm("ibm_fez")

        assert isinstance(executor, IBMDeviceExecutor)
        assert isinstance(executor, DeviceExecutor)

    def test_missing_token_raises_on_call(self, monkeypatch):
        from superfermion.devices.ibm import IBMDevice

        monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
        ibm = IBMDevice()
        with pytest.raises(ValueError, match="requires a token"):
            ibm("ibm_fez")

    def test_env_token_used_when_no_token_passed(self, monkeypatch):
        from superfermion.devices.ibm import IBMDevice

        monkeypatch.setenv("QISKIT_IBM_TOKEN", "env-token")
        mock_service_cls = MagicMock()
        mock_service_cls.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {"qiskit_ibm_runtime": MagicMock(QiskitRuntimeService=mock_service_cls)},
        ):
            executor = IBMDevice()("ibm_fez")

        mock_service_cls.assert_called_once_with(
            channel="ibm_quantum_platform", token="env-token",
        )
        assert executor._backend_name == "ibm_fez"

    def test_service_created_with_token(self):
        from superfermion.devices.ibm import IBMDevice

        ibm = IBMDevice(token="fake-token")
        mock_service_cls = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_cls.return_value = mock_service_instance

        with patch.dict(
            "sys.modules",
            {"qiskit_ibm_runtime": MagicMock(QiskitRuntimeService=mock_service_cls)},
        ):
            executor = ibm("ibm_fez")

        mock_service_cls.assert_called_once_with(
            channel="ibm_quantum_platform", token="fake-token",
        )
        assert executor._backend_name == "ibm_fez"

    def test_execute_returns_run_result_with_mocked_runtime(self, bell_circuit):
        pytest.importorskip("qiskit_ibm_runtime")
        from superfermion.devices.ibm import IBMDeviceExecutor

        mock_service = MagicMock()
        mock_pub = MagicMock()
        mock_pub.data.meas.get_counts.return_value = {"00": 512, "11": 512}
        mock_result = MagicMock()
        mock_result.__getitem__.return_value = mock_pub

        mock_sampler = MagicMock()
        mock_sampler.run.return_value.result.return_value = mock_result
        mock_service.backend.return_value = MagicMock()

        executor = IBMDeviceExecutor(mock_service, "ibm_fez")

        with patch("superfermion.bridge.to_qiskit", return_value=MagicMock()):
            with patch(
                "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager",
            ) as mock_pm:
                mock_pm.return_value.run.return_value = MagicMock()
                with patch("qiskit_ibm_runtime.SamplerV2", return_value=mock_sampler):
                    result = executor.execute(bell_circuit, shots=1024)

        assert result.counts == {"00": 512, "11": 512}
        assert result.shots == 1024
        assert result.metadata["provider"] == "ibm"

    def test_execute_forwards_shots_to_sampler(self, bell_circuit):
        pytest.importorskip("qiskit_ibm_runtime")
        from superfermion.devices.ibm import IBMDeviceExecutor

        mock_service = MagicMock()
        mock_pub = MagicMock()
        mock_pub.data.meas.get_counts.return_value = {"00": 2000, "11": 2000}
        mock_result = MagicMock()
        mock_result.__getitem__.return_value = mock_pub

        mock_sampler = MagicMock()
        mock_sampler.run.return_value.result.return_value = mock_result
        mock_service.backend.return_value = MagicMock()

        executor = IBMDeviceExecutor(mock_service, "ibm_fez")

        with patch("superfermion.bridge.to_qiskit", return_value=MagicMock()):
            with patch(
                "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager",
            ) as mock_pm:
                mock_pm.return_value.run.return_value = MagicMock()
                with patch("qiskit_ibm_runtime.SamplerV2", return_value=mock_sampler):
                    executor.execute(bell_circuit, shots=4000)

        mock_sampler.run.assert_called_once()
        _, call_kwargs = mock_sampler.run.call_args
        assert call_kwargs.get("shots") == 4000


class TestIonQDeviceAdapter:
    def test_callable_returns_device_executor(self):
        from superfermion.devices.ionq import IonQDevice, IonQDeviceExecutor

        ionq = IonQDevice(api_key="fake-key")
        executor = ionq("ionq.aria-1")

        assert isinstance(executor, IonQDeviceExecutor)
        assert isinstance(executor, DeviceExecutor)

    def test_missing_api_key_raises_on_call(self):
        from superfermion.devices.ionq import IonQDevice

        ionq = IonQDevice(api_key=None)
        with pytest.raises(ValueError, match="requires an api_key"):
            ionq("ionq.aria-1")

    def test_missing_api_key_default_constructor(self):
        from superfermion.devices.ionq import IonQDevice

        ionq = IonQDevice()
        with pytest.raises(ValueError, match="requires an api_key"):
            ionq("ionq.aria-1")


class TestBraketDeviceAdapter:
    def test_requires_boto3_on_construction(self):
        from superfermion.devices.braket import BraketDevice

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no boto3")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="boto3"):
                BraketDevice(s3_bucket="test-bucket")

    def test_callable_returns_device_executor_with_mocked_boto3(self):
        pytest.importorskip("boto3")
        from superfermion.devices.braket import BraketDevice, BraketDeviceExecutor

        with patch("boto3.Session"):
            braket = BraketDevice(s3_bucket="test-bucket")

        with patch.object(
            braket,
            "_resolve_device_arn",
            return_value="arn:aws:braket:us-east-1::device/quantum-simulator/amazon/sv1",
        ):
            executor = braket("sv1")

        assert isinstance(executor, BraketDeviceExecutor)
        assert isinstance(executor, DeviceExecutor)

    def test_unknown_device_raises_value_error(self):
        pytest.importorskip("boto3")
        from superfermion.devices.braket import BraketDevice

        with patch("boto3.Session"):
            braket = BraketDevice(s3_bucket="test-bucket")

        with patch.object(braket, "_get_aws_session") as mock_session:
            mock_session.return_value.search_devices.return_value = []
            with pytest.raises(ValueError, match="No Braket device found"):
                braket("nonexistent_device_xyz")


class TestAdapterProtocolCompliance:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: __import__(
                "superfermion.devices.ibm", fromlist=["IBMDevice"],
            ).IBMDevice(token="fake"),
            lambda: __import__(
                "superfermion.devices.ionq", fromlist=["IonQDevice"],
            ).IonQDevice(api_key="fake"),
        ],
    )
    def test_factory_executors_satisfy_device_executor(self, factory):
        adapter = factory()
        if hasattr(adapter, "_ensure_service"):
            with patch.object(adapter, "_ensure_service", return_value=MagicMock()):
                executor = adapter("test-backend")
        else:
            executor = adapter("test-backend")

        assert isinstance(executor, DeviceExecutor)
        caps = executor.capabilities()
        assert caps.max_qubits > 0

    def test_sf_run_accepts_mocked_local_device_executor(self, bell_circuit):
        mock_executor = MagicMock(spec=DeviceExecutor)
        mock_executor.capabilities.return_value = sf.DeviceCapabilities(skip_fusion=True)
        mock_executor.execute.return_value = sf.RunResult(
            counts={"00": 500, "11": 500},
            shots=1000,
        )

        result = sf.run(bell_circuit, device=mock_executor, shots=1000)

        mock_executor.execute.assert_called_once()
        assert result.counts == {"00": 500, "11": 500}
        assert result.shots == 1000
