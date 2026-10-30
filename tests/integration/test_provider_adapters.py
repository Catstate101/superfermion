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

    def test_ensure_measurements_appends_when_missing(self):
        """SamplerV2 rejects measure-less circuits (error 1515); the adapter
        must append measure_all() before transpilation."""
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        from superfermion.devices.ibm import _ensure_measurements

        qc = QuantumCircuit(2)
        qc.h(0)
        out = _ensure_measurements(qc)
        names = [i.operation.name for i in out.data]
        assert "measure" in names
        assert out.num_clbits == 2

        # circuits that already carry measurements are left untouched
        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.measure([0, 1], [0, 1])
        out2 = _ensure_measurements(qc2)
        n_meas = sum(1 for i in out2.data if i.operation.name == "measure")
        assert n_meas == 2  # no extra measure_all appended

    def test_ensure_measurements_skips_unintrospectable_circuit(self):
        """A mocked/non-qiskit circuit must not break the pipeline."""
        from superfermion.devices.ibm import _ensure_measurements
        qc = MagicMock()
        assert _ensure_measurements(qc) is qc

    def test_sampler_keys_already_match_sf_order(self):
        """SamplerV2 keys need no reversal before reaching SF.

        ``to_qiskit`` keeps qubit labels unchanged (SF and Qiskit share
        little-endian q0 = LSB) and ``measure_all`` maps creg[q] <- qubit q,
        so a key already carries SF qubit q at position n-1-q -- SF's
        documented q0-last order (results.py). A reversal here compensated for
        the bridge's n-1-q reversal, which 59f29bc removed; left in place it
        mirrored every multi-qubit hardware result.
        """
        pytest.importorskip("qiskit")
        from superfermion.bridge import to_qiskit
        from superfermion.devices.ibm import _ensure_measurements
        from superfermion.results import RunResult

        circ = sf.Circuit(3).x(0)  # SF qubit 0 flipped
        qc = _ensure_measurements(to_qiskit(circ))

        assert qc.data[0].operation.name == "x"
        # to_qiskit keeps labels: SF qubit 0 stays Qiskit qubit 0
        assert qc.find_bit(qc.data[0].qubits[0]).index == 0

        # SF circuits carry n_cbits == n_qubits, so to_qiskit builds a 'c'
        # register and measure_all adds its own 'meas' one; execute() reads
        # data.meas, whose keys are indexed by the register, not the circuit.
        meas_reg = next(r for r in qc.cregs if r.name == "meas")
        meas = [i for i in qc.data if i.operation.name == "measure"]
        assert len(meas) == 3
        for inst in meas:
            qubit = qc.find_bit(inst.qubits[0]).index
            assert inst.clbits[0] in meas_reg
            assert inst.clbits[0] == meas_reg[qubit]  # creg[q] <- qubit q

        # So q0=1 arrives as key '001' (leftmost char is q2). SF's q0-last
        # parser reads the rightmost char as q0: the two already agree, and
        # reversing '001' -> '100' would flip the sign of <Z0>.
        res = RunResult(counts={"001": 800, "000": 200}, shots=1000)
        assert res.expectation([([3, 0], 1.0, 0.0)]) == pytest.approx(-0.6)

    def test_execute_preserves_sampler_counts_order(self, bell_circuit):
        """Raw SamplerV2 counts must reach RunResult unchanged.

        A reversal here mirrors every multi-qubit hardware result; the
        q0-last parser is expected to agree with the key as delivered.
        """
        pytest.importorskip("qiskit_ibm_runtime")
        from superfermion.devices.ibm import IBMDeviceExecutor

        mock_service = MagicMock()
        mock_pub = MagicMock()
        # real SamplerV2 counts for x(0): q0 is the rightmost character
        mock_pub.data.meas.get_counts.return_value = {"01": 700, "00": 300}
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
                    result = executor.execute(bell_circuit, shots=1000)

        assert result.counts == {"01": 700, "00": 300}
        # q0-last parser: <Z0> = P('00') - P('01') = 0.3 - 0.7 = -0.4
        assert result.expectation([([3, 0], 1.0, 0.0)]) == pytest.approx(-0.4)
        assert "job_id" in result.metadata

    def test_list_devices_returns_sf_device_info(self):
        """list_devices() maps account backends to SF-style DeviceInfo."""
        from types import SimpleNamespace

        from superfermion.devices import DeviceInfo
        from superfermion.devices.ibm import IBMDevice

        def fake_backend(name, n, status=None, sim=False, operational=None):
            if status is not None:  # older runtime: status().name
                st = SimpleNamespace(name=status)
            else:  # newer runtime: status().operational only
                st = SimpleNamespace(operational=operational, status_msg="")
            return SimpleNamespace(
                name=name, num_qubits=n, simulator=sim, status=lambda: st,
            )

        mock_service = MagicMock()
        mock_service.backends.return_value = [
            fake_backend("ibm_fez", 156, status="operational"),
            fake_backend("ibm_marrakesh", 156, operational=True),
            fake_backend("ibm_kingston", 156, operational=False),
        ]

        ibm = IBMDevice(token="fake-token")
        with patch.object(ibm, "_ensure_service", return_value=mock_service):
            infos = ibm.list_devices()

        assert all(isinstance(d, DeviceInfo) for d in infos)
        assert [d.name for d in infos] == [
            "ibm_fez", "ibm_marrakesh", "ibm_kingston"]
        assert infos[0].n_qubits == 156
        assert infos[0].status == "operational"  # older .name shape
        assert infos[1].status == "operational"  # newer .operational shape
        assert infos[2].status == "offline"      # newer offline mapping
        assert infos[0].is_simulator is False

    def test_list_devices_requires_token(self, monkeypatch):
        from superfermion.devices.ibm import IBMDevice

        monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
        with pytest.raises(ValueError, match="requires a token"):
            IBMDevice().list_devices()


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

    def test_histogram_is_unwrapped_and_reordered(self):
        """The adapter used to store the whole /results document as counts."""
        from superfermion.devices.ionq import _extract_ionq_counts

        payload = {
            "id": "abc",
            "status": "completed",
            "shots": 40,
            "histogram": {"0": 10, "3": 30},
        }
        assert _extract_ionq_counts(payload, shots=40, n_qubits=2) == {
            "00": 10, "11": 30,
        }
        # 2 = '10' with q0 leftmost -> q0=1, q1=0 -> SF '01'
        assert _extract_ionq_counts({"histogram": {"2": 5}}, 5, 2) == {"01": 5}

    def test_probabilities_are_scaled_to_shots(self):
        from superfermion.devices.ionq import _extract_ionq_counts

        counts = _extract_ionq_counts(
            {"probabilities": {"0": 0.25, "1": 0.75}}, shots=400, n_qubits=1,
        )
        assert counts == {"0": 100, "1": 300}

    def test_unparseable_payload_raises(self):
        from superfermion.devices.ionq import _extract_ionq_counts

        with pytest.raises(RuntimeError, match="Could not parse IonQ result"):
            _extract_ionq_counts({"id": "abc", "status": "completed"}, 10, 2)


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

    def test_list_devices_returns_sf_device_info(self):
        """list_devices() maps Braket devices to SF-style DeviceInfo; a
        device whose properties cannot be fetched falls back to -1."""
        import types as pytypes

        from superfermion.devices import DeviceInfo
        from superfermion.devices.braket import BraketDevice

        def fake_device(name, qubit_count, status, dtype):
            props = pytypes.SimpleNamespace(
                paradigm=pytypes.SimpleNamespace(qubitCount=qubit_count))
            return pytypes.SimpleNamespace(
                name=name, status=status, type=dtype, properties=props)

        fake_sv1 = fake_device("SV1", 34, "ONLINE", "SIMULATOR")
        fake_qpu = fake_device("Rigetti", 80, "ONLINE", "QPU")
        fake_qpu.properties = None  # properties unavailable -> -1 fallback

        class FakeAwsDevice:
            @classmethod
            def get_devices(cls, aws_session=None, **kwargs):
                return [fake_sv1, fake_qpu]

        fake_aws = pytypes.ModuleType("braket.aws")
        fake_aws.AwsDevice = FakeAwsDevice

        braket = BraketDevice.__new__(BraketDevice)  # no boto3 needed
        braket._s3_bucket = "test-bucket"

        with patch.dict("sys.modules", {
            "braket": pytypes.ModuleType("braket"),
            "braket.aws": fake_aws,
        }):
            with patch.object(braket, "_get_aws_session",
                              return_value=MagicMock()):
                infos = braket.list_devices()

        assert all(isinstance(d, DeviceInfo) for d in infos)
        assert [d.name for d in infos] == ["SV1", "Rigetti"]
        assert infos[0].n_qubits == 34
        assert infos[0].is_simulator is True
        assert infos[1].n_qubits == -1  # graceful fallback
        assert infos[1].status == "ONLINE"

    def test_counts_are_flipped_into_sf_order(self):
        """Braket puts q0 leftmost; SF puts q0 last."""
        from superfermion.devices.braket import _braket_counts_to_sf

        assert _braket_counts_to_sf({"001": 7}) == {"100": 7}
        assert _braket_counts_to_sf({"1": 7}) == {"1": 7}

    def test_untouched_qubits_are_scattered_back(self):
        """Braket drops untouched qubits from the key; they must be restored."""
        from superfermion.devices.braket import _braket_counts_to_sf

        # 3-qubit run that only touched q0 and q2, key '10' = q0:1, q2:0
        counts = _braket_counts_to_sf({"10": 4}, width=3, qubits=[0, 2])
        assert counts == {"001": 4}
        # full-width case still behaves like a plain reversal
        assert _braket_counts_to_sf({"001": 2}, width=3, qubits=[0, 1, 2]) == {"100": 2}

    def test_reset_is_refused_not_dropped(self):
        pytest.importorskip("braket")
        from superfermion.devices.braket import _to_braket

        circuit = sf.Circuit(1)
        circuit.x(0).reset(0)
        with pytest.raises(ValueError, match="RESET"):
            _to_braket(circuit)

    def test_recently_added_gate_names_map(self):
        """SDG/TDG/SX/P/CP/RXX/RYY/RZZ/U used to raise 'Cannot map gate'."""
        pytest.importorskip("braket")
        from superfermion.devices.braket import _to_braket

        circuit = sf.Circuit(2)
        circuit.sdg(0).tdg(1).sx(0).p(0.4, 0)
        circuit.cp(0.7, 0, 1).rxx(0.3, 0, 1).ryy(0.3, 0, 1).rzz(0.3, 0, 1)
        circuit.u3(0.3, 0.2, 0.1, 1)

        braket_circuit = _to_braket(circuit)
        assert len(braket_circuit.instructions) >= 9


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


class TestIBMMeasurementCompletion:
    def test_partial_measurement_is_completed(self):
        """Measuring one qubit used to leave the others out of the counts."""
        pytest.importorskip("qiskit")
        from superfermion.bridge import to_qiskit
        from superfermion.devices.ibm import _ensure_measurements

        circuit = sf.Circuit(3)
        circuit.measure(0)
        out = _ensure_measurements(to_qiskit(circuit))

        measured = {
            out.find_bit(inst.qubits[0]).index
            for inst in out.data
            if inst.operation.name == "measure"
        }
        assert measured == {0, 1, 2}
        assert any(reg.name == "meas" for reg in out.cregs)

    def test_circuit_without_measurement_gets_measure_all(self):
        pytest.importorskip("qiskit")
        from superfermion.bridge import to_qiskit
        from superfermion.devices.ibm import _ensure_measurements

        out = _ensure_measurements(to_qiskit(sf.Circuit(2)))
        measured = {
            out.find_bit(inst.qubits[0]).index
            for inst in out.data
            if inst.operation.name == "measure"
        }
        assert measured == {0, 1}

    def test_capabilities_reports_backend_width(self):
        import types

        from superfermion.devices.ibm import IBMDeviceExecutor

        service = MagicMock()
        service.backend.return_value = types.SimpleNamespace(num_qubits=156)
        caps = IBMDeviceExecutor(service, "ibm_x").capabilities()
        assert caps.max_qubits == 156
        assert caps.is_simulator is False

    def test_capabilities_falls_back_offline(self):
        from superfermion.devices.ibm import IBMDeviceExecutor

        service = MagicMock()
        service.backend.side_effect = RuntimeError("no network")
        assert IBMDeviceExecutor(service, "ibm_x").capabilities().max_qubits == 127


class _StubDevice:
    def __init__(self, max_qubits, is_simulator):
        self._max_qubits = max_qubits
        self._is_simulator = is_simulator
        self.calls = 0

    def capabilities(self):
        return sf.DeviceCapabilities(
            max_qubits=self._max_qubits, is_simulator=self._is_simulator)

    def execute(self, circuit, shots=8, **kwargs):
        self.calls += 1
        return sf.RunResult(counts={"00000": shots}, shots=shots)


class TestRunnerDeviceWidthGuard:
    def test_hardware_width_is_enforced(self):
        """A 5-qubit circuit reached a 2-qubit device and returned counts."""
        device = _StubDevice(max_qubits=2, is_simulator=False)
        with pytest.raises(ValueError, match="supports at most 2"):
            sf.run(sf.Circuit(5), device=device, shots=8)
        assert device.calls == 0

    def test_simulator_is_not_gated(self):
        device = _StubDevice(max_qubits=2, is_simulator=True)
        sf.run(sf.Circuit(5), device=device, shots=8)
        assert device.calls == 1

    def test_unknown_width_is_not_gated(self):
        device = _StubDevice(max_qubits=0, is_simulator=False)
        sf.run(sf.Circuit(5), device=device, shots=8)
        assert device.calls == 1


class TestIBMTimeout:
    """The runtime wait must honour ``timeout=`` without losing the job id."""

    @staticmethod
    def _executor():
        from superfermion.devices.ibm import IBMDeviceExecutor

        service = MagicMock()
        service.backend.return_value = MagicMock()
        return IBMDeviceExecutor(service, "ibm_fez")

    @staticmethod
    def _job(result=None, side_effect=None, job_id="job-123"):
        job = MagicMock()
        job.job_id.return_value = job_id
        if side_effect is not None:
            job.result.side_effect = side_effect
        else:
            job.result.return_value = result
        return job

    @staticmethod
    def _pub(counts=None):
        pub = MagicMock()
        pub.data.meas.get_counts.return_value = counts or {"0": 8}
        remote = MagicMock()
        remote.__getitem__.return_value = pub
        return remote

    def _execute(self, job, **kwargs):
        sampler = MagicMock()
        sampler.run.return_value = job
        with patch("superfermion.bridge.to_qiskit", return_value=MagicMock()):
            with patch(
                "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager",
            ) as mock_pm:
                mock_pm.return_value.run.return_value = MagicMock()
                with patch("qiskit_ibm_runtime.SamplerV2", return_value=sampler):
                    return self._executor().execute(sf.Circuit(1), shots=8, **kwargs)

    def test_timeout_is_forwarded_to_result(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_ibm_runtime")

        job = self._job(result=self._pub({"0": 8}))
        self._execute(job, timeout=0.01)
        job.result.assert_called_once_with(timeout=0.01)

    def test_expired_timeout_keeps_the_job_id(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_ibm_runtime")

        from qiskit_ibm_runtime.exceptions import RuntimeJobMaxTimeoutError

        job = self._job(side_effect=RuntimeJobMaxTimeoutError("waited too long"))
        with pytest.raises(RuntimeJobMaxTimeoutError, match="job-123"):
            self._execute(job, timeout=0.01)

    def test_runtime_without_timeout_kwarg_retries_plain(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_ibm_runtime")

        remote = self._pub({"0": 8})
        job = self._job(side_effect=[TypeError("unexpected keyword"), remote])
        result = self._execute(job, timeout=0.01)
        assert result.counts == {"0": 8}
        assert job.result.call_count == 2
        assert job.result.call_args_list[1].kwargs == {}
