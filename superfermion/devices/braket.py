"""
BraketDevice — DeviceExecutor adapter for Amazon Braket.

Absorbs logic from the old ``runtime/providers/aws.py`` and exposes it
through the ``DeviceExecutor`` protocol.

Requires: ``pip install amazon-braket-sdk boto3``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from superfermion.devices import DeviceCapabilities


def _to_braket(circuit: "Circuit") -> Any:
    """Convert a Superfermion Circuit to a Braket Circuit."""
    from braket.circuits import Circuit as BraketCircuit

    GATE_MAP: Dict[str, str] = {
        "H": "h", "X": "x", "Y": "y", "Z": "z",
        "S": "s", "SI": "si", "T": "t", "TI": "ti",
        "V": "v", "VI": "vi",
        "RX": "rx", "RY": "ry", "RZ": "rz",
        "CX": "cnot", "CNOT": "cnot",
        "CZ": "cz", "CY": "cy",
        "SWAP": "swap", "ISWAP": "iswap",
        "CCX": "ccnot", "CSWAP": "cswap",
        "ECR": "ecr",
        "XX": "xx", "YY": "yy", "ZZ": "zz",
    }

    bc = BraketCircuit()
    for gate in circuit._gates:
        name = gate.name.upper()
        braket_name = GATE_MAP.get(name)
        if braket_name is None:
            if name in ("MEASURE", "BARRIER", "ID", "RESET"):
                continue
            raise ValueError(f"Cannot map gate '{gate.name}' to Braket")
        method = getattr(bc, braket_name)
        qubits = list(gate.qubits)
        if gate.params:
            method(*qubits, *gate.params)
        else:
            method(*qubits)
    return bc


def _extract_braket_counts(result: Any) -> Dict[str, int]:
    """Extract measurement counts from a Braket result object."""
    try:
        if hasattr(result, "measurement_counts"):
            mc = result.measurement_counts
            if mc is not None:
                return dict(mc) if not isinstance(mc, dict) else mc
    except Exception:
        pass

    try:
        if hasattr(result, "measurements"):
            measurements = result.measurements
            if measurements is not None:
                counts: Dict[str, int] = {}
                for m in measurements:
                    key = "".join(str(bit) for bit in m)
                    counts[key] = counts.get(key, 0) + 1
                return counts
    except Exception:
        pass

    return {}


class BraketDeviceExecutor:
    """Executor bound to a specific Braket device ARN."""

    def __init__(self, aws_session: Any, device_arn: str, s3_bucket: str) -> None:
        self._aws_session = aws_session
        self._device_arn = device_arn
        self._s3_bucket = s3_bucket

    def execute(self, circuit: "Circuit", shots: int = 1000, **kwargs: Any) -> "RunResult":
        from braket.aws import AwsDevice
        from superfermion.results import RunResult

        aws_device = AwsDevice(arn=self._device_arn, aws_session=self._aws_session)
        bc = _to_braket(circuit)

        s3_path = (self._s3_bucket, kwargs.pop("s3_folder", "superfermion-jobs"))
        task = aws_device.run(bc, s3_path, shots=shots, poll_timeout_seconds=5)

        timeout = kwargs.pop("timeout", 600)
        start = time.time()
        while True:
            state = task.state()
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            if time.time() - start > timeout:
                raise TimeoutError(f"Braket job timed out after {timeout}s")
            time.sleep(5)

        if state == "FAILED":
            raise RuntimeError(f"Braket job {task.id} FAILED")
        if state == "CANCELLED":
            raise RuntimeError(f"Braket job {task.id} CANCELLED")

        result = task.result()
        counts = _extract_braket_counts(result)

        return RunResult(
            counts=counts,
            shots=shots,
            metadata={"backend": self._device_arn, "provider": "braket", "job_id": task.id},
        )

    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            max_qubits=80,
            skip_fusion=False,
            supports_statevector=False,
            is_simulator=False,
        )


class BraketDevice:
    """Callable factory returning a ``BraketDeviceExecutor`` for a device.

    Usage::

        braket = BraketDevice(region="us-east-1", s3_bucket="my-bucket")
        result = sf.run(circuit, device=braket("rigetti"))
    """

    def __init__(
        self,
        region: str = "us-east-1",
        s3_bucket: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        import boto3
        import os

        self._region = region
        self._s3_bucket = s3_bucket or os.getenv("BRAKET_S3_BUCKET", "")
        if not self._s3_bucket:
            self._s3_bucket = f"amazon-braket-{region}-{int(time.time())}"

        session_kwargs: dict = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self._boto_session = boto3.Session(**session_kwargs)

    def _get_aws_session(self) -> Any:
        from braket.aws import AwsSession
        return AwsSession(boto_session=self._boto_session)

    def _resolve_device_arn(self, device: str) -> str:
        if device.startswith("arn:aws:braket:"):
            return device
        QPU_MAP = {
            "rigetti": "Rigetti", "ionq": "IonQ", "oqc": "OQC",
            "sv1": "SV1", "dm1": "DM1", "tn1": "TN1",
        }
        provider = QPU_MAP.get(device.lower().split("_")[0])
        aws_session = self._get_aws_session()
        devices = aws_session.search_devices(
            filters=[{"name": "providerName", "values": [provider]}]
        ) if provider else []

        for d in devices:
            if device.lower() in d.get("deviceName", "").lower():
                return d["deviceArn"]

        if devices:
            return devices[0]["deviceArn"]

        raise ValueError(f"No Braket device found for '{device}'")

    def __call__(self, device: str = "sv1") -> BraketDeviceExecutor:
        device_arn = self._resolve_device_arn(device)
        return BraketDeviceExecutor(self._get_aws_session(), device_arn, self._s3_bucket)
