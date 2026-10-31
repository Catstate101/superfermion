"""
BraketDevice — DeviceExecutor adapter for Amazon Braket.

Absorbs logic from the old ``runtime/providers/aws.py`` and exposes it
through the ``DeviceExecutor`` protocol.

Requires: ``pip install amazon-braket-sdk boto3``
"""
from __future__ import annotations


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from superfermion.devices import DeviceCapabilities, DeviceInfo


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
        # Additive aliases for names ``sf.Circuit`` actually stores but the
        # original map missed (each was a hard "Cannot map gate" failure).
        "SDG": "si", "TDG": "ti", "SX": "v",
        "P": "phaseshift", "CP": "cphaseshift",
        "RXX": "xx", "RYY": "yy", "RZZ": "zz",
    }

    bc = BraketCircuit()
    circuit._ensure_gates()  # no-op for default Python storage
    for gate in circuit._gates:
        name = gate.name.upper()

        # SF stores RESET, Braket's Circuit has no reset method in this SDK.
        # The original code skipped it silently, so a circuit that relied on a
        # reset ran with the reset missing.  Refuse rather than corrupt.
        if name == "RESET":
            raise ValueError(
                "Cannot map gate 'RESET' to Braket: this SDK has no reset "
                "instruction, and dropping it would change the circuit. "
                "Remove the reset or decompose the circuit before submitting."
            )

        if name == "U" and len(gate.params) == 3:
            theta, phi, lam = (float(p) for p in gate.params)
            _apply_u3(bc, int(list(gate.qubits)[0]), theta, phi, lam)
            continue

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


def _braket_counts_to_sf(
    counts: Dict[str, int],
    width: int = 0,
    qubits: Optional[List[int]] = None,
) -> Dict[str, int]:
    """Rewrite Braket bitstrings into SF order.

    Braket orders every bitstring with the first measured qubit in the
    leftmost character (``LocalSimulator`` and ``AwsDevice`` alike), while SF
    documents q0-last (see ``results.py``: qubit q is bit n-1-q).
    ``_to_braket`` maps qubit labels through unchanged, so without this flip a
    multi-qubit hardware result is mirrored: on a 3-qubit circuit where q0
    should read 0.02 and q2 0.58, the delivered counts read 0.58 on q0
    (verified against Braket's LocalSimulator on the same circuit).

    Braket also drops untouched qubits from the key entirely: a 3-qubit SF
    circuit that only touches q0 and q2 returns 2-character keys over
    ``(q0, q2)`` in ascending order.  Pass ``width=circuit.n_qubits`` and
    ``qubits=sorted(braket_circuit.qubits)`` to scatter the bits back onto
    their real positions; with no ``qubits`` the key is treated as a plain
    reversal over all ``width`` positions.
    """
    out: Dict[str, int] = {}
    for key, value in (counts or {}).items():
        key = str(key)
        if qubits and width > 0:
            placed = ["0"] * width
            for i, q in enumerate(qubits):
                if i < len(key) and 0 <= width - 1 - int(q) < width:
                    placed[width - 1 - int(q)] = key[i]
            flipped = "".join(placed)
        else:
            if width > len(key):
                key = key.rjust(width, "0")
            flipped = key[::-1] if len(key) > 1 else key
        out[flipped] = out.get(flipped, 0) + value
    return out


def _apply_u3(braket_circuit: Any, qubit: int, theta: float, phi: float, lam: float) -> None:
    """Emit an SF ``U(theta, phi, lam)`` as rz/ry/rz.

    ``U3 = e^{-i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam)``, so the three rotations
    reproduce the gate up to a global phase, which measurement cannot see.
    Verified against Braket's ``to_unitary`` to 1e-16.  Braket has no ``u3``
    method, so this replaces raising "Cannot map gate 'U'".
    """
    braket_circuit.rz(qubit, lam)
    braket_circuit.ry(qubit, theta)
    braket_circuit.rz(qubit, phi)


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
        counts = _braket_counts_to_sf(
            counts,
            getattr(circuit, "n_qubits", 0),
            sorted(int(q) for q in bc.qubits),
        )

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

    def list_devices(self) -> List[DeviceInfo]:
        """List Amazon Braket devices available to this account.

        Returns SF-style ``DeviceInfo`` summaries (name, n_qubits, status,
        is_simulator). Requires valid AWS credentials in the session;
        ``n_qubits`` falls back to -1 when device properties cannot be
        fetched (e.g. offline listing stubs).
        """
        from braket.aws import AwsDevice

        aws_session = self._get_aws_session()
        infos: List[DeviceInfo] = []
        for device in AwsDevice.get_devices(aws_session=aws_session):
            try:
                n_qubits = int(device.properties.paradigm.qubitCount)
            except Exception:
                n_qubits = -1
            infos.append(DeviceInfo(
                name=str(device.name),
                n_qubits=n_qubits,
                status=str(getattr(device, "status", "UNKNOWN")),
                is_simulator=str(getattr(device, "type", "")) == "SIMULATOR",
            ))
        return infos
