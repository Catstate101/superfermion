"""
AWS Braket Provider — Access Rigetti, IonQ, OQC, and simulators via Amazon Braket.

Requires:
    pip install amazon-braket-sdk boto3

AWS credentials are read from:
    - ~/.aws/credentials  (standard boto3 credential chain)
    - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    - .env file in project root

Usage:
    provider = BraketProvider(region="us-east-1", s3_bucket="my-braket-bucket")
    job = provider.run(circuit, device="rigetti", shots=1000)
    result = job.result(timeout=300)
    print(result.counts)
"""

from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import superfermion as sf
from superfermion.runtime import Job, JobStatus
from superfermion.results import RunResult


# ---------------------------------------------------------------------------
# Circuit bridge: SF -> Braket
# ---------------------------------------------------------------------------

def to_braket(circuit: sf.Circuit):
    """Convert a Superfermion Circuit to an Amazon Braket Circuit.

    Args:
        circuit: Superfermion Circuit.

    Returns:
        braket.circuits.Circuit
    """
    from braket.circuits import Circuit as BraketCircuit

    bc = BraketCircuit()

    GATE_MAP: dict[str, str] = {
        'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z',
        'S': 's', 'SI': 'si', 'T': 't', 'TI': 'ti',
        'V': 'v', 'VI': 'vi',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CX': 'cnot', 'CNOT': 'cnot',
        'CZ': 'cz', 'CY': 'cy',
        'SWAP': 'swap', 'ISWAP': 'iswap',
        'CCX': 'ccnot', 'CSWAP': 'cswap',
        'ECR': 'ecr',
        'XX': 'xx', 'YY': 'yy', 'ZZ': 'zz',
    }

    for gate in circuit._gates:
        name = gate.name.upper()
        braket_name = GATE_MAP.get(name)

        if braket_name is None:
            if name in ('MEASURE', 'BARRIER', 'ID', 'RESET'):
                continue
            raise ValueError(f"Cannot map gate '{gate.name}' to Braket")

        method = getattr(bc, braket_name)
        qubits = list(gate.qubits)

        if gate.params:
            method(*qubits, *gate.params)
        else:
            method(*qubits)

    return bc


# ---------------------------------------------------------------------------
# BraketJob
# ---------------------------------------------------------------------------

class BraketJob(Job):
    """Asynchronous job on AWS Braket (real QPU or simulator)."""

    def __init__(self, arn: str, device_name: str, aws_session=None):
        super().__init__()
        self.arn = arn
        self.device_name = device_name
        self._aws_session = aws_session

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Wait for job completion and return counts."""
        from braket.aws import AwsQuantumTask

        sf.utils.info(f"Waiting for Braket job {self.arn} on {self.device_name}...")

        task = AwsQuantumTask(arn=self.arn, aws_session=self._aws_session)
        if timeout is not None:
            start = time.time()
            while True:
                state = task.state()
                if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    break
                if time.time() - start > timeout:
                    raise TimeoutError(f"Braket job {self.arn} timed out after {timeout}s")
                time.sleep(5)
        else:
            # Block until complete
            task = AwsQuantumTask(arn=self.arn, aws_session=self._aws_session)

        state = task.state()
        if state == 'FAILED':
            raise RuntimeError(f"Braket job {self.arn} FAILED")
        if state == 'CANCELLED':
            raise RuntimeError(f"Braket job {self.arn} CANCELLED")

        result = task.result()
        # Extract measurement counts
        counts = _extract_braket_counts(result, task)
        shots_val = getattr(result, 'task_metadata', {}).get('shots', 0) if hasattr(result, 'task_metadata') else 0

        return RunResult(counts=counts, shots=shots_val, metadata={
            'backend': self.device_name,
            'job_id': self.arn,
            'state': state,
        })

    def cancel(self):
        """Cancel the Braket job."""
        from braket.aws import AwsQuantumTask

        sf.utils.info(f"Cancelling Braket job {self.arn}")
        task = AwsQuantumTask(arn=self.arn, aws_session=self._aws_session)
        task.cancel()

    @property
    def status(self) -> JobStatus:
        """Map Braket state to SF JobStatus."""
        try:
            from braket.aws import AwsQuantumTask
            task = AwsQuantumTask(arn=self.arn, aws_session=self._aws_session)
            state = task.state()
            mapping = {
                'CREATED': JobStatus.CREATED,
                'QUEUED': JobStatus.QUEUED,
                'RUNNING': JobStatus.RUNNING,
                'COMPLETED': JobStatus.COMPLETED,
                'FAILED': JobStatus.FAILED,
                'CANCELLED': JobStatus.CANCELLED,
            }
            return mapping.get(state, JobStatus.CREATED)
        except Exception:
            return JobStatus.FAILED


# ---------------------------------------------------------------------------
# BraketProvider
# ---------------------------------------------------------------------------

class BraketProvider:
    """Entry point for Amazon Braket quantum services.

    Provides access to:
      - Rigetti (Ankaa, Aspen-M)
      - IonQ (Aria, Harmony, Forte)
      - OQC (Lucy)
      - Amazon simulators (SV1, DM1, TN1)

    Args:
        region: AWS region (default: us-east-1).
        s3_bucket: S3 bucket for storing results (REQUIRED by Braket).
            Can also be set via BRAKET_S3_BUCKET env var.
        aws_access_key_id: Optional AWS access key.
        aws_secret_access_key: Optional AWS secret key.
    """

    QPU_DEVICE_NAMES = {
        'rigetti': 'Rigetti',
        'rigetti_ankaa': 'Rigetti',
        'ionq': 'IonQ',
        'ionq_aria': 'IonQ',
        'ionq_harmony': 'IonQ',
        'ionq_forte': 'IonQ',
        'oqc': 'OQC',
        'oqc_lucy': 'OQC',
        'sv1': 'SV1',
        'dm1': 'DM1',
        'tn1': 'TN1',
    }

    def __init__(
        self,
        region: str = "us-east-1",
        s3_bucket: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        import boto3
        import os

        self.region = region

        # Resolve S3 bucket
        self.s3_bucket = s3_bucket or os.getenv('BRAKET_S3_BUCKET', '')
        if not self.s3_bucket:
            self.s3_bucket = f'amazon-braket-{region}-{int(time.time())}'

        # Build AWS session
        session_kwargs: dict = {'region_name': region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key

        self._boto_session = boto3.Session(**session_kwargs)

        # Verify credentials
        try:
            sts = self._boto_session.client('sts')
            identity = sts.get_caller_identity()
            self._account_id = identity['Account']
            sf.utils.info(f"Braket: authenticated as AWS account {self._account_id}")
        except Exception as e:
            self._account_id = None
            print(f"[Braket] AWS credentials not found or invalid: {e}")
            print(f"[Braket] Configure via: aws configure OR add to .env:")
            print(f"         AWS_ACCESS_KEY_ID=...")
            print(f"         AWS_SECRET_ACCESS_KEY=...")
            print(f"         AWS_DEFAULT_REGION=us-east-1")

    def _get_aws_session(self):
        """Return the Braket AwsSession."""
        from braket.aws import AwsSession
        return AwsSession(boto_session=self._boto_session)

    def list_devices(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available Braket devices.

        Args:
            provider: Filter by provider ('Rigetti', 'IonQ', 'OQC', 'Amazon').
                      None returns all.

        Returns:
            List of dicts with keys: name, arn, status, type, provider, qubits.
        """
        aws_session = self._get_aws_session()
        devices = []

        try:
            all_devices = aws_session.get_device_region_names()
            for device_arn, region in all_devices:
                try:
                    d = aws_session.get_device(device_arn)
                    device_provider = d.get('deviceType', 'unknown')
                    if provider and provider.lower() not in device_provider.lower():
                        continue

                    para = d.get('deviceCapabilities', {})
                    action = para.get('action', {}) if isinstance(para, dict) else {}
                    if isinstance(action, str):
                        action = {}

                    devices.append({
                        'name': d.get('deviceName', device_arn),
                        'arn': d.get('deviceArn', device_arn),
                        'status': d.get('deviceStatus', 'UNKNOWN'),
                        'type': d.get('deviceType', 'unknown'),
                        'provider': d.get('providerName', 'unknown'),
                        'qubits': action.get('paradigm', {}).get('qubitCount', 0) if isinstance(action, dict) else 0,
                        'region': region,
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"[Braket] Device listing failed: {e}")

        return devices

    def _resolve_device_arn(self, device: str) -> str:
        """Resolve a shorthand device name to a full ARN."""
        provider_filter = self.QPU_DEVICE_NAMES.get(device.lower())
        available = self.list_devices(provider=provider_filter)

        # Filter by status: prefer ONLINE
        online = [d for d in available if d.get('status') == 'ONLINE']
        candidates = online if online else available

        if not candidates:
            raise ValueError(
                f"No available Braket devices for '{device}'. "
                f"Available: {[d['name'] for d in available]}"
            )

        # Try exact match first
        device_lower = device.lower()
        for d in candidates:
            if device_lower in d['name'].lower():
                return d['arn']

        # Return first available
        return candidates[0]['arn']

    def get_device_calibration(self, device: str = "rigetti") -> Dict[str, Any]:
        """Fetch calibration data for a Braket device.

        Returns T1, T2, gate fidelities, and readout errors.
        """
        arn = self._resolve_device_arn(device)
        aws_session = self._get_aws_session()

        try:
            device_info = aws_session.get_device(arn)
            caps = device_info.get('deviceCapabilities', {})

            cal_data = {}
            if isinstance(caps, dict):
                action = caps.get('action', {})
                if isinstance(action, dict):
                    paradigm = action.get('paradigm', {})
                    if isinstance(paradigm, dict):
                        cal_data['qubit_count'] = paradigm.get('qubitCount', 0)
                        cal_data['connectivity'] = paradigm.get('connectivity', {})

                    # Gate fidelities
                    for gate_name in ('1q', '2q', 'readout'):
                        fids = action.get(f'{gate_name}_fidelity', [])
                        if fids:
                            cal_data[f'{gate_name}_fidelity'] = fids

                    # Coherence
                    for metric in ('T1', 'T2'):
                        vals = action.get(metric.lower(), [])
                        if vals:
                            cal_data[metric.lower()] = vals

            sf.utils.info(f"Calibration from {device_info.get('deviceName', device)}: {len(cal_data)} metrics")
            return cal_data

        except Exception as e:
            print(f"[Braket] Calibration fetch failed: {e}")
            return {}

    def run(
        self,
        circuit: sf.Circuit,
        device: str = "sv1",
        shots: int = 1000,
        s3_folder: Optional[str] = None,
    ) -> BraketJob:
        """Submit a circuit to AWS Braket.

        Args:
            circuit: Superfermion Circuit.
            device: Device shorthand ('rigetti', 'ionq', 'oqc', 'sv1', 'dm1', 'tn1')
                    or full device ARN.
            shots: Number of measurement shots.
            s3_folder: Custom S3 folder prefix. Default: 'superfermion-jobs/'.

        Returns:
            BraketJob with the ARN.
        """
        if not self._account_id:
            raise RuntimeError(
                "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY in .env or ~/.aws/credentials."
            )

        from braket.aws import AwsDevice
        from braket.circuits import Circuit as BraketCircuit

        # Resolve device ARN
        if device.startswith('arn:aws:braket:'):
            device_arn = device
        else:
            device_arn = self._resolve_device_arn(device)

        aws_session = self._get_aws_session()
        aws_device = AwsDevice(arn=device_arn, aws_session=aws_session)
        device_name = getattr(aws_device, 'name', device)

        sf.utils.info(f"Submitting to Braket: {device_name} ({device_arn})")

        # Add measurements if not present
        bc = to_braket(circuit)
        # Braket requires explicit measurement on all qubits
        # Check if any measurement instruction is already present
        has_measure = any(
            str(instr).startswith('Measure') or 'measure' in str(instr).lower()
            for instr in bc.instructions
        )
        if not has_measure:
            from braket.circuits import Observable
            for q in range(circuit.n_qubits):
                bc = bc.add_result_type(Observable.Z(), target=[q])

        # Build S3 path
        folder = s3_folder or 'superfermion-jobs'
        s3_path = (self.s3_bucket, folder)

        task = aws_device.run(bc, s3_path, shots=shots, poll_timeout_seconds=5)

        return BraketJob(task.id, device_name, aws_session=self._get_aws_session())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_braket_counts(result, task=None) -> Dict[str, int]:
    """Extract measurement counts from a Braket result.

    Braket returns results in various formats depending on the device type.
    This normalizes them to a {bitstring: count} dict.
    """
    try:
        # Try standard measurement_counts first
        if hasattr(result, 'measurement_counts'):
            mc = result.measurement_counts
            if mc is not None:
                return dict(mc) if not isinstance(mc, dict) else mc
    except Exception:
        pass

    try:
        # Try result_types for simulators
        if hasattr(result, 'result_types'):
            for rt in result.result_types:
                if hasattr(rt, 'value'):
                    val = rt.value
                    if isinstance(val, dict):
                        return val
                    if hasattr(val, 'items'):
                        return dict(val)
    except Exception:
        pass

    try:
        # Try measurements directly
        if hasattr(result, 'measurements'):
            measurements = result.measurements
            if measurements is not None:
                counts = {}
                for m in measurements:
                    key = ''.join(str(bit) for bit in m)
                    counts[key] = counts.get(key, 0) + 1
                return counts
    except Exception:
        pass

    print("[Braket] Could not extract counts from result")
    return {}
