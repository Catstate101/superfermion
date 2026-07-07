"""Unit tests for SFEstimator, SFSampler, and PrimitiveJob."""

import math

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.observables.core import PauliString, SparsePauliOp
from superfermion.parameters import param
from superfermion.primitives import (
    EstimatorData,
    EstimatorPubResult,
    PrimitiveJob,
    SFEstimator,
    SFSampler,
    SamplerData,
    SamplerPubResult,
    ShotResult,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def bell_circuit() -> Circuit:
    return Circuit(2).h(0).cnot(0, 1)


class TestPrimitiveJob:
    def test_result_returns_pub_results(self):
        data = EstimatorPubResult(data=EstimatorData(evs=1.0))
        job = PrimitiveJob([data])
        assert job.result() == [data]

    def test_getitem_access(self):
        first = EstimatorPubResult(data=EstimatorData(evs=0.5))
        second = EstimatorPubResult(data=EstimatorData(evs=-0.5))
        job = PrimitiveJob([first, second])
        assert job[0] is first
        assert job[1] is second


class TestSFEstimator:
    def test_run_exact_zz_on_bell_state(self, bell_circuit):
        estimator = SFEstimator(device="cpu", shots=0)
        job = estimator.run([(bell_circuit, PauliString("ZZ"))])
        result = job.result()
        assert len(result) == 1
        assert isinstance(result[0], EstimatorPubResult)
        assert result[0].data.evs == pytest.approx(1.0, abs=1e-10)
        assert result[0].data.stds == 0.0
        assert result[0].metadata["method"] == "statevector"

    def test_run_multiple_pubs(self, bell_circuit):
        estimator = SFEstimator(device="cpu", shots=0)
        pubs = [
            (bell_circuit, PauliString("ZZ")),
            (bell_circuit, PauliString("XX")),
        ]
        results = estimator.run(pubs).result()
        assert len(results) == 2
        assert results[0].data.evs == pytest.approx(1.0, abs=1e-10)
        assert results[1].data.evs == pytest.approx(1.0, abs=1e-10)

    def test_run_with_parameter_binding(self):
        theta = param("theta")
        circuit = Circuit(1).rx(theta, 0)
        observable = PauliString("Z")
        estimator = SFEstimator(device="cpu", shots=0)
        job = estimator.run([(circuit, observable, [math.pi])])
        result = job.result()[0]
        assert result.data.evs == pytest.approx(-1.0, abs=1e-10)

    def test_run_with_dict_parameter_binding(self):
        theta = param("theta")
        circuit = Circuit(1).rx(theta, 0)
        observable = PauliString("Z")
        estimator = SFEstimator(device="cpu", shots=0)
        job = estimator.run([(circuit, observable, {"theta": math.pi})])
        result = job.result()[0]
        assert result.data.evs == pytest.approx(-1.0, abs=1e-10)

    def test_run_shot_override_uses_exact_path_with_statevector_backend(self, bell_circuit):
        estimator = SFEstimator(device="cpu", shots=0, seed=0)
        job = estimator.run([(bell_circuit, PauliString("ZZ"))], shots=500)
        result = job.result()[0]
        assert result.data.evs == pytest.approx(1.0, abs=1e-10)
        assert result.data.stds == 0.0
        assert result.metadata["shots"] == 500

    def test_sparse_pauli_op_observable(self, bell_circuit):
        observable = SparsePauliOp.from_dict({"ZZ": 1.0, "II": 0.25})
        estimator = SFEstimator(device="cpu", shots=0)
        ev = estimator.run([(bell_circuit, observable)]).result()[0].data.evs
        assert ev == pytest.approx(1.25, abs=1e-10)

    def test_parameter_mismatch_raises(self):
        theta = param("theta")
        phi = param("phi")
        circuit = Circuit(1).rx(theta, 0).ry(phi, 0)
        estimator = SFEstimator(device="cpu", shots=0)
        with pytest.raises(ValueError, match="Parameter mismatch"):
            estimator.run([(circuit, PauliString("Z"), [0.1])])


class TestSFSampler:
    def test_run_returns_counts(self, bell_circuit):
        sampler = SFSampler(device="cpu", default_shots=200, seed=42)
        job = sampler.run([bell_circuit], shots=200)
        result = job.result()
        assert len(result) == 1
        assert isinstance(result[0], SamplerPubResult)
        meas = result[0].data.meas
        assert isinstance(meas, ShotResult)
        assert meas.n_qubits == 2
        assert sum(meas.counts.values()) == 200
        assert set(meas.counts.keys()).issubset({"00", "11"})

    def test_run_quasi_probs_sum_to_one(self, bell_circuit):
        sampler = SFSampler(device="cpu", default_shots=100, seed=1)
        meas = sampler.run([bell_circuit]).result()[0].data.meas
        assert sum(meas.quasi_probs.values()) == pytest.approx(1.0, abs=1e-10)

    def test_get_counts_property(self, bell_circuit):
        sampler = SFSampler(device="cpu", default_shots=50, seed=7)
        meas = sampler.run([bell_circuit]).result()[0].data.meas
        assert meas.get_counts == meas.counts

    def test_run_tuple_pub_with_parameters(self):
        theta = param("theta")
        circuit = Circuit(1).rx(theta, 0).measure(0)
        sampler = SFSampler(device="cpu", default_shots=100, seed=0)
        job = sampler.run([(circuit, [math.pi])], shots=100)
        counts = job.result()[0].data.meas.counts
        assert counts.get("1", 0) == 100

    def test_run_multiple_circuits(self, bell_circuit):
        plus = Circuit(1).h(0)
        sampler = SFSampler(device="cpu", default_shots=64, seed=3)
        results = sampler.run([bell_circuit, plus], shots=64).result()
        assert len(results) == 2
        assert results[0].data.meas.n_qubits == 2
        assert results[1].data.meas.n_qubits == 1

    def test_sampler_data_structure(self, bell_circuit):
        sampler = SFSampler(device="cpu", default_shots=32, seed=5)
        pub = sampler.run([bell_circuit]).result()[0]
        assert isinstance(pub.data, SamplerData)
        assert isinstance(pub.data.meas, ShotResult)
