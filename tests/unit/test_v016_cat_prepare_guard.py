"""Regression tests for the 0.1.16 cat/prepare guard on the D1 proxy.

``Circuit.to_ir()`` returns ``LargeUnitaryDAG`` when a circuit carries an
opaque unitary above three qubits.  ``sf.cat`` and ``sf.prepare`` pass their
argument to a PyO3 constructor typed ``QuantumDAG``, so before 0.1.16 the
proxy surfaced as ``TypeError: argument 'dag': 'LargeUnitaryDAG' object
cannot be converted to 'QuantumDAG'``.  The guard turns that into a directed
``ValueError`` naming the API the caller used, and leaves every other path
unchanged.
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion._fixes import LargeUnitaryDAG


pytestmark = pytest.mark.unit


def _large_unitary_circuit(n=4):
    circuit = sf.Circuit(n)
    circuit.h(0)
    circuit.unitary(np.eye(2**n, dtype=complex), list(range(n)))
    return circuit


def _small_unitary_circuit(n=3):
    circuit = sf.Circuit(n)
    circuit.h(0)
    circuit.unitary(np.eye(2**n, dtype=complex), list(range(n)))
    return circuit


def test_to_ir_still_returns_the_proxy_and_applies_the_matrix():
    dag = _large_unitary_circuit().to_ir()
    assert isinstance(dag, LargeUnitaryDAG)
    state = np.asarray(dag.simulate_to_state("statevector").numpy())
    assert abs(abs(state[0]) ** 2 - 0.5) < 1e-12
    assert abs(abs(state[1]) ** 2 - 0.5) < 1e-12


@pytest.mark.parametrize("api_name", ["cat", "prepare"])
def test_compat_proxy_is_refused_with_a_directed_error(api_name):
    api = getattr(sf, api_name)
    with pytest.raises(ValueError) as excinfo:
        api(_large_unitary_circuit())
    message = str(excinfo.value)
    assert f"sf.{api_name}" in message
    assert "QuantumDAG" in message
    assert "<=3-qubit" in message


@pytest.mark.parametrize("api_name", ["cat", "prepare"])
def test_three_qubit_unitary_still_binds(api_name):
    handle = getattr(sf, api_name)(_small_unitary_circuit())
    assert handle is not None


@pytest.mark.parametrize("api_name", ["cat", "prepare"])
def test_plain_circuit_is_unaffected(api_name):
    circuit = sf.Circuit(4)
    circuit.h(0)
    circuit.cnot(0, 3)
    assert getattr(sf, api_name)(circuit) is not None


def test_statevector_run_of_the_same_circuit_is_still_correct():
    result = sf.run(_large_unitary_circuit(), method="statevector", shots=0)
    state = np.asarray(result.state.numpy())
    assert abs(abs(state[0]) ** 2 - 0.5) < 1e-12
    assert abs(abs(state[1]) ** 2 - 0.5) < 1e-12
