"""The GPU probe must answer, never raise.

``sf-gpu`` documents that it "gracefully returns ``is_available() = false``"
on a machine without CUDA. Before this test existed, a driver without a
loadable nvrtc made the probe panic inside cudarc's dynamic loader, and the
panic surfaced in Python as ``pyo3_runtime.PanicException`` — which is not
catchable as ``RuntimeError`` and breaks every caller that guards the GPU.
"""

import pytest

from superfermion._sf_core import gpu_available, gpu_diagnose


def test_gpu_available_returns_bool_without_raising():
    assert isinstance(gpu_available(), bool)


def test_gpu_diagnose_returns_str_without_raising():
    text = gpu_diagnose()
    assert isinstance(text, str)
    assert text.strip()


def test_gpu_request_without_gpu_raises_runtime_error():
    """A missing GPU must be a RuntimeError with a diagnostic, not a panic."""
    import superfermion as sf

    if gpu_available():
        pytest.skip("GPU present: the unavailable path cannot be exercised")

    circuit = sf.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    with pytest.raises(RuntimeError) as excinfo:
        sf.run(circuit, device="gpu", shots=1)
    assert "GPU not available" in str(excinfo.value)
