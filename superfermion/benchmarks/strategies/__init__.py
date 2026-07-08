"""SDK strategy implementations for benchmarking."""

from superfermion.benchmarks.strategies.sf_strategy import SuperfermionStrategy

_STRATEGIES = {"superfermion": SuperfermionStrategy}

try:
    from superfermion.benchmarks.strategies.qiskit_strategy import QiskitStrategy
    _STRATEGIES["qiskit"] = QiskitStrategy
except ImportError:
    pass

try:
    from superfermion.benchmarks.strategies.cirq_strategy import CirqStrategy
    _STRATEGIES["cirq"] = CirqStrategy
except ImportError:
    pass


def get_strategy(name: str):
    """Get an SDK strategy by name. Raises KeyError if not available."""
    cls = _STRATEGIES.get(name.lower())
    if cls is None:
        available = ", ".join(sorted(_STRATEGIES))
        raise KeyError(f"SDK strategy '{name}' not available. Installed: {available}")
    return cls()


def list_strategies() -> list[str]:
    """List all available SDK strategy names."""
    return sorted(_STRATEGIES)
