"""
Superfermion Benchmarks — Provider-agnostic benchmarking with Benchpress-style workouts.

Usage::

    from superfermion.benchmarks import BenchmarkRunner, TopologyFactory
    from superfermion.benchmarks.strategies import SuperfermionStrategy

    runner = BenchmarkRunner()
    backend = TopologyFactory.create("heavy_hex", n_qubits=127,
                                     basis_gates=["rz", "sx", "x", "ecr"])
    report = runner.run(
        strategies=[SuperfermionStrategy()],
        backend=backend,
    )
    print(report.summary_table())
    report.plot(save_path="benchmark_results.png")
    report.to_json("benchmark_results.json")
"""

from superfermion.benchmarks.runner import BenchmarkRunner
from superfermion.benchmarks.topologies import TopologyFactory
from superfermion.benchmarks.strategies import list_strategies, get_strategy

__all__ = [
    "BenchmarkRunner",
    "TopologyFactory",
    "list_strategies",
    "get_strategy",
]
