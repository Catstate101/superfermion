"""
BenchmarkRunner — Pipeline pattern for executing benchmark workouts.

Pattern: Pipeline
Problem: Benchmark execution is a fixed sequence of steps.
Solution: Each step (discover → setup → time → validate → report) is
          independently testable and replaceable.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence

from superfermion.benchmarks.protocols import (
    BenchmarkBackend,
    BenchmarkReport,
    SDKStrategy,
    WorkoutResult,
)


class BenchmarkRunner:
    """Execute benchmark workouts across multiple SDK strategies.

    Usage::

        runner = BenchmarkRunner()
        report = runner.run(
            workouts=["bench_qv_build", "bench_dtc_build"],
            strategies=[SuperfermionStrategy(), QiskitStrategy()],
            backend=backend,
        )
        report.plot()
    """

    def run(
        self,
        workouts: Sequence[str] | None = None,
        strategies: Sequence[SDKStrategy] | None = None,
        backend: BenchmarkBackend | None = None,
        rounds: int = 3,
        category: str = "all",
    ) -> RunnerReport:
        """Run benchmarks and collect results.

        Args:
            workouts: List of workout names to run. If None, runs all
                      in the specified category.
            strategies: SDKStrategy instances to benchmark.
            backend: Target hardware for transpilation workouts.
            rounds: Number of timed iterations per workout.
            category: "all", "construction", "manipulation", or "transpilation".
        """
        from superfermion.benchmarks.workouts import (
            ALL_WORKOUTS,
            CONSTRUCTION_WORKOUTS,
            MANIPULATION_WORKOUTS,
            TRANSPILATION_WORKOUTS,
        )

        if strategies is None:
            from superfermion.benchmarks.strategies import get_strategy
            strategies = [get_strategy("superfermion")]

        workout_map: Dict[str, Callable] = {}
        if category == "all" or category is None:
            workout_map = ALL_WORKOUTS
        elif category == "construction":
            workout_map = CONSTRUCTION_WORKOUTS
        elif category == "manipulation":
            workout_map = MANIPULATION_WORKOUTS
        elif category == "transpilation":
            workout_map = TRANSPILATION_WORKOUTS
        else:
            workout_map = ALL_WORKOUTS

        if workouts is not None:
            workout_map = {k: v for k, v in workout_map.items() if k in workouts}

        report = RunnerReport()
        total = len(workout_map) * len(strategies)
        done = 0

        for workout_name, workout_fn in workout_map.items():
            for strategy in strategies:
                done += 1
                label = f"[{done}/{total}] {workout_name} ({strategy.name})"
                try:
                    result = workout_fn(
                        strategy=strategy,
                        rounds=rounds,
                        backend=backend,
                    )
                    report.add(result)
                    print(f"  OK  {label}: {result.wall_time_s:.4f}s")
                except Exception as e:
                    print(f"  FAIL {label}: {e}")
                    report.add(WorkoutResult(
                        test_name=workout_name,
                        sdk_name=strategy.name,
                        sdk_version=strategy.version,
                        wall_time_s=-1.0,
                        extra_info={"error": str(e)},
                    ))

        return report


class RunnerReport(BenchmarkReport):
    """Extended report with display and export capabilities."""

    def summary_table(self) -> str:
        """Generate an ASCII summary table of results."""
        if not self.results:
            return "No results."

        header = f"{'Test':<40} {'SDK':<15} {'Time (s)':>12} {'Rounds':>8}"
        lines = [header, "-" * len(header)]
        for r in self.results:
            time_str = f"{r.wall_time_s:.4f}" if r.wall_time_s >= 0 else "FAILED"
            lines.append(
                f"{r.test_name:<40} {r.sdk_name:<15} {time_str:>12} {r.rounds:>8}"
            )
        return "\n".join(lines)

    def speedup_table(self, baseline: str = "superfermion") -> str:
        """Show speedup ratios relative to a baseline SDK."""
        by_test: Dict[str, Dict[str, float]] = {}
        for r in self.results:
            by_test.setdefault(r.test_name, {})[r.sdk_name] = r.wall_time_s

        header = f"{'Test':<40} {'Speedup vs ' + baseline:>25}"
        lines = [header, "-" * len(header)]
        for test_name, sdk_times in by_test.items():
            base_time = sdk_times.get(baseline, -1)
            for sdk, t in sdk_times.items():
                if sdk == baseline:
                    continue
                if base_time > 0 and t > 0:
                    ratio = t / base_time
                    lines.append(f"{test_name:<40} {sdk}: {ratio:.2f}x")
                else:
                    lines.append(f"{test_name:<40} {sdk}: N/A")
        return "\n".join(lines)

    def to_json(self, path: str | None = None) -> str:
        """Export to pytest-benchmark-compatible JSON."""
        import json
        data = self.to_dict()
        data["machine_info"] = _machine_info()
        data["commit_info"] = {}
        data["datetime"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        text = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def plot(self, title: str = "Benchpress Workouts", save_path: str | None = None):
        """Generate side-by-side bar chart. Requires matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot.")
            return

        by_test: Dict[str, Dict[str, float]] = {}
        for r in self.results:
            if r.wall_time_s >= 0:
                by_test.setdefault(r.test_name, {})[r.sdk_name] = r.wall_time_s

        tests = list(by_test.keys())
        sdks = sorted({sdk for times in by_test.values() for sdk in times})
        import numpy as np
        x = np.arange(len(tests))
        width = 0.8 / max(len(sdks), 1)

        fig, ax = plt.subplots(figsize=(max(14, len(tests) * 1.2), 6))
        for i, sdk in enumerate(sdks):
            vals = [by_test[t].get(sdk, 0) for t in tests]
            ax.bar(x + i * width, vals, width, label=sdk)

        ax.set_ylabel("Wall time (s)")
        ax.set_title(title)
        ax.set_xticks(x + width * (len(sdks) - 1) / 2)
        ax.set_xticklabels(tests, rotation=45, ha="right", fontsize=8)
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Chart saved to {save_path}")
        plt.close(fig)
        return fig


def _machine_info() -> dict:
    import platform
    return {
        "node": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }
