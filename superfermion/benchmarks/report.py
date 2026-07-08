"""
Report generation utilities for benchmark results.

Generates:
- pytest-benchmark-compatible JSON
- Side-by-side bar charts
- Speedup ratio tables
- Output quality metrics
"""

from __future__ import annotations

from typing import Any, Dict, List

from superfermion.benchmarks.protocols import BenchmarkReport, WorkoutResult


def generate_json_report(report: BenchmarkReport, path: str) -> None:
    """Write a pytest-benchmark-compatible JSON report."""
    import json
    import time
    import platform

    data = report.to_dict()
    data["machine_info"] = {
        "node": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }
    data["datetime"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_comparison_table(report: BenchmarkReport,
                              baseline_sdk: str = "superfermion") -> str:
    """Generate an ASCII comparison table with speedup ratios."""
    by_test: Dict[str, Dict[str, WorkoutResult]] = {}
    for r in report.results:
        by_test.setdefault(r.test_name, {})[r.sdk_name] = r

    lines = [
        f"{'Test':<40} {'SDK':<12} {'Time (s)':>10} {'vs ' + baseline_sdk:>12}",
        "-" * 78,
    ]
    for test_name, sdks in by_test.items():
        base = sdks.get(baseline_sdk)
        base_time = base.wall_time_s if base and base.wall_time_s > 0 else None
        for sdk_name, r in sdks.items():
            t = r.wall_time_s
            if t < 0:
                ratio_str = "FAILED"
            elif base_time and base_time > 0 and t > 0:
                ratio = t / base_time
                ratio_str = f"{ratio:.2f}x"
            else:
                ratio_str = "-"
            lines.append(f"{test_name:<40} {sdk_name:<12} {t:>10.4f} {ratio_str:>12}")
    return "\n".join(lines)


def generate_quality_table(report: BenchmarkReport) -> str:
    """Generate a table of output quality metrics (2Q gate counts, depth)."""
    lines = [
        f"{'Test':<40} {'SDK':<12} {'2Q Gates':>10} {'Depth':>8} {'Total Gates':>12}",
        "-" * 86,
    ]
    for r in report.results:
        g2q = r.extra_info.get("output_gate_count_2q", r.extra_info.get("gate_count_2q", ""))
        depth = r.extra_info.get("output_depth", "")
        total = r.extra_info.get("output_gate_count", r.extra_info.get("gate_count", ""))
        lines.append(
            f"{r.test_name:<40} {r.sdk_name:<12} {str(g2q):>10} {str(depth):>8} {str(total):>12}"
        )
    return "\n".join(lines)
