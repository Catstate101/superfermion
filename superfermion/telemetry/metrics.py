"""
Metrics Collector — Counter, gauge, and histogram metrics for observability.

Collects metrics like circuit_depth, compilation_time_ms, hardware_queue_depth, etc.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class Counter:
    """A monotonically increasing counter.

    Usage:
        >>> c = Counter("sf_circuits_compiled")
        >>> c.inc()
        >>> c.inc(5)
        >>> c.value
        6
    """

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter."""
        if amount < 0:
            raise ValueError("Counter can only be incremented (use Gauge for decrements)")
        self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        """Reset to zero (use sparingly)."""
        self._value = 0.0

    def to_metric(self) -> Metric:
        return Metric(self.name, self._value, MetricType.COUNTER, labels=self.labels)

    def __repr__(self) -> str:
        return f"Counter('{self.name}', value={self._value})"


class Gauge:
    """A gauge that can go up and down.

    Usage:
        >>> g = Gauge("sf_active_jobs")
        >>> g.set(5)
        >>> g.inc()
        >>> g.dec()
        >>> g.value
        5
    """

    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value: float = 0.0

    def set(self, value: float) -> None:
        """Set the gauge to a specific value."""
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value

    def to_metric(self) -> Metric:
        return Metric(self.name, self._value, MetricType.GAUGE, labels=self.labels)

    def __repr__(self) -> str:
        return f"Gauge('{self.name}', value={self._value})"


class Histogram:
    """A histogram for measuring distributions (e.g., latency).

    Usage:
        >>> h = Histogram("sf_compilation_time_ms", buckets=[10, 50, 100, 500])
        >>> h.observe(45.2)
        >>> h.observe(120.5)
        >>> h.mean
        82.85
    """

    DEFAULT_BUCKETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "ms",
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.unit = unit
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._values: List[float] = []
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self._buckets}
        self._sum: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._values.append(value)
        self._sum += value
        self._count += 1
        self._min = min(self._min, value)
        self._max = max(self._max, value)

        for bucket in self._buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    @property
    def min(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self._count > 0 else 0.0

    @property
    def stddev(self) -> float:
        """Standard deviation."""
        if self._count < 2:
            return 0.0
        mean = self.mean
        variance = sum((v - mean) ** 2 for v in self._values) / (self._count - 1)
        return math.sqrt(variance)

    def percentile(self, p: float) -> float:
        """Calculate a percentile (0-100)."""
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self._count,
            "sum": self._sum,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "stddev": self.stddev,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "buckets": self._bucket_counts,
        }

    def __repr__(self) -> str:
        return (
            f"Histogram('{self.name}', count={self._count}, "
            f"mean={self.mean:.2f}{self.unit})"
        )


class MetricsCollector:
    """Central metrics registry for all Superfermion metrics.

    Provides pre-defined metrics for quantum circuit operations.

    Usage:
        >>> mc = MetricsCollector()
        >>> mc.circuit_depth.observe(15)
        >>> mc.compilation_time.observe(42.5)
        >>> mc.circuits_executed.inc()
        >>> report = mc.report()
    """

    def __init__(self) -> None:
        # Pre-defined counters
        self.circuits_created = Counter("sf_circuits_created", "Total circuits created")
        self.circuits_compiled = Counter("sf_circuits_compiled", "Total circuits compiled")
        self.circuits_executed = Counter("sf_circuits_executed", "Total circuits executed")
        self.gradient_evaluations = Counter("sf_gradient_evaluations", "Total gradient evals")
        self.hardware_jobs_submitted = Counter("sf_hardware_jobs", "Jobs sent to hardware")

        # Pre-defined gauges
        self.active_jobs = Gauge("sf_active_jobs", "Currently running jobs")
        self.hardware_queue_depth = Gauge("sf_hardware_queue_depth", "Hardware queue depth")
        self.memory_usage_mb = Gauge("sf_memory_usage_mb", "Memory usage in MB")

        # Pre-defined histograms
        self.circuit_depth = Histogram("sf_circuit_depth", "Circuit depth distribution", unit="")
        self.compilation_time = Histogram(
            "sf_compilation_time_ms", "Compilation time",
            buckets=[1, 5, 10, 50, 100, 500, 1000], unit="ms"
        )
        self.execution_time = Histogram(
            "sf_execution_time_ms", "Execution time",
            buckets=[10, 50, 100, 500, 1000, 5000], unit="ms"
        )
        self.qec_logical_error_rate = Histogram(
            "sf_qec_logical_error_rate", "QEC logical error rate", unit=""
        )

        # Custom metrics registry
        self._custom_counters: Dict[str, Counter] = {}
        self._custom_gauges: Dict[str, Gauge] = {}
        self._custom_histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a custom counter."""
        if name not in self._custom_counters:
            self._custom_counters[name] = Counter(name, description)
        return self._custom_counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a custom gauge."""
        if name not in self._custom_gauges:
            self._custom_gauges[name] = Gauge(name, description)
        return self._custom_gauges[name]

    def histogram(self, name: str, description: str = "", **kwargs: Any) -> Histogram:
        """Get or create a custom histogram."""
        if name not in self._custom_histograms:
            self._custom_histograms[name] = Histogram(name, description, **kwargs)
        return self._custom_histograms[name]

    def report(self) -> Dict[str, Any]:
        """Generate a full metrics report."""
        return {
            "counters": {
                "circuits_created": self.circuits_created.value,
                "circuits_compiled": self.circuits_compiled.value,
                "circuits_executed": self.circuits_executed.value,
                "gradient_evaluations": self.gradient_evaluations.value,
                "hardware_jobs_submitted": self.hardware_jobs_submitted.value,
                **{k: v.value for k, v in self._custom_counters.items()},
            },
            "gauges": {
                "active_jobs": self.active_jobs.value,
                "hardware_queue_depth": self.hardware_queue_depth.value,
                "memory_usage_mb": self.memory_usage_mb.value,
                **{k: v.value for k, v in self._custom_gauges.items()},
            },
            "histograms": {
                "circuit_depth": self.circuit_depth.to_dict(),
                "compilation_time_ms": self.compilation_time.to_dict(),
                "execution_time_ms": self.execution_time.to_dict(),
                **{k: v.to_dict() for k, v in self._custom_histograms.items()},
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.__init__()

    def __repr__(self) -> str:
        total = (
            int(self.circuits_created.value)
            + int(self.circuits_compiled.value)
            + int(self.circuits_executed.value)
        )
        return f"MetricsCollector(total_operations={total})"
