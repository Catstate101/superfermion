"""
Superfermion Telemetry — Structured logging, tracing, and metrics.

Production-grade observability for quantum circuit execution pipelines.

Usage:
    >>> from superfermion.telemetry import Tracer, MetricsCollector, StructuredLogger
    >>> tracer = Tracer()
    >>> with tracer.span("compile") as span:
    ...     span.set_attribute("qubits", 10)
"""

from __future__ import annotations

from superfermion.telemetry.structured_logging import (
    StructuredLogger, LogLevel, get_structured_logger,
)
from superfermion.telemetry.tracing import (
    Tracer, Span, SpanContext,
)
from superfermion.telemetry.metrics import (
    MetricsCollector, Metric, MetricType,
    Counter, Gauge, Histogram,
)

__all__ = [
    "StructuredLogger", "LogLevel", "get_structured_logger",
    "Tracer", "Span", "SpanContext",
    "MetricsCollector", "Metric", "MetricType",
    "Counter", "Gauge", "Histogram",
]
