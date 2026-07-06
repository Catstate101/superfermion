"""
Distributed Tracing — Span-based tracing for quantum circuit execution pipelines.

Traces the full lifecycle: circuit_build → compile → route → submit → execute → decode → mitigate.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class SpanContext:
    """Context for distributed tracing across services."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None

    def child(self) -> SpanContext:
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
        )


@dataclass
class Span:
    """A single trace span representing a unit of work.

    Attributes:
        name: Span name (e.g., 'compile', 'execute').
        context: Trace/span IDs.
        start_time: Start timestamp.
        end_time: End timestamp (set when span finishes).
        attributes: Key-value attributes.
        events: List of timestamped events within the span.
        status: OK, ERROR, or UNSET.
        children: Child spans.
    """
    name: str
    context: SpanContext = field(default_factory=SpanContext)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "UNSET"
    children: List[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time <= 0:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def is_finished(self) -> bool:
        return self.end_time > 0

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a timestamped event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str, description: str = "") -> None:
        """Set span status (OK, ERROR)."""
        self.status = status
        if description:
            self.attributes["status.description"] = description

    def finish(self) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()
        if self.status == "UNSET":
            self.status = "OK"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
        }

    def __repr__(self) -> str:
        dur = f"{self.duration_ms:.1f}ms" if self.is_finished else "running"
        return f"Span('{self.name}', {dur}, status={self.status})"


class Tracer:
    """Distributed tracer for quantum circuit execution.

    Creates and manages spans across the execution pipeline.

    Args:
        service_name: Name of this service/component.
        parent_context: Optional parent context for distributed tracing.

    Examples:
        >>> tracer = Tracer("sf.compiler")
        >>> with tracer.span("compile") as span:
        ...     span.set_attribute("n_qubits", 10)
        ...     with tracer.span("decompose", parent=span) as child:
        ...         child.set_attribute("target", "ibm")
        >>> tracer.get_trace()  # Returns all spans
    """

    # Standard span names for quantum pipeline
    SPAN_CIRCUIT_BUILD = "circuit_build"
    SPAN_COMPILE = "compile"
    SPAN_DECOMPOSE = "decompose"
    SPAN_ROUTE = "route"
    SPAN_OPTIMIZE = "optimize"
    SPAN_SUBMIT = "submit"
    SPAN_EXECUTE = "execute"
    SPAN_DECODE = "decode"
    SPAN_MITIGATE = "mitigate"

    def __init__(
        self,
        service_name: str = "superfermion",
        parent_context: Optional[SpanContext] = None,
    ) -> None:
        self.service_name = service_name
        self._root_context = parent_context or SpanContext()
        self._spans: List[Span] = []
        self._active_span: Optional[Span] = None

    @contextmanager
    def span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Create a new trace span as a context manager.

        Args:
            name: Span name.
            parent: Optional parent span.
            attributes: Initial span attributes.

        Yields:
            The created Span.
        """
        if parent:
            ctx = parent.context.child()
        elif self._active_span:
            ctx = self._active_span.context.child()
        else:
            ctx = self._root_context.child()

        new_span = Span(
            name=name,
            context=ctx,
            attributes={"service": self.service_name, **(attributes or {})},
        )

        previous_active = self._active_span
        self._active_span = new_span

        try:
            yield new_span
            new_span.set_status("OK")
        except Exception as e:
            new_span.set_status("ERROR", str(e))
            new_span.add_event("exception", {
                "exception.type": type(e).__name__,
                "exception.message": str(e),
            })
            raise
        finally:
            new_span.finish()
            self._spans.append(new_span)
            if parent:
                parent.children.append(new_span)
            self._active_span = previous_active

    def start_span(self, name: str, **attributes: Any) -> Span:
        """Start a span manually (must call span.finish())."""
        ctx = self._root_context.child()
        span = Span(
            name=name,
            context=ctx,
            attributes={"service": self.service_name, **attributes},
        )
        self._spans.append(span)
        return span

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get all recorded spans as dictionaries."""
        return [s.to_dict() for s in self._spans]

    def get_spans(self, name: Optional[str] = None) -> List[Span]:
        """Get all spans, optionally filtered by name."""
        if name:
            return [s for s in self._spans if s.name == name]
        return self._spans[:]

    @property
    def trace_id(self) -> str:
        """The root trace ID."""
        return self._root_context.trace_id

    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
        self._active_span = None

    def __repr__(self) -> str:
        return f"Tracer(service='{self.service_name}', spans={len(self._spans)})"
