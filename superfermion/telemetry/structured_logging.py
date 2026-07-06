"""
Structured Logging — JSON-formatted, correlation-aware logging.

Produces machine-parseable log entries with correlation IDs,
context fields, and structured metadata.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TextIO


class LogLevel(Enum):
    """Log levels matching standard Python logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def numeric(self) -> int:
        return getattr(logging, self.value)


@dataclass
class LogEntry:
    """A structured log entry."""
    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    logger_name: str = "superfermion"
    correlation_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        entry = {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "logger": self.logger_name,
            "message": self.message,
        }
        if self.correlation_id:
            entry["correlation_id"] = self.correlation_id
        if self.context:
            entry["context"] = self.context
        return json.dumps(entry, default=str)

    def to_human(self) -> str:
        """Serialize to human-readable string."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        ctx = f" | {self.context}" if self.context else ""
        cid = f" [{self.correlation_id[:8]}]" if self.correlation_id else ""
        return f"{ts} | {self.level.value:8s} | {self.logger_name}{cid} | {self.message}{ctx}"


class StructuredLogger:
    """JSON-structured logger with correlation IDs and context binding.

    Args:
        name: Logger name.
        level: Minimum log level.
        json_output: If True, emit JSON. If False, emit human-readable.
        output: Output stream. Defaults to stdout.
        correlation_id: Optional correlation ID for request tracing.

    Examples:
        >>> logger = StructuredLogger("sf.compiler")
        >>> logger.info("Compiling circuit", qubits=10, backend="ibm")
        >>> child = logger.bind(job_id="abc123")
        >>> child.info("Job started")
    """

    def __init__(
        self,
        name: str = "superfermion",
        level: LogLevel = LogLevel.INFO,
        json_output: bool = False,
        output: Optional[TextIO] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.level = level
        self.json_output = json_output
        self._output = output or sys.stdout
        self._correlation_id = correlation_id or ""
        self._context: Dict[str, Any] = {}
        self._entries: List[LogEntry] = []
        self._max_buffer = 1000

    def bind(self, **kwargs: Any) -> StructuredLogger:
        """Create a child logger with additional context fields.

        Args:
            **kwargs: Context fields to attach to all log entries.

        Returns:
            New StructuredLogger with bound context.
        """
        child = StructuredLogger(
            name=self.name,
            level=self.level,
            json_output=self.json_output,
            output=self._output,
            correlation_id=self._correlation_id,
        )
        child._context = {**self._context, **kwargs}
        child._entries = self._entries  # Share buffer
        return child

    def with_correlation_id(self, correlation_id: str) -> StructuredLogger:
        """Create a child logger with a specific correlation ID."""
        child = self.bind()
        child._correlation_id = correlation_id
        return child

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Internal log method."""
        if level.numeric < self.level.numeric:
            return

        context = {**self._context, **kwargs}

        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            correlation_id=self._correlation_id,
            context=context,
        )

        # Buffer
        self._entries.append(entry)
        if len(self._entries) > self._max_buffer:
            self._entries = self._entries[-self._max_buffer:]

        # Output
        line = entry.to_json() if self.json_output else entry.to_human()
        try:
            self._output.write(line + "\n")
            self._output.flush()
        except Exception:
            pass

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def get_entries(self, level: Optional[LogLevel] = None, limit: int = 100) -> List[LogEntry]:
        """Retrieve buffered log entries."""
        entries = self._entries
        if level:
            entries = [e for e in entries if e.level == level]
        return entries[-limit:]

    def set_level(self, level: LogLevel) -> None:
        """Change minimum log level."""
        self.level = level

    def __repr__(self) -> str:
        return f"StructuredLogger(name='{self.name}', level={self.level.value})"


def get_structured_logger(
    name: str = "superfermion",
    json_output: bool = False,
) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name.
        json_output: Whether to output JSON format.

    Returns:
        Configured StructuredLogger.
    """
    return StructuredLogger(name=name, json_output=json_output)
