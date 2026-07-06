"""
Audit Log — Cryptographic audit trail for hardware access and operations.

Records all significant operations with timestamps, user context,
and SHA-256 integrity hashes for compliance and debugging.

Usage:
    >>> audit = AuditLog()
    >>> audit.log(AuditEventType.CIRCUIT_SUBMIT, user="alice", details={"backend": "ibm"})
    >>> entries = audit.query(user="alice")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class AuditEventType(Enum):
    """Types of auditable events."""
    # Authentication
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"

    # Circuit operations
    CIRCUIT_CREATE = "circuit.create"
    CIRCUIT_COMPILE = "circuit.compile"
    CIRCUIT_SUBMIT = "circuit.submit"
    CIRCUIT_RESULT = "circuit.result"

    # Hardware access
    HARDWARE_CONNECT = "hardware.connect"
    HARDWARE_DISCONNECT = "hardware.disconnect"
    HARDWARE_CALIBRATION = "hardware.calibration"

    # Data operations
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_DELETE = "data.delete"

    # Admin
    CONFIG_CHANGE = "config.change"
    CREDENTIAL_ROTATE = "credential.rotate"
    BUDGET_ALERT = "budget.alert"
    SECURITY_VIOLATION = "security.violation"


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        event_type: Type of event.
        timestamp: Unix timestamp.
        user: User identifier.
        details: Event-specific details.
        integrity_hash: SHA-256 hash of the entry for tamper detection.
        session_id: Session identifier for correlation.
        source_ip: Source IP address (if available).
    """
    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    user: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""
    session_id: str = ""
    source_ip: str = ""
    correlation_id: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 integrity hash."""
        payload = json.dumps({
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user": self.user,
            "details": self.details,
            "session_id": self.session_id,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the entry has not been tampered with."""
        return self.integrity_hash == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEntry:
        """Deserialize from dictionary."""
        data["event_type"] = AuditEventType(data["event_type"])
        return cls(**data)


class AuditLog:
    """Append-only audit log with integrity verification.

    Supports in-memory, file-based, and callback-based logging.

    Args:
        log_path: Path for persistent audit log file.
        max_memory_entries: Maximum entries to keep in memory.
        on_event: Optional callback for each logged event.

    Examples:
        >>> log = AuditLog()
        >>> log.log(AuditEventType.CIRCUIT_SUBMIT, user="alice",
        ...         details={"circuit": "bell_state", "backend": "ibm_eagle"})
        >>> log.log(AuditEventType.CIRCUIT_RESULT, user="alice",
        ...         details={"job_id": "abc123", "status": "completed"})
        >>> entries = log.query(user="alice", event_type=AuditEventType.CIRCUIT_SUBMIT)
        >>> len(entries)
        1
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        max_memory_entries: int = 10000,
        on_event: Optional[Callable[[AuditEntry], None]] = None,
    ) -> None:
        self._entries: List[AuditEntry] = []
        self._max_memory = max_memory_entries
        self._log_path = log_path
        self._on_event = on_event
        self._chain_hash = "0" * 64  # Genesis hash

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: AuditEventType,
        user: str = "",
        details: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        source_ip: str = "",
        correlation_id: str = "",
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            event_type: Type of event to log.
            user: User identifier.
            details: Event-specific details.
            session_id: Session ID for correlation.
            source_ip: Client IP address.
            correlation_id: Cross-service correlation ID.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            event_type=event_type,
            user=user,
            details=details or {},
            session_id=session_id,
            source_ip=source_ip,
            correlation_id=correlation_id,
        )

        # Chain hash for tamper detection
        chain_payload = f"{self._chain_hash}:{entry.integrity_hash}"
        self._chain_hash = hashlib.sha256(chain_payload.encode()).hexdigest()

        # Store in memory
        self._entries.append(entry)
        if len(self._entries) > self._max_memory:
            self._entries = self._entries[-self._max_memory:]

        # Persist to file
        if self._log_path:
            self._append_to_file(entry)

        # Invoke callback
        if self._on_event:
            try:
                self._on_event(entry)
            except Exception:
                pass  # Never let callback failures break audit logging

        return entry

    def _append_to_file(self, entry: AuditEntry) -> None:
        """Append entry to log file."""
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception:
            pass  # Audit logging should never crash the system

    def query(
        self,
        user: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit log entries.

        Args:
            user: Filter by user.
            event_type: Filter by event type.
            since: Only entries after this timestamp.
            until: Only entries before this timestamp.
            limit: Maximum entries to return.

        Returns:
            List of matching AuditEntry objects.
        """
        results: List[AuditEntry] = []

        for entry in reversed(self._entries):
            if user and entry.user != user:
                continue
            if event_type and entry.event_type != event_type:
                continue
            if since and entry.timestamp < since:
                continue
            if until and entry.timestamp > until:
                continue
            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire audit chain.

        Returns:
            True if no entries have been tampered with.
        """
        chain = "0" * 64
        for entry in self._entries:
            if not entry.verify_integrity():
                return False
            chain_payload = f"{chain}:{entry.integrity_hash}"
            chain = hashlib.sha256(chain_payload.encode()).hexdigest()

        return chain == self._chain_hash

    @property
    def count(self) -> int:
        """Number of entries in memory."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear in-memory entries (file entries are preserved)."""
        self._entries.clear()
        self._chain_hash = "0" * 64

    def export_json(self) -> str:
        """Export all in-memory entries as JSON."""
        return json.dumps(
            [e.to_dict() for e in self._entries],
            indent=2,
            default=str,
        )

    def __repr__(self) -> str:
        return f"AuditLog(entries={self.count}, path={self._log_path})"
