"""
Superfermion Security — Credential management, input sanitization, and audit trails.

Production-grade security layer for enterprise quantum computing.

Usage:
    >>> from superfermion.security import CredentialStore, Sanitizer, AuditLog
    >>> store = CredentialStore()
    >>> store.set("ibm_token", "abc123")
    >>> token = store.get("ibm_token")
    >>> sanitizer = Sanitizer()
    >>> safe_qasm = sanitizer.sanitize_qasm(user_input)
"""

from __future__ import annotations

from superfermion.security.credentials import CredentialStore, Credential
from superfermion.security.sanitize import Sanitizer, SanitizationError
from superfermion.security.audit import AuditLog, AuditEntry, AuditEventType
from superfermion.security.tls import TLSConfig, create_ssl_context
from superfermion.security.tokens import TokenManager, ScopedToken, TokenScope

__all__ = [
    "CredentialStore", "Credential",
    "Sanitizer", "SanitizationError",
    "AuditLog", "AuditEntry", "AuditEventType",
    "TLSConfig", "create_ssl_context",
    "TokenManager", "ScopedToken", "TokenScope",
]
