"""
Token Manager — Scoped, time-limited API token management.

Generates, validates, and manages scoped authentication tokens
for the Superfermion cloud API.

Usage:
    >>> tm = TokenManager(secret_key="my-secret")
    >>> token = tm.create_token(user="alice", scopes=[TokenScope.CIRCUIT_EXECUTE])
    >>> tm.validate_token(token.token_string)
    True
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class TokenScope(Enum):
    """Permission scopes for API tokens."""
    CIRCUIT_READ = "circuit:read"
    CIRCUIT_WRITE = "circuit:write"
    CIRCUIT_EXECUTE = "circuit:execute"
    HARDWARE_ACCESS = "hardware:access"
    RESULT_READ = "result:read"
    ADMIN = "admin"
    BILLING_READ = "billing:read"
    BILLING_WRITE = "billing:write"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_WRITE = "experiment:write"


@dataclass
class ScopedToken:
    """A scoped, time-limited API token.

    Attributes:
        token_string: The actual token string.
        user: User the token belongs to.
        scopes: Permission scopes granted.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp.
        token_id: Unique token identifier.
    """
    token_string: str
    user: str
    scopes: List[TokenScope]
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    token_id: str = field(default_factory=lambda: secrets.token_hex(8))

    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        if self.expires_at <= 0:
            return False  # No expiration
        return time.time() > self.expires_at

    def has_scope(self, scope: TokenScope) -> bool:
        """Check if token has a specific scope."""
        if TokenScope.ADMIN in self.scopes:
            return True  # Admin has all scopes
        return scope in self.scopes

    def to_dict(self) -> Dict:
        """Serialize (excluding token_string for security)."""
        return {
            "token_id": self.token_id,
            "user": self.user,
            "scopes": [s.value for s in self.scopes],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class TokenManager:
    """Manages creation, validation, and revocation of scoped tokens.

    Args:
        secret_key: HMAC secret for token signing.
        default_ttl: Default token time-to-live in seconds (1 hour).

    Examples:
        >>> tm = TokenManager(secret_key="my-server-secret")
        >>> token = tm.create_token(user="researcher", 
        ...                          scopes=[TokenScope.CIRCUIT_EXECUTE, TokenScope.RESULT_READ])
        >>> tm.validate_token(token.token_string)
        True
        >>> tm.revoke_token(token.token_id)
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        default_ttl: int = 3600,
    ) -> None:
        self._secret = (secret_key or secrets.token_hex(32)).encode()
        self._default_ttl = default_ttl
        self._tokens: Dict[str, ScopedToken] = {}
        self._revoked: Set[str] = set()

    def _sign(self, payload: str) -> str:
        """Create HMAC signature for payload."""
        return hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()

    def create_token(
        self,
        user: str,
        scopes: Optional[List[TokenScope]] = None,
        ttl: Optional[int] = None,
    ) -> ScopedToken:
        """Create a new scoped token.

        Args:
            user: User identifier.
            scopes: Permission scopes. Defaults to read-only.
            ttl: Time-to-live in seconds. Defaults to default_ttl.

        Returns:
            A new ScopedToken.
        """
        if scopes is None:
            scopes = [TokenScope.CIRCUIT_READ, TokenScope.RESULT_READ]

        ttl_val = ttl if ttl is not None else self._default_ttl
        now = time.time()
        expires = now + ttl_val if ttl_val > 0 else 0.0

        token_id = secrets.token_hex(8)
        payload = json.dumps({
            "id": token_id,
            "user": user,
            "scopes": [s.value for s in scopes],
            "exp": expires,
        }, sort_keys=True)

        signature = self._sign(payload)
        token_string = f"sf_{token_id}_{signature[:16]}"

        token = ScopedToken(
            token_string=token_string,
            user=user,
            scopes=scopes,
            created_at=now,
            expires_at=expires,
            token_id=token_id,
        )

        self._tokens[token_id] = token
        return token

    def validate_token(self, token_string: str) -> bool:
        """Validate a token string.

        Args:
            token_string: The token to validate.

        Returns:
            True if the token is valid and not expired/revoked.
        """
        # Parse token
        parts = token_string.split("_")
        if len(parts) < 3 or parts[0] != "sf":
            return False

        token_id = parts[1]

        # Check revocation
        if token_id in self._revoked:
            return False

        # Check existence
        token = self._tokens.get(token_id)
        if token is None:
            return False

        # Check expiration
        if token.is_expired:
            return False

        return True

    def get_token(self, token_string: str) -> Optional[ScopedToken]:
        """Get the full token object from a token string."""
        parts = token_string.split("_")
        if len(parts) < 3 or parts[0] != "sf":
            return None
        return self._tokens.get(parts[1])

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token by ID.

        Args:
            token_id: Token ID to revoke.

        Returns:
            True if token was found and revoked.
        """
        if token_id in self._tokens:
            self._revoked.add(token_id)
            return True
        return False

    def list_tokens(self, user: Optional[str] = None) -> List[ScopedToken]:
        """List all active (non-revoked, non-expired) tokens."""
        results = []
        for tid, token in self._tokens.items():
            if tid in self._revoked:
                continue
            if token.is_expired:
                continue
            if user and token.user != user:
                continue
            results.append(token)
        return results

    def cleanup_expired(self) -> int:
        """Remove expired tokens from memory. Returns count of removed tokens."""
        expired = [
            tid for tid, token in self._tokens.items()
            if token.is_expired
        ]
        for tid in expired:
            del self._tokens[tid]
        return len(expired)

    def __repr__(self) -> str:
        active = len([t for t in self._tokens.values() if not t.is_expired])
        revoked = len(self._revoked)
        return f"TokenManager(active={active}, revoked={revoked})"
