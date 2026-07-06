"""
Credential Store — Encrypted at-rest credential management.

Supports keyring-based OS credential store, environment variables,
and file-based fallback with Fernet symmetric encryption.

Usage:
    >>> store = CredentialStore()
    >>> store.set("ibm_token", "my-secret-api-key")
    >>> store.get("ibm_token")
    'my-secret-api-key'
    >>> store.list_keys()
    ['ibm_token']
    >>> store.delete("ibm_token")
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Use cryptography if available, otherwise fallback to base64 obfuscation
try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

# Use keyring if available for OS-level credential store
try:
    import keyring
    _HAS_KEYRING = True
except ImportError:
    _HAS_KEYRING = False


class CredentialBackend(Enum):
    """Backend for credential storage."""
    KEYRING = "keyring"          # OS keyring (most secure)
    ENCRYPTED_FILE = "file"      # Fernet-encrypted JSON file
    ENVIRONMENT = "env"          # Environment variables (read-only)
    MEMORY = "memory"            # In-memory only (session-scoped)


@dataclass
class Credential:
    """A stored credential with metadata.

    Attributes:
        key: Identifier for the credential (e.g., 'ibm_token').
        value: The secret value.
        provider: Which quantum provider this belongs to.
        created_at: Unix timestamp of creation.
        expires_at: Optional expiration timestamp.
        scopes: List of permission scopes.
    """
    key: str
    value: str
    provider: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    scopes: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (value is NOT included for safety)."""
        return {
            "key": self.key,
            "provider": self.provider,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "scopes": self.scopes,
        }


class CredentialStore:
    """Encrypted credential store with multiple backend support.

    Priority order for credential resolution:
    1. Environment variables (SF_<KEY> prefix)
    2. OS keyring (if available)
    3. Encrypted file store
    4. In-memory store

    Args:
        backend: Which backend to use. Default auto-detects best available.
        store_path: Path for file-based store. Default ~/.superfermion/credentials.enc
        service_name: Keyring service name.

    Examples:
        >>> store = CredentialStore()
        >>> store.set("ibm_token", "abc123", provider="ibm")
        >>> store.get("ibm_token")
        'abc123'
        >>> store.rotate("ibm_token", "new_value_456")
    """

    KEYRING_SERVICE = "superfermion"
    ENV_PREFIX = "SF_"

    def __init__(
        self,
        backend: Optional[CredentialBackend] = None,
        store_path: Optional[Path] = None,
        service_name: str = "superfermion",
    ) -> None:
        self._service_name = service_name
        self._memory_store: Dict[str, Credential] = {}

        # Auto-detect backend
        if backend is None:
            if _HAS_KEYRING:
                self._backend = CredentialBackend.KEYRING
            elif _HAS_CRYPTO:
                self._backend = CredentialBackend.ENCRYPTED_FILE
            else:
                self._backend = CredentialBackend.MEMORY
        else:
            self._backend = backend

        # File store setup
        self._store_path = store_path or Path.home() / ".superfermion" / "credentials.enc"
        self._encryption_key: Optional[bytes] = None

        if self._backend == CredentialBackend.ENCRYPTED_FILE:
            self._init_file_store()

    def _init_file_store(self) -> None:
        """Initialize file-based encrypted store."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        key_path = self._store_path.parent / ".key"

        if _HAS_CRYPTO:
            if key_path.exists():
                self._encryption_key = key_path.read_bytes()
            else:
                self._encryption_key = Fernet.generate_key()
                key_path.write_bytes(self._encryption_key)
                # Restrict permissions on key file
                try:
                    os.chmod(key_path, 0o600)
                except (OSError, AttributeError):
                    pass  # Windows doesn't support chmod the same way

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string value."""
        if _HAS_CRYPTO and self._encryption_key:
            f = Fernet(self._encryption_key)
            return f.encrypt(plaintext.encode()).decode()
        # Fallback: base64 obfuscation (NOT secure, but better than plaintext)
        return base64.b64encode(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string value."""
        if _HAS_CRYPTO and self._encryption_key:
            f = Fernet(self._encryption_key)
            return f.decrypt(ciphertext.encode()).decode()
        return base64.b64decode(ciphertext.encode()).decode()

    def _load_file_store(self) -> Dict[str, Any]:
        """Load credentials from encrypted file."""
        if not self._store_path.exists():
            return {}
        try:
            encrypted = self._store_path.read_text()
            decrypted = self._decrypt(encrypted)
            return json.loads(decrypted)
        except Exception:
            return {}

    def _save_file_store(self, data: Dict[str, Any]) -> None:
        """Save credentials to encrypted file."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        plaintext = json.dumps(data, indent=2)
        encrypted = self._encrypt(plaintext)
        self._store_path.write_text(encrypted)
        try:
            os.chmod(self._store_path, 0o600)
        except (OSError, AttributeError):
            pass

    def set(
        self,
        key: str,
        value: str,
        provider: str = "",
        expires_at: Optional[float] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Store a credential.

        Args:
            key: Credential identifier.
            value: Secret value. 
            provider: Quantum provider name (ibm, ionq, rigetti, etc.).
            expires_at: Optional unix timestamp for expiration.
            scopes: Permission scopes for the credential.
        """
        cred = Credential(
            key=key,
            value=value,
            provider=provider,
            expires_at=expires_at,
            scopes=scopes or [],
        )

        if self._backend == CredentialBackend.KEYRING and _HAS_KEYRING:
            keyring.set_password(self._service_name, key, value)
            # Store metadata in memory
            self._memory_store[key] = cred

        elif self._backend == CredentialBackend.ENCRYPTED_FILE:
            store = self._load_file_store()
            store[key] = {
                "value": value,
                "provider": provider,
                "created_at": cred.created_at,
                "expires_at": expires_at,
                "scopes": scopes or [],
            }
            self._save_file_store(store)
            self._memory_store[key] = cred

        else:  # MEMORY
            self._memory_store[key] = cred

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a credential value.

        Resolution order: env var > keyring > file > memory.

        Args:
            key: Credential identifier.
            default: Default value if not found.

        Returns:
            The secret value, or default if not found.
        """
        # 1. Check environment variable
        env_key = f"{self.ENV_PREFIX}{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val

        # 2. Check keyring
        if self._backend == CredentialBackend.KEYRING and _HAS_KEYRING:
            try:
                val = keyring.get_password(self._service_name, key)
                if val is not None:
                    return val
            except Exception:
                pass

        # 3. Check file store
        if self._backend == CredentialBackend.ENCRYPTED_FILE:
            store = self._load_file_store()
            if key in store:
                entry = store[key]
                # Check expiration
                exp = entry.get("expires_at")
                if exp and time.time() > exp:
                    return default
                return entry.get("value", default)

        # 4. Check memory
        if key in self._memory_store:
            cred = self._memory_store[key]
            if cred.is_expired:
                return default
            return cred.value

        return default

    def delete(self, key: str) -> bool:
        """Delete a credential.

        Args:
            key: Credential identifier to remove.

        Returns:
            True if credential was found and deleted.
        """
        deleted = False

        if self._backend == CredentialBackend.KEYRING and _HAS_KEYRING:
            try:
                keyring.delete_password(self._service_name, key)
                deleted = True
            except Exception:
                pass

        if self._backend == CredentialBackend.ENCRYPTED_FILE:
            store = self._load_file_store()
            if key in store:
                del store[key]
                self._save_file_store(store)
                deleted = True

        if key in self._memory_store:
            del self._memory_store[key]
            deleted = True

        return deleted

    def rotate(self, key: str, new_value: str) -> None:
        """Rotate a credential to a new value.

        Preserves metadata (provider, scopes) while updating the value
        and resetting the creation timestamp.

        Args:
            key: Credential to rotate.
            new_value: New secret value.

        Raises:
            KeyError: If the credential does not exist.
        """
        # Get existing metadata
        old_cred = self._memory_store.get(key)
        if old_cred is None:
            raise KeyError(f"Credential '{key}' not found. Cannot rotate.")

        self.set(
            key=key,
            value=new_value,
            provider=old_cred.provider,
            expires_at=old_cred.expires_at,
            scopes=old_cred.scopes,
        )

    def list_keys(self) -> List[str]:
        """List all stored credential keys."""
        keys = set(self._memory_store.keys())

        if self._backend == CredentialBackend.ENCRYPTED_FILE:
            store = self._load_file_store()
            keys.update(store.keys())

        # Also list environment-based credentials
        for env_key, _ in os.environ.items():
            if env_key.startswith(self.ENV_PREFIX):
                keys.add(env_key[len(self.ENV_PREFIX):].lower())

        return sorted(keys)

    def get_credential(self, key: str) -> Optional[Credential]:
        """Get full credential object with metadata."""
        if key in self._memory_store:
            return self._memory_store[key]
        return None

    def has(self, key: str) -> bool:
        """Check if a credential exists."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clear all in-memory credentials."""
        self._memory_store.clear()
        if self._backend == CredentialBackend.ENCRYPTED_FILE:
            self._save_file_store({})

    def __repr__(self) -> str:
        n = len(self.list_keys())
        return f"CredentialStore(backend={self._backend.value}, credentials={n})"
