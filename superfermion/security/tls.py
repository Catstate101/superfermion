"""
TLS Configuration — mTLS setup for cloud ↔ hardware broker communication.

Provides SSL context creation and certificate management helpers.

Usage:
    >>> config = TLSConfig(cert_file="client.pem", key_file="client.key")
    >>> ssl_ctx = create_ssl_context(config)
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TLSConfig:
    """TLS/mTLS configuration for secure communication.

    Attributes:
        cert_file: Path to client certificate (PEM).
        key_file: Path to client private key.
        ca_file: Path to CA certificate bundle.
        verify: Whether to verify server certificates.
        min_version: Minimum TLS version (default TLSv1.2).
        check_hostname: Whether to check hostname against certificate.
        ciphers: Allowed cipher suites (None for default).
    """
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    verify: bool = True
    min_version: str = "TLSv1.2"
    check_hostname: bool = True
    ciphers: Optional[str] = None

    def validate(self) -> bool:
        """Validate that referenced files exist."""
        for path_attr in [self.cert_file, self.key_file, self.ca_file]:
            if path_attr and not Path(path_attr).exists():
                return False
        return True

    def __repr__(self) -> str:
        verify_str = "verify" if self.verify else "no-verify"
        mtls_str = "mTLS" if self.cert_file else "TLS"
        return f"TLSConfig({mtls_str}, {verify_str}, min={self.min_version})"


def create_ssl_context(config: Optional[TLSConfig] = None) -> ssl.SSLContext:
    """Create an SSL context from TLS configuration.

    Args:
        config: TLS configuration. Uses secure defaults if None.

    Returns:
        Configured ssl.SSLContext.

    Examples:
        >>> config = TLSConfig(verify=True)
        >>> ctx = create_ssl_context(config)
        >>> # Use with httpx, aiohttp, etc.
    """
    if config is None:
        config = TLSConfig()

    # Map version string
    version_map = {
        "TLSv1.2": ssl.TLSVersion.TLSv1_2,
        "TLSv1.3": ssl.TLSVersion.TLSv1_3,
    }

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = version_map.get(config.min_version, ssl.TLSVersion.TLSv1_2)
    ctx.check_hostname = config.check_hostname

    if config.verify:
        ctx.verify_mode = ssl.CERT_REQUIRED
        if config.ca_file:
            ctx.load_verify_locations(config.ca_file)
        else:
            ctx.load_default_certs()
    else:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    # Load client certificate for mTLS
    if config.cert_file:
        ctx.load_cert_chain(
            certfile=config.cert_file,
            keyfile=config.key_file,
        )

    # Set ciphers
    if config.ciphers:
        ctx.set_ciphers(config.ciphers)

    return ctx
