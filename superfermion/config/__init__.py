"""
Superfermion Config — Unified configuration management.

Hierarchical configuration: SF_* env vars > superfermion.toml > ~/.superfermion/config.toml > defaults.

Usage:
    >>> from superfermion.config import Config, load_config
    >>> config = load_config()
    >>> config.get("backends.default")
    'simulator'
"""

from __future__ import annotations

from superfermion.config.manager import (
    Config, load_config, save_config, get_default_config,
    ConfigValidationError,
)

__all__ = [
    "Config", "load_config", "save_config", "get_default_config",
    "ConfigValidationError",
]
