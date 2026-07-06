"""
Configuration Manager — TOML-based config with environment variable overrides.

Resolution order: SF_* env vars > project superfermion.toml > user ~/.superfermion/config.toml > defaults.

Supports dot-notation access, schema validation, and CLI helpers.
"""

from __future__ import annotations

import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Use tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

try:
    import tomli_w
    _HAS_TOML_WRITE = True
except ImportError:
    _HAS_TOML_WRITE = False


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# Default configuration schema
DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "",
        "version": "0.1.0",
    },
    "backends": {
        "default": "simulator",
        "fallback_chain": ["jax", "simulator"],
        "auto_select": True,
    },
    "simulation": {
        "default_shots": 1024,
        "max_qubits": 28,
        "gpu_enabled": False,
        "jit_enabled": True,
        "seed": None,
    },
    "compilation": {
        "optimization_level": 2,
        "target_basis": ["rx", "ry", "rz", "cx"],
        "auto_decompose": True,
        "auto_route": True,
    },
    "hardware": {
        "ibm_token": "",
        "ionq_token": "",
        "rigetti_token": "",
        "iqm_token": "",
        "openquantum_client_id": "",
        "openquantum_client_secret": "",
        "preferred_backend": "",
        "max_retries": 3,
        "timeout_seconds": 300,
    },
    "telemetry": {
        "enabled": True,
        "log_level": "INFO",
        "json_logs": False,
        "trace_enabled": False,
    },
    "security": {
        "credential_backend": "auto",
        "max_qubits_free": 12,
        "max_qubits_pro": 28,
        "max_qubits_unlimited": 127,
        "sanitize_inputs": True,
    },
    "experiment": {
        "auto_track": False,
        "storage_dir": "",
    },
}


class Config:
    """Unified configuration manager.

    Supports dot-notation access, environment variable overrides,
    and TOML file serialization.

    Args:
        data: Initial configuration data.
        env_prefix: Prefix for environment variable overrides.

    Examples:
        >>> config = Config()
        >>> config.get("backends.default")
        'simulator'
        >>> config.set("simulation.max_qubits", 40)
        >>> config.get("simulation.max_qubits")
        40
    """

    ENV_PREFIX = "SF_"

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        env_prefix: str = "SF_",
    ) -> None:
        self._data = copy.deepcopy(data or DEFAULT_CONFIG)
        self._env_prefix = env_prefix
        self._overrides: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Resolution: env var > override > config file > default.

        Args:
            key: Dot-separated key (e.g., 'backends.default').
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        # 1. Check environment variable
        env_key = self._env_prefix + key.upper().replace(".", "_")
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return self._parse_env_value(env_val)

        # 2. Check overrides
        if key in self._overrides:
            return self._overrides[key]

        # 3. Navigate nested dict
        parts = key.split(".")
        current: Any = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Args:
            key: Dot-separated key.
            value: Value to set.
        """
        self._overrides[key] = value

        # Also update nested dict
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None

    def _parse_env_value(self, value: str) -> Any:
        """Parse an environment variable value to appropriate Python type."""
        # Boolean
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        # None
        if value.lower() in ("none", "null", ""):
            return None
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        # List (comma-separated)
        if "," in value:
            return [v.strip() for v in value.split(",")]
        # String
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dictionary."""
        return copy.deepcopy(self._data)

    def validate(self) -> List[str]:
        """Validate configuration against schema.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Check simulation limits
        max_q = self.get("simulation.max_qubits", 28)
        if isinstance(max_q, int) and max_q > 50:
            errors.append(
                f"simulation.max_qubits={max_q} is very high. "
                f"Simulations above 30 qubits require GPU."
            )

        shots = self.get("simulation.default_shots", 1024)
        if isinstance(shots, int) and shots < 1:
            errors.append("simulation.default_shots must be >= 1")

        opt_level = self.get("compilation.optimization_level", 2)
        if isinstance(opt_level, int) and not (0 <= opt_level <= 3):
            errors.append("compilation.optimization_level must be 0-3")

        log_level = self.get("telemetry.log_level", "INFO")
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"telemetry.log_level='{log_level}' is not valid")

        return errors

    def merge(self, other: Dict[str, Any]) -> None:
        """Merge another config dict into this one (other takes precedence)."""
        self._deep_merge(self._data, other)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all configuration keys."""
        keys: List[str] = []
        self._collect_keys(self._data, prefix, keys)
        return keys

    def _collect_keys(self, data: Any, prefix: str, keys: List[str]) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    self._collect_keys(v, full_key, keys)
                else:
                    keys.append(full_key)

    def __repr__(self) -> str:
        n_keys = len(self.list_keys())
        return f"Config(keys={n_keys}, overrides={len(self._overrides)})"


def load_config(
    project_path: Optional[str] = None,
    user_path: Optional[str] = None,
) -> Config:
    """Load configuration from TOML files and environment.

    Resolution order:
    1. Environment variables (SF_*)
    2. Project-level superfermion.toml
    3. User-level ~/.superfermion/config.toml
    4. Built-in defaults

    Args:
        project_path: Path to project config file.
        user_path: Path to user config file.

    Returns:
        Merged Config object.
    """
    config = Config()

    # Load user-level config
    user_config_path = Path(user_path) if user_path else Path.home() / ".superfermion" / "config.toml"
    if user_config_path.exists() and tomllib:
        try:
            with open(user_config_path, "rb") as f:
                user_data = tomllib.load(f)
            config.merge(user_data)
        except Exception:
            pass

    # Load project-level config
    project_config_path = Path(project_path) if project_path else Path("superfermion.toml")
    if project_config_path.exists() and tomllib:
        try:
            with open(project_config_path, "rb") as f:
                project_data = tomllib.load(f)
            config.merge(project_data)
        except Exception:
            pass

    return config


def save_config(config: Config, path: str) -> Path:
    """Save configuration to a TOML file.

    Args:
        config: Config to save.
        path: Output file path.

    Returns:
        Path to saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # TOML has no null representation — strip None values before serializing.
    data = _strip_none(config.to_dict())

    if _HAS_TOML_WRITE:
        with open(p, "wb") as f:
            tomli_w.dump(data, f)
    else:
        # Fallback: write as pseudo-TOML
        with open(p, "w", encoding="utf-8") as f:
            _write_toml_fallback(f, data)

    return p


def _strip_none(d):
    if isinstance(d, dict):
        return {k: _strip_none(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [_strip_none(v) for v in d if v is not None]
    return d


def _write_toml_fallback(f: Any, data: Dict, prefix: str = "") -> None:
    """Write a simple TOML file without tomli_w."""
    for key, value in data.items():
        if isinstance(value, dict):
            section = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            f.write(f"\n[{section}]\n")
            _write_toml_fallback(f, value, "")
        elif isinstance(value, bool):
            f.write(f"{key} = {'true' if value else 'false'}\n")
        elif isinstance(value, (int, float)):
            f.write(f"{key} = {value}\n")
        elif isinstance(value, str):
            f.write(f'{key} = "{value}"\n')
        elif isinstance(value, list):
            items = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
            f.write(f"{key} = [{items}]\n")
        elif value is None:
            f.write(f"# {key} = (not set)\n")


def get_default_config() -> Config:
    """Get a Config with all default values."""
    return Config()
