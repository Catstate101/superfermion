"""
Plugin Ecosystem — Third-party backend, template, and pass registration.

Enables community extensions without modifying the core codebase.

Usage:
    >>> from superfermion.plugins import register_backend, register_template
    >>>
    >>> @register_backend("my_simulator")
    ... class MySimulator(BaseBackend):
    ...     def run(self, circuit, shots=0):
    ...         ...
    >>>
    >>> @register_template("my_ansatz")
    ... def my_ansatz(n_qubits: int, n_layers: int = 2):
    ...     circuit = sf.Circuit(n_qubits)
    ...     # ... build ansatz ...
    ...     return circuit
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

# ── Global registries ────────────────────────────────────────────────────

_registered_backends: Dict[str, Type] = {}
_registered_templates: Dict[str, Callable] = {}
_registered_passes: Dict[str, Type] = {}
_registered_observables: Dict[str, Type] = {}


# ── Backend registration ────────────────────────────────────────────────

def register_backend(name: str):
    """Decorator to register a custom backend.

    Args:
        name: Unique name for the backend (used with ``sf.get_backend(name)``).

    Returns:
        Decorator that registers the class.

    Example:
        >>> @register_backend("ionq_direct")
        ... class IonQDirectBackend(BaseBackend):
        ...     def run(self, circuit, shots=0):
        ...         ...
    """
    def decorator(cls: Type) -> Type:
        _registered_backends[name] = cls
        # Auto-register with superfermion's declarative factory
        try:
            from superfermion.backends.factory import register_backend as _reg
            _reg(name, lambda c=cls: c())
        except (ImportError, AttributeError, TypeError):
            import logging
            logging.getLogger(__name__).debug(
                "Failed to auto-register plugin backend '%s'", name, exc_info=True
            )
        return cls
    return decorator


# ── Template registration ───────────────────────────────────────────────

def register_template(name: str):
    """Decorator to register a circuit template (ansatz / feature map).

    Args:
        name: Unique template name.

    Returns:
        Decorator that registers the function.

    Example:
        >>> @register_template("qaoa_mixer")
        ... def qaoa_mixer(n_qubits: int, beta: float):
        ...     circuit = sf.Circuit(n_qubits)
        ...     for q in range(n_qubits):
        ...         circuit.rx(q, 2 * beta)
        ...     return circuit
    """
    def decorator(fn: Callable) -> Callable:
        _registered_templates[name] = fn
        return fn
    return decorator


# ── Compiler pass registration ──────────────────────────────────────────

def register_pass(name: str):
    """Decorator to register a custom compiler pass.

    Args:
        name: Unique pass name.

    Example:
        >>> @register_pass("noise_aware_routing")
        ... class NoiseAwareRouter(BasePass):
        ...     def run(self, circuit):
        ...         ...
    """
    def decorator(cls: Type) -> Type:
        _registered_passes[name] = cls
        # Auto-register with the compiler PassManager
        try:
            from superfermion.compiler.manager import PassManager
            PassManager.add_plugin_pass(name, cls)
        except (ImportError, AttributeError):
            import logging
            logging.getLogger(__name__).debug(
                "Failed to auto-register plugin pass '%s'", name, exc_info=True
            )
        return cls
    return decorator


# ── Observable registration ──────────────────────────────────────────────

def register_observable(name: str):
    """Decorator to register a custom observable type.

    Args:
        name: Unique observable name.

    Example:
        >>> @register_observable("ising_hamiltonian")
        ... class IsingHamiltonian(SparsePauliOp):
        ...     ...
    """
    def decorator(cls: Type) -> Type:
        _registered_observables[name] = cls
        return cls
    return decorator


# ── Query functions ──────────────────────────────────────────────────────

def get_backend(name: str) -> Optional[Type]:
    """Retrieve a registered plugin backend class."""
    return _registered_backends.get(name)


def get_template(name: str) -> Optional[Callable]:
    """Retrieve a registered plugin template function."""
    return _registered_templates.get(name)


def get_pass(name: str) -> Optional[Type]:
    """Retrieve a registered plugin compiler pass."""
    return _registered_passes.get(name)


def list_backends() -> List[str]:
    """List all registered plugin backends."""
    return sorted(_registered_backends.keys())


def list_templates() -> List[str]:
    """List all registered plugin templates."""
    return sorted(_registered_templates.keys())


def list_passes() -> List[str]:
    """List all registered plugin passes."""
    return sorted(_registered_passes.keys())


def list_all() -> Dict[str, List[str]]:
    """Return all registered plugins organized by category."""
    return {
        "backends": list_backends(),
        "templates": list_templates(),
        "passes": list_passes(),
        "observables": sorted(_registered_observables.keys()),
    }


# ── Auto-discovery ───────────────────────────────────────────────────────

def discover_plugins(paths: Optional[List[str]] = None):
    """Auto-discover plugins from the file system.

    Searches for Python files matching ``sf_plugin_*.py`` in the given
    paths (defaults to ``superfermion/plugins/contrib/``).

    Args:
        paths: Directories to search for plugins.
    """
    import importlib
    import os
    import sys

    search_paths = paths or []
    if not search_paths:
        # Default: look in contrib directory
        contrib_dir = os.path.join(os.path.dirname(__file__), "contrib")
        if os.path.isdir(contrib_dir):
            search_paths.append(contrib_dir)

    for sp in search_paths:
        if not os.path.isdir(sp):
            continue
        for fname in os.listdir(sp):
            if fname.startswith("sf_plugin_") and fname.endswith(".py"):
                module_name = fname[:-3]
                # Add parent dir to path for import
                sys.path.insert(0, sp)
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    pass
                finally:
                    if sp in sys.path:
                        sys.path.remove(sp)
