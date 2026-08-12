"""Reusable lazy-loading module via PEP 562 __getattr__/__dir__.

Replaces duplicated __getattr__ implementations in ``superfermion/__init__.py``
and ``superfermion/qec/__init__.py``.

Usage:
    import sys
    from superfermion._lazy import LazyModule

    _LAZY_SUBMODULES = {"qml": "superfermion.qml"}
    _LAZY_ATTRS = {"VQE": "superfermion.algorithms.variational"}
    _ALL = ["qml", "VQE"]

    sys.modules[__name__].__class__ = type(
        __name__, (LazyModule,), {}
    )
    # LazyModule.__getattr__ and __dir__ now handle all lookups.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict


class LazyModule(ModuleType):
    """A module that lazily loads submodules and attributes on first access.

    Set ``sys.modules[__name__].__class__ = LazyModule`` and define
    ``_LAZY_SUBMODULES``, ``_LAZY_ATTRS``, and ``__all__`` as module-level
    attributes.
    """

    def __getattr__(self, name: str):
        # Guard special / dunder names
        if name.startswith("_") and name != "__all__":
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        submods = getattr(self, "_LAZY_SUBMODULES", {})
        attrs = getattr(self, "_LAZY_ATTRS", {})
        getattr(self, "__all__", [])  # __all__ set on module via __init__.py

        # 1. Submodule (e.g. sf.qml)
        mod_name = submods.get(name)
        if mod_name is not None:
            mod = importlib.import_module(mod_name)
            setattr(self, name, mod)
            return mod

        # 2. Individual attribute (e.g. sf.VQE)
        attr_mod = attrs.get(name)
        if attr_mod is not None:
            mod = importlib.import_module(attr_mod)
            attr = getattr(mod, name)
            setattr(self, name, attr)
            return attr

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __dir__(self):
        import builtins
        base = builtins.dir(type(self))
        extra = getattr(self, "__all__", [])
        return sorted(set(base + extra))
