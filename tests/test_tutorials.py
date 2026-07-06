"""Guardrail suite — every tutorial under ``docs/tutorials/`` must run green.

This test ensures shipped tutorials never silently rot when public APIs
change. Each ``XX_name.py`` file is imported as a module, and its
``main()`` function is invoked. Failures here block the release.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TUTORIAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"


def _tutorial_ids():
    if not TUTORIAL_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in TUTORIAL_DIR.glob("*.py")
        if p.name[0:2].isdigit()
    )


@pytest.mark.parametrize("filename", _tutorial_ids())
def test_tutorial_runs(filename: str) -> None:
    path = TUTORIAL_DIR / filename
    spec = importlib.util.spec_from_file_location(f"tut_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)            # import-time must succeed
        assert hasattr(module, "main"), f"{filename}: missing main()"
        module.main()                              # runtime must succeed
    finally:
        sys.modules.pop(spec.name, None)
