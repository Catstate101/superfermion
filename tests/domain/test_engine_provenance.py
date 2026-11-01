"""Engine provenance — prove *which* binary produced a test result.

The gradient defects of 0.1.22/0.1.23 were invisible in an environment where a
stale wheel shadowed the source build, and the first question asked of every
failure report is "which engine ran this?".  This module answers it on every
machine, with no opt-in:

* always — the engine extension must exist, and its identity (interpreter,
  package path, engine path, sha256, version) is printed as test output;
* when ``SF_EXPECT_ENGINE_ROOT`` is set — the engine must live under that
  root.  CI sets it per lane (the checkout for editable lanes,
  ``$RUNNER_TEMP/venv`` for wheel lanes), which kills the "I tested a stale
  wheel" class of false result.

``tests/conftest.py`` prints the same identity in the pytest header so that
every run in the logs is self-describing.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

import superfermion as sf

pytestmark = pytest.mark.domain

#: Environment variable naming the directory the engine binary must live under.
ENGINE_ROOT_ENV = "SF_EXPECT_ENGINE_ROOT"


def _sha256(path: Path) -> str:
    """Full-file sha256, streamed so a 200 MB debug engine stays cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_identity() -> dict:
    """Everything needed to tell two installs apart."""
    engine = Path(sf._sf_core.__file__).resolve()
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "package": str(Path(sf.__file__).resolve()),
        "version": getattr(sf, "__version__", "?"),
        "engine": str(engine),
        "engine_sha256": _sha256(engine) if engine.exists() else "<missing>",
        "engine_size": engine.stat().st_size if engine.exists() else 0,
    }


def test_engine_extension_exists():
    """The compiled engine must be importable from this process."""
    engine = Path(sf._sf_core.__file__).resolve()
    assert engine.exists(), f"engine reported at {engine} but no file is there"

    identity = engine_identity()
    # Printed, not asserted: the values are for the log, the existence is
    # the assertion.  pytest shows this with -s and in the failure report.
    print("\nengine provenance:")
    for key, value in identity.items():
        print(f"  {key:14s} {value}")

    assert identity["engine_size"] > 0
    assert len(identity["engine_sha256"]) == 64


def test_engine_lives_under_expected_root():
    """``SF_EXPECT_ENGINE_ROOT`` turns "which install?" into an assertion."""
    root = os.environ.get(ENGINE_ROOT_ENV)
    engine = Path(sf._sf_core.__file__).resolve()
    if not root:
        print(
            f"\n{ENGINE_ROOT_ENV} not set - engine={engine} "
            f"(set it to assert that this run used the install you intended)"
        )
        return

    expected = Path(root).resolve()
    assert expected.is_dir(), (
        f"{ENGINE_ROOT_ENV}={root} does not exist; the probe is misconfigured"
    )
    assert engine.is_relative_to(expected), (
        f"engine {engine} is outside {expected} ({ENGINE_ROOT_ENV}); this run "
        f"tested a different SuperFermion install than the one under test "
        f"(package={Path(sf.__file__).resolve()}, python={sys.executable})"
    )
