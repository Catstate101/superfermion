#!/usr/bin/env python3
"""Check that one version number is used everywhere users can see it.

`pyproject.toml` becomes the wheel/sdist metadata, `Cargo.toml` names the crate
inside the sdist, and `superfermion.__version__` is what a user reads at runtime
(and what serialization manifests and error messages embed). They drift apart
silently: the artifacts still install, `pip show` says 0.1.17 and the imported
package keeps answering 0.1.16. On a tag build the tag itself has to match too.

Usage:
    python scripts/check_version.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def init_version() -> str:
    text = (ROOT / "superfermion" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("superfermion/__init__.py has no __version__ literal")
    return match.group(1)


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))

    versions = {
        "pyproject.toml": pyproject["project"]["version"],
        "Cargo.toml": cargo["workspace"]["package"]["version"],
        "superfermion/__init__.py": init_version(),
    }
    # Branch builds have nothing to compare against; only tags pin a version.
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ.get("GITHUB_REF_NAME", "")
        versions[f"tag {tag}"] = tag.removeprefix("v")

    distinct = set(versions.values())
    if len(distinct) != 1:
        for source, value in versions.items():
            print(f"  {source:<26} {value}", file=sys.stderr)
        print("error: version mismatch across the release surface", file=sys.stderr)
        return 1

    print(f"version {distinct.pop()} consistent: {', '.join(versions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
