#!/usr/bin/env python3
"""Check that the wheels we are about to publish are the ones we promise.

Every artifact must carry the same stable-ABI tag (cp310-abi3) -- a wheel built
against one interpreter instead of the limited API would silently install on
that interpreter only -- and, before an upload, the full platform matrix must be
present. A release that quietly ships six of seven platforms is the failure mode
this guards against.

Usage:
    python scripts/check_wheels.py dist                      # tags only
    python scripts/check_wheels.py dist --require-platforms   # pre-publish gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Substrings that must all appear in a wheel filename for the platform to count.
REQUIRED_PLATFORMS = {
    "Linux x86_64 (glibc)": ("manylinux", "x86_64"),
    "Linux aarch64 (glibc)": ("manylinux", "aarch64"),
    "Linux x86_64 (musl)": ("musllinux", "x86_64"),
    "Linux aarch64 (musl)": ("musllinux", "aarch64"),
    "macOS universal2": ("macosx", "universal2"),
    "Windows amd64": ("win_amd64",),
    "Windows arm64": ("win_arm64",),
}

TAG = "cp310-abi3"


def collect(directory: Path) -> list[str]:
    wheels = sorted(p.name for p in directory.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels in {directory}")
    return wheels


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dist", nargs="?", default="dist", type=Path)
    parser.add_argument("--require-platforms", action="store_true")
    parser.add_argument("--tag", default=TAG, help=f"required ABI tag (default {TAG})")
    args = parser.parse_args()

    wheels = collect(args.dist)

    untagged = [name for name in wheels if args.tag not in name]
    if untagged:
        raise SystemExit(
            f"wheels missing the {args.tag} tag (they would only install on one "
            f"interpreter):\n  " + "\n  ".join(untagged)
        )

    duplicates = {name for name in wheels if wheels.count(name) > 1}
    if duplicates:
        raise SystemExit(f"duplicate wheels: {sorted(duplicates)}")

    if args.require_platforms:
        missing = [
            label
            for label, needles in REQUIRED_PLATFORMS.items()
            if not any(all(n in name for n in needles) for name in wheels)
        ]
        if missing:
            raise SystemExit(
                "release is missing platforms (pip would fall back to the sdist "
                "there):\n  " + "\n  ".join(missing)
            )

    print(f"OK    {len(wheels)} wheels, all {args.tag}")
    for name in wheels:
        print(f"      {name}")
    if args.require_platforms:
        print(f"OK    all {len(REQUIRED_PLATFORMS)} platforms present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
