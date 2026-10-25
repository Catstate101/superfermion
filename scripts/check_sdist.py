#!/usr/bin/env python3
"""Fail if a built sdist cannot be installed from source.

Maturin derives the sdist file list from the *enabled* cargo dependency graph.
A crate that is only reachable through an optional, non-default path dependency
is therefore left out of the tarball while the depending manifest still points
at `../that-crate` -- and every `pip install superfermion` from source dies
while loading manifests.  That is exactly what shipped in 0.1.16:

    failed to load manifest for dependency `sf-gpu`
    failed to read `.../crates/sf-gpu/Cargo.toml`

This script checks the artifact instead of the build configuration: unpack it,
then require every workspace member, every `path =` dependency and every
`include_str!`/`include_bytes!` asset to be inside the tarball, and finally let
cargo load the extracted workspace for real.

Usage:
    python scripts/check_sdist.py dist          # directory containing *.tar.gz
    python scripts/check_sdist.py dist --skip-cargo
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import tomllib

DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
ASSET_RE = re.compile(r'include_(?:str|bytes)!\s*\(\s*"([^"]+)"\s*\)')


def find_sdist(directory: Path) -> Path:
    tarballs = sorted(directory.glob("*.tar.gz"))
    if len(tarballs) != 1:
        raise SystemExit(f"expected exactly one *.tar.gz in {directory}, found {len(tarballs)}")
    return tarballs[0]


def extract(tarball: Path, destination: Path) -> Path:
    with tarfile.open(tarball) as tf:
        try:
            tf.extractall(destination, filter="data")
        except TypeError:  # interpreter without the extraction filter
            tf.extractall(destination)
    roots = [p for p in destination.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"sdist must unpack to a single directory, got {[p.name for p in roots]}")
    root = roots[0]
    for required in ("pyproject.toml", "Cargo.toml"):
        if not (root / required).is_file():
            raise SystemExit(f"sdist root {root.name} has no {required}")
    return root


def manifests(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("Cargo.toml") if "target" not in p.parts)


def workspace_members(root: Path) -> list[Path]:
    """Manifest files that the root Cargo.toml promises to contain."""
    data = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    members: list[Path] = [root / "Cargo.toml"]
    for entry in data.get("workspace", {}).get("members", []):
        matched = sorted(p for p in root.glob(entry) if p.is_dir())
        if not matched:
            raise SystemExit(f"workspace member {entry!r} is not in the sdist")
        members.extend(p / "Cargo.toml" for p in matched)
    return members


def path_dependencies(manifest: Path) -> list[tuple[str, Path]]:
    """(dependency name, target directory) for every `path =` dependency."""
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    tables = [data.get(name, {}) for name in DEP_TABLES]
    tables.append(data.get("workspace", {}).get("dependencies", {}))
    for target in data.get("target", {}).values():
        tables.extend(target.get(name, {}) for name in DEP_TABLES)

    found = []
    for table in tables:
        for name, spec in table.items():
            if isinstance(spec, dict) and "path" in spec:
                found.append((name, (manifest.parent / spec["path"]).resolve()))
    return found


def missing_path_deps(root: Path) -> list[str]:
    problems = []
    base = root.resolve()
    for manifest in manifests(root):
        where = manifest.relative_to(root).as_posix()
        for name, target in path_dependencies(manifest):
            if base not in target.parents and target != base:
                problems.append(f"{where}: dependency {name} points outside the sdist ({target})")
            elif not (target / "Cargo.toml").is_file():
                missing = (target / "Cargo.toml").relative_to(base).as_posix()
                problems.append(f"{where}: dependency {name} is missing {missing}")
    return problems


def missing_assets(root: Path) -> list[str]:
    """`include_str!`/`include_bytes!` targets are resolved by rustc, not cargo."""
    problems = []
    for source in root.rglob("*.rs"):
        if "target" in source.parts:
            continue
        for relative in ASSET_RE.findall(source.read_text(encoding="utf-8", errors="replace")):
            if not (source.parent / relative).is_file():
                problems.append(
                    f"{source.relative_to(root).as_posix()}: include target {relative} is missing"
                )
    return problems


def cargo_loads(root: Path) -> None:
    if shutil.which("cargo") is None:
        print("SKIP  cargo is not installed; manifest resolution was not exercised")
        return
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = "\n".join(result.stderr.strip().splitlines()[:8])
        raise SystemExit(
            f"cargo metadata failed inside the sdist (exit {result.returncode}):\n{detail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dist", nargs="?", default="dist", type=Path, help="directory holding the sdist"
    )
    parser.add_argument("--skip-cargo", action="store_true", help="do not run cargo metadata")
    args = parser.parse_args()

    tarball = find_sdist(args.dist)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = extract(tarball, Path(tmp))

        problems = missing_path_deps(root) + missing_assets(root)
        if problems:
            raise SystemExit("sdist is not self-contained:\n  " + "\n  ".join(problems))
        if not args.skip_cargo:
            cargo_loads(root)

        crates = len(workspace_members(root))
        print(
            f"OK    {tarball.name}: {crates} workspace manifests, "
            f"all path dependencies and include assets present"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
