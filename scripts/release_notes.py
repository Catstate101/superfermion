#!/usr/bin/env python3
"""Render a GitHub Release body for a tag from CHANGELOG.md.

Why this exists
---------------
The repository has 26 tags but no GitHub Release objects: the releases page is
empty, so a reader who arrives from PyPI finds no per-version notes, no install
snippet and no diff link.  The CHANGELOG already carries the text -- this turns
one of its sections into a release body so the two cannot drift.

The body is the CHANGELOG section verbatim, followed by an install block
generated from the tag itself.  The section's own ``###`` headings sit one level
below the release title this script prepends, so they are not rewritten.
Versions the CHANGELOG does not cover (its entries start at 0.1.16) have no
section; rather than invent prose, the fallback lists the commits between the
previous tag and this one and says so.

Usage::

    python scripts/release_notes.py                 # latest tag, to stdout
    python scripts/release_notes.py v0.1.25         # a specific tag
    python scripts/release_notes.py v0.1.18 --out notes.md

    # Publishing (needs `gh auth login` first; nothing is pushed by this script)
    gh release create "$(python scripts/release_notes.py --tag-only)" \\
        --title "..." --notes-file notes.md

Exit status is 0 on success and 1 on any of: unknown tag, missing CHANGELOG,
or a version the CHANGELOG does not cover when --no-fallback is passed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

# ``## [0.1.25] - 2026-11-05`` or ``## [0.1.24] — 2026-10-31``: the repo uses
# both hyphens and em dashes, so neither separator is assumed.
_SECTION_RE = re.compile(r"^## \[(?P<version>[^\]]+)\](?:[^\n]*)$", re.MULTILINE)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def latest_tag() -> str:
    return _git("describe", "--tags", "--abbrev=0")


def version_of(tag: str) -> str:
    """``v0.1.25`` -> ``0.1.25``; a bare version is returned unchanged."""
    return tag[1:] if tag[:1] in ("v", "V") else tag


def changelog_sections(text: str) -> dict[str, str]:
    """Map version -> section body, stopping each body at the next heading."""
    matches = list(_SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group("version").strip()] = text[start:end].strip("\n")
    return sections


def footer(version: str, tag: str) -> str:
    previous = _previous_tag(tag)
    compare = (
        f"https://github.com/Catstate101/superfermion/compare/{previous}...{tag}"
        if previous
        else f"https://github.com/Catstate101/superfermion/releases/tag/{tag}"
    )
    return f"""---

**Install**

```bash
pip install superfermion=={version}
```

One `cp310-abi3` wheel per platform covers Python 3.10-3.15: Linux
(`manylinux_2_28`, glibc 2.28+, and `musllinux_1_2`), macOS (universal2), and
Windows (amd64, arm64). Anything outside that — PyPy, 32-bit, older glibc,
free-threaded builds — compiles from the sdist and needs Rust 1.85+.

**Links** — [Docs](https://superfermion.com/docs) ·
[Changelog](https://github.com/Catstate101/superfermion/blob/main/CHANGELOG.md) ·
[PyPI](https://pypi.org/project/superfermion/{version}/) ·
[Full diff]({compare})
"""


def _previous_tag(tag: str) -> str | None:
    try:
        return _git("describe", "--tags", "--abbrev=0", f"{tag}^")
    except SystemExit:
        return None


def fallback(version: str, tag: str) -> str:
    """Commit list for a version the CHANGELOG predates."""
    previous = _previous_tag(tag)
    span = f"{previous}..{tag}" if previous else tag
    log = _git("log", "--pretty=format:- %s (`%h`)", span)
    scope = f"since `{previous}`" if previous else f"up to `{tag}`"
    return (
        f"_No CHANGELOG entry for {version}: the file's history begins at "
        f"0.1.16. The commits {scope} are listed instead._\n\n{log}"
    )


def build(tag: str, *, allow_fallback: bool) -> str:
    version = version_of(tag)
    if not CHANGELOG.is_file():
        raise SystemExit(f"{CHANGELOG} is missing")
    body = changelog_sections(CHANGELOG.read_text(encoding="utf-8")).get(version)
    if body is None:
        if not allow_fallback:
            raise SystemExit(
                f"CHANGELOG.md has no '## [{version}]' section and "
                "--no-fallback was passed"
            )
        body = fallback(version, tag)
    header = f"## superfermion {version}\n"
    return f"{header}\n{body}\n\n{footer(version, tag)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", nargs="?", help="tag to render (default: latest)")
    parser.add_argument("--out", type=Path, help="write here instead of stdout")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail when the CHANGELOG has no section for this version",
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="print just the tag, for scripting gh release create",
    )
    args = parser.parse_args(argv)

    tag = args.tag or latest_tag()
    if args.tag_only:
        print(tag)
        return 0

    text = build(tag, allow_fallback=not args.no_fallback)
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out} ({len(text)} chars)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
