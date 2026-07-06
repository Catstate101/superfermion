"""
Cross-platform encoding utilities.

On Windows, Python defaults to cp1252 for stdout/stderr, causing
UnicodeEncodeError when scripts print Unicode characters (∇≈✓✗Δ●✅❌).
This module provides `ensure_utf8()` to reconfigure stdout/stderr to UTF-8.
"""

import sys
import io
import os


def ensure_utf8() -> None:
    """Reconfigure stdout and stderr to use UTF-8 encoding.

    Safe to call on all platforms — on Linux/macOS this is a no-op
    since UTF-8 is already the default. On Windows, this prevents
    cp1252 encoding crashes when printing Unicode symbols.

    Usage:
        from superfermion.utils.encoding import ensure_utf8
        ensure_utf8()
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            elif hasattr(stream, "buffer"):
                # Python < 3.7 fallback
                stream = io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=stream.line_buffering,
                )
        except Exception:
            # If anything fails (e.g. redirected/closed stream), silently ignore
            pass
