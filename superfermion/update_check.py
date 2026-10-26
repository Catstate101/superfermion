"""One-line notice when a newer superfermion is on PyPI.

Design constraints, all of them load-bearing:

* import must never wait on the network, so the request runs in a daemon
  thread with a short timeout and the caller returns immediately;
* it runs at most once a day, because the answer is cached on disk;
* it is silent when offline, inside CI, under pytest, or when the user opts
  out with ``SF_NO_UPDATE_CHECK=1`` (also honouring ``NO_UPDATE_NOTIFIER``
  and ``SF_OFFLINE``);
* it never raises, never writes to stdout, and never changes the process
  exit status.

The notice only goes to stderr and only mentions the upgrade command, so a
program's own output stays untouched.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time

_PYPI_URL = "https://pypi.org/pypi/superfermion/json"
_TTL_SECONDS = 24 * 60 * 60
_TIMEOUT_SECONDS = 1.5
_USER_AGENT = "superfermion-update-check"

_lock = threading.Lock()
_thread_started = False
_notified = False

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def disabled() -> bool:
    """True when this process must not check or print anything."""
    if os.environ.get("SF_NO_UPDATE_CHECK") is not None:
        return True
    for name in ("SF_OFFLINE", "NO_UPDATE_NOTIFIER"):
        if _flag(name):
            return True
    ci = os.environ.get("CI", "").strip().lower()
    if ci and ci not in {"0", "false", "no", "off"}:
        return True
    # A network call during collection would be a surprise in every test run.
    if "pytest" in sys.modules:
        return True
    return False


def cache_path() -> pathlib.Path:
    override = os.environ.get("SF_UPDATE_CHECK_CACHE")
    if override:
        return pathlib.Path(override)
    base = (
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("XDG_CACHE_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache")
    )
    return pathlib.Path(base) / "superfermion" / "update_check.json"


def parse_version(text: str) -> tuple[int, ...]:
    """'0.1.20' -> (0, 1, 20).  Anything unparseable -> ().

    Only numeric release segments are compared; a suffix such as 'rc1' or
    '+local' ends the segment and the version is treated as that release.
    """
    text = (text or "").strip()
    if not text:
        return ()
    parts: list[int] = []
    for chunk in text.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return ()
        parts.append(int(digits))
    return tuple(parts)


def is_newer(candidate: str, current: str) -> bool:
    """True only when both versions parse and candidate > current."""
    left, right = parse_version(candidate), parse_version(current)
    if not left or not right:
        return False
    width = max(len(left), len(right))
    left = left + (0,) * (width - len(left))
    right = right + (0,) * (width - len(right))
    return left > right


def fetch_latest(timeout: float = _TIMEOUT_SECONDS) -> str | None:
    """Latest version on PyPI, or None.  Swallows every failure on purpose.

    ``urllib.request`` is imported here, not at module scope: its import chain
    (http, ssl, email) is the most expensive part of this module, and doing it
    inside the worker keeps it off the import path of the package.
    """
    try:
        import urllib.request

        request = urllib.request.Request(
            _PYPI_URL,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        version = str(payload.get("info", {}).get("version") or "")
        return version if parse_version(version) else None
    except Exception:
        return None


def _read_cache(path: pathlib.Path, now: float) -> tuple[bool, str | None]:
    """(is_fresh, latest).  is_fresh with latest=None means 'PyPI had no answer'."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        checked_at = float(data["checked_at"])
        if now - checked_at < _TTL_SECONDS:
            return True, (str(data["latest"]) if data.get("latest") else None)
    except Exception:
        pass
    return False, None


def _write_cache(path: pathlib.Path, now: float, latest: str | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_suffix(".tmp")
        scratch.write_text(
            json.dumps({"checked_at": now, "latest": latest or ""}),
            encoding="utf-8",
        )
        os.replace(scratch, path)
    except Exception:
        pass


def check_once(
    current: str,
    *,
    now: float | None = None,
    fetch=None,
    cache_file: pathlib.Path | str | None = None,
) -> str | None:
    """Return the notice text when a newer release exists, else None.

    Pure enough to test: pass ``now``, ``fetch`` and ``cache_file`` to keep it
    off the network and out of the real cache.
    """
    moment = time.time() if now is None else float(now)
    path = pathlib.Path(cache_file) if cache_file is not None else cache_path()
    fresh, latest = _read_cache(path, moment)
    if not fresh:
        try:
            latest = (fetch or fetch_latest)()
        except Exception:
            latest = None
        _write_cache(path, moment, latest)
    if not latest or not is_newer(latest, current):
        return None
    return (
        f"[superfermion] {current} is installed; {latest} is available "
        f"(pip install -U superfermion)"
    )


def _emit(message: str) -> None:
    global _notified
    with _lock:
        if _notified:
            return
        _notified = True
    try:
        stream = sys.stderr
        if stream is not None:
            stream.write(message + "\n")
            stream.flush()
    except Exception:
        pass


def notify_if_outdated(current: str) -> None:
    """Start the one-off background check.  Safe to call on every import."""
    global _thread_started
    if disabled():
        return
    with _lock:
        if _thread_started:
            return
        _thread_started = True
    thread = threading.Thread(
        target=_check_and_emit,
        args=(current,),
        name="sf-update-check",
        daemon=True,
    )
    thread.start()


def _check_and_emit(current: str) -> None:
    try:
        message = check_once(current)
    except Exception:
        return
    if message:
        _emit(message)
