"""Update-notice behaviour: version ordering, cache TTL, opt-outs, threading.

Every test injects its own fetcher and cache file, so nothing here touches
PyPI.  The package-level guard (`disabled()` is True under pytest) is asserted
rather than bypassed, which is what keeps the suite offline.
"""

import json
import sys
import threading
import time

import pytest

import superfermion as sf
from superfermion import update_check as uc


def cache_in(tmp_path):
    return tmp_path / "update_check.json"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.1.20", (0, 1, 20)),
        ("1.0.0", (1, 0, 0)),
        ("0.1.20rc1", (0, 1, 20)),
        ("0.1.20+local", (0, 1, 20)),
        ("10.2.3", (10, 2, 3)),
    ],
)
def test_parse_version_reads_release_segments(text, expected):
    assert uc.parse_version(text) == expected


@pytest.mark.parametrize("text", ["", None, "dev", "0..1", ".1", "v0.1"])
def test_parse_version_rejects_junk(text):
    assert uc.parse_version(text) == ()


@pytest.mark.parametrize(
    "candidate, current, newer",
    [
        ("0.1.20", "0.1.19", True),
        ("0.2.0", "0.1.99", True),
        ("1.0.0", "0.9.9", True),
        ("0.1.19", "0.1.19", False),
        ("0.1.18", "0.1.19", False),
        ("0.1", "0.1.0", False),
        ("0.10.0", "0.9.0", True),
        ("junk", "0.1.19", False),
    ],
)
def test_is_newer(candidate, current, newer):
    assert uc.is_newer(candidate, current) is newer


def test_check_once_reports_newer_release(tmp_path):
    message = uc.check_once(
        "0.1.19", now=1_000.0, fetch=lambda: "0.1.20", cache_file=cache_in(tmp_path)
    )
    assert message is not None
    assert "0.1.19" in message and "0.1.20" in message
    assert "pip install -U superfermion" in message


def test_check_once_silent_when_up_to_date(tmp_path):
    assert (
        uc.check_once("0.1.20", now=1_000.0, fetch=lambda: "0.1.20", cache_file=cache_in(tmp_path))
        is None
    )


def test_check_once_silent_when_local_is_ahead(tmp_path):
    assert (
        uc.check_once("0.1.21", now=1_000.0, fetch=lambda: "0.1.20", cache_file=cache_in(tmp_path))
        is None
    )


def test_check_once_swallows_fetch_errors(tmp_path):
    def boom():
        raise RuntimeError("network down")

    assert uc.check_once("0.1.19", now=1_000.0, fetch=boom, cache_file=cache_in(tmp_path)) is None


def test_check_once_swallows_unparseable_answers(tmp_path):
    assert uc.check_once("0.1.19", now=1_000.0, fetch=lambda: "", cache_file=cache_in(tmp_path)) is None


def test_cache_prevents_a_second_fetch_within_a_day(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return "0.1.20"

    path = cache_in(tmp_path)
    uc.check_once("0.1.19", now=1_000.0, fetch=fetch, cache_file=path)
    uc.check_once("0.1.19", now=1_000.0 + 3_600, fetch=fetch, cache_file=path)
    assert len(calls) == 1


def test_cache_expires_after_a_day(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return "0.1.20"

    path = cache_in(tmp_path)
    uc.check_once("0.1.19", now=1_000.0, fetch=fetch, cache_file=path)
    uc.check_once("0.1.19", now=1_000.0 + 24 * 3600 + 1, fetch=fetch, cache_file=path)
    assert len(calls) == 2


def test_cache_file_records_the_answer(tmp_path):
    path = cache_in(tmp_path)
    uc.check_once("0.1.19", now=1_000.0, fetch=lambda: "0.1.20", cache_file=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["latest"] == "0.1.20"
    assert payload["checked_at"] == 1_000.0


def test_cached_newer_release_still_notifies(tmp_path):
    path = cache_in(tmp_path)
    uc.check_once("0.1.19", now=1_000.0, fetch=lambda: "0.1.20", cache_file=path)

    def must_not_run():
        raise AssertionError("a fresh cache must not trigger a fetch")

    assert uc.check_once("0.1.19", now=1_000.0 + 60, fetch=must_not_run, cache_file=path) is not None


def test_corrupt_cache_is_ignored_and_rewritten(tmp_path):
    path = cache_in(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    message = uc.check_once("0.1.19", now=1_000.0, fetch=lambda: "0.1.20", cache_file=path)
    assert message is not None
    assert json.loads(path.read_text(encoding="utf-8"))["latest"] == "0.1.20"


@pytest.mark.parametrize(
    "name, value",
    [
        ("SF_NO_UPDATE_CHECK", "1"),
        ("SF_NO_UPDATE_CHECK", ""),
        ("SF_OFFLINE", "1"),
        ("SF_OFFLINE", "true"),
        ("NO_UPDATE_NOTIFIER", "true"),
        ("CI", "true"),
        ("CI", "1"),
    ],
)
def test_disabled_by_environment(monkeypatch, name, value):
    for other in ("SF_NO_UPDATE_CHECK", "SF_OFFLINE", "NO_UPDATE_NOTIFIER", "CI"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, value)
    assert uc.disabled() is True


def test_disabled_under_pytest_even_with_ci_false(monkeypatch):
    for other in ("SF_NO_UPDATE_CHECK", "SF_OFFLINE", "NO_UPDATE_NOTIFIER"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv("CI", "false")
    # The pytest guard is what keeps every local test run offline.
    assert uc.disabled() is True


def test_ci_false_and_no_pytest_does_not_disable(monkeypatch):
    for other in ("SF_NO_UPDATE_CHECK", "SF_OFFLINE", "NO_UPDATE_NOTIFIER", "CI"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv("CI", "false")
    monkeypatch.delitem(sys.modules, "pytest")
    assert uc.disabled() is False


def test_cache_path_honours_override(monkeypatch, tmp_path):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("SF_UPDATE_CHECK_CACHE", str(target))
    assert uc.cache_path() == target


def test_notify_returns_before_the_check_finishes(monkeypatch):
    done = threading.Event()

    def slow_check(current):
        time.sleep(0.4)
        done.set()
        return None

    monkeypatch.setattr(uc, "_thread_started", False)
    monkeypatch.setattr(uc, "disabled", lambda: False)
    monkeypatch.setattr(uc, "check_once", slow_check)

    started = time.perf_counter()
    uc.notify_if_outdated("0.1.19")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"import-time call blocked for {elapsed:.3f}s"
    assert done.wait(3.0)


def test_notify_starts_only_one_thread(monkeypatch):
    monkeypatch.setattr(uc, "_thread_started", False)
    monkeypatch.setattr(uc, "disabled", lambda: False)
    monkeypatch.setattr(uc, "check_once", lambda current: None)
    uc.notify_if_outdated("0.1.19")
    first = uc._thread_started
    uc.notify_if_outdated("0.1.19")
    assert first is True
    assert uc._thread_started is True


def test_notify_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(uc, "_thread_started", False)
    monkeypatch.setattr(uc, "disabled", lambda: True)
    uc.notify_if_outdated("0.1.19")
    assert uc._thread_started is False


def test_emit_writes_to_stderr_only_once(monkeypatch, capsys):
    monkeypatch.setattr(uc, "_notified", False)
    uc._emit("[superfermion] test notice")
    uc._emit("[superfermion] test notice")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("[superfermion] test notice") == 1


def test_installed_version_is_parseable():
    assert uc.parse_version(sf.__version__) != ()
