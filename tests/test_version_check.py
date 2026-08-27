"""Tests for the PyPI update-nag (``introspect.version_check``).

All network and clock access is injected or monkeypatched — no test ever
touches PyPI or the real wall clock.
"""

import json

import pytest

from introspect import version_check as vc

# A "now" far from the epoch so freshness math is unambiguous.
NOW = 1_700_000_000.0


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Neutral environment: no CI markers, opt-out unset, cache in ``tmp_path``,
    and a non-editable stable install by default. Individual tests override."""
    for name in (*vc._CI_ENV_VARS, vc.ENV_ENABLED, vc.ENV_INTERVAL):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INTROSPECT_DB_PATH", str(tmp_path / "introspect.duckdb"))
    monkeypatch.setattr(vc, "_current_version", lambda: "1.0.0")
    monkeypatch.setattr(vc, "_is_editable_install", lambda: False)
    return tmp_path


def _write_cache(tmp_path, *, checked_at, latest):
    vc._write_cache(
        tmp_path / "version_check.json",
        vc.VersionCache(checked_at=checked_at, latest=latest),
    )


# --- happy path: nag when behind --------------------------------------------


def test_nags_when_behind_with_fresh_cache(clean_env):
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    line = vc._evaluate_nag("stats", now=NOW, is_tty=True)
    assert line is not None
    assert "1.2.0" in line
    assert "1.0.0" in line
    assert "uvx introspy@latest" in line
    assert "uv tool upgrade introspy" in line


def test_no_nag_when_up_to_date(clean_env):
    _write_cache(clean_env, checked_at=NOW, latest="1.0.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


def test_no_nag_when_current_newer_than_latest(clean_env, monkeypatch):
    monkeypatch.setattr(vc, "_current_version", lambda: "2.0.0")
    _write_cache(clean_env, checked_at=NOW, latest="1.9.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


# --- suppression gates -------------------------------------------------------


def test_no_nag_for_mcp_command(clean_env, monkeypatch):
    triggered = []
    monkeypatch.setattr(vc, "_trigger_refresh", lambda *a: triggered.append(a))
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("mcp", now=NOW, is_tty=True) is None
    # mcp must not even reach the cache/refresh machinery.
    assert triggered == []


def test_no_nag_when_stderr_not_tty(clean_env):
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=False) is None


def test_no_nag_when_ci_env_set(clean_env, monkeypatch):
    monkeypatch.setenv("CI", "true")
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


def test_no_nag_when_opt_out(clean_env, monkeypatch):
    monkeypatch.setenv(vc.ENV_ENABLED, "off")
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


def test_ci_false_value_does_not_suppress(clean_env, monkeypatch):
    monkeypatch.setenv("CI", "false")
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is not None


def test_no_nag_for_dev_or_editable_build(clean_env, monkeypatch):
    monkeypatch.setattr(vc, "_is_editable_install", lambda: True)
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


def test_no_nag_for_local_version_build(clean_env, monkeypatch):
    monkeypatch.setattr(vc, "_current_version", lambda: "1.0.0+local")
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


def test_no_nag_when_version_unknown(clean_env, monkeypatch):
    monkeypatch.setattr(vc, "_current_version", lambda: None)
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


# --- cache freshness / refresh ----------------------------------------------


def test_first_run_missing_cache_is_silent_and_refreshes(clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(vc, "_trigger_refresh", lambda path, now: calls.append(now))
    # No cache file written.
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None
    assert calls == [NOW]


def test_stale_cache_refreshes_but_does_not_nag(clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(vc, "_trigger_refresh", lambda path, now: calls.append(now))
    stale = NOW - vc.DEFAULT_INTERVAL_SECONDS - 1
    _write_cache(clean_env, checked_at=stale, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None
    assert calls == [NOW]


def test_fresh_cache_does_not_refresh(clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(vc, "_trigger_refresh", lambda path, now: calls.append(now))
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    vc._evaluate_nag("stats", now=NOW, is_tty=True)
    assert calls == []


def test_interval_override_env(clean_env, monkeypatch):
    monkeypatch.setenv(vc.ENV_INTERVAL, "10")
    calls = []
    monkeypatch.setattr(vc, "_trigger_refresh", lambda path, now: calls.append(now))
    # 11s old: stale under a 10s interval, fresh under the default.
    _write_cache(clean_env, checked_at=NOW - 11, latest="1.2.0")
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None
    assert calls == [NOW]


# --- cache read robustness ---------------------------------------------------


def test_read_cache_absent(tmp_path):
    assert vc._read_cache(tmp_path / "nope.json") is None


def test_read_cache_corrupt(tmp_path):
    path = tmp_path / "version_check.json"
    path.write_text("{ this is not json")
    assert vc._read_cache(path) is None


def test_read_cache_wrong_schema(tmp_path):
    path = tmp_path / "version_check.json"
    path.write_text(json.dumps({"schema": 999, "checked_at": NOW, "latest": "9.9.9"}))
    assert vc._read_cache(path) is None


def test_corrupt_cache_does_not_raise_in_evaluate(clean_env):
    (clean_env / "version_check.json").write_text("garbage")
    # Corrupt cache is treated as missing → silent refresh, no exception.
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "version_check.json"
    vc._write_cache(path, vc.VersionCache(checked_at=NOW, latest="3.1.4"))
    cache = vc._read_cache(path)
    assert cache == vc.VersionCache(checked_at=NOW, latest="3.1.4")


# --- version comparison ------------------------------------------------------


def test_is_newer_uses_semantic_ordering():
    # Lexically "0.9.0" > "0.10.0"; semantically it is not.
    assert vc._is_newer("0.10.0", "0.9.0") is True
    assert vc._is_newer("0.9.0", "0.10.0") is False


def test_is_newer_equal_is_false():
    assert vc._is_newer("1.2.3", "1.2.3") is False


def test_is_newer_ignores_prerelease_for_stable_user():
    assert vc._is_newer("2.0.0rc1", "1.9.0") is False


def test_is_newer_invalid_version_is_false():
    assert vc._is_newer("not-a-version", "1.0.0") is False


# --- fetch / refresh network handling ---------------------------------------


def test_fetch_latest_success(monkeypatch):
    class _Resp:
        status = 200

        def read(self):
            return json.dumps({"info": {"version": "4.5.6"}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    assert vc._fetch_latest_version() == "4.5.6"


def test_fetch_latest_non_200(monkeypatch):
    class _Resp:
        status = 503

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    assert vc._fetch_latest_version() is None


def test_fetch_latest_network_error(monkeypatch):
    def _boom(*a, **k):
        raise OSError

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert vc._fetch_latest_version() is None


def test_fetch_latest_unparseable_body(monkeypatch):
    class _Resp:
        status = 200

        def read(self):
            return b"<html>not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    assert vc._fetch_latest_version() is None


def test_refresh_stamps_unknown_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_fetch_latest_version", lambda: None)
    path = tmp_path / "version_check.json"
    vc._refresh_cache_now(path, NOW)
    cache = vc._read_cache(path)
    assert cache == vc.VersionCache(checked_at=NOW, latest=None)


def test_refresh_stores_latest_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "_fetch_latest_version", lambda: "5.0.0")
    path = tmp_path / "version_check.json"
    vc._refresh_cache_now(path, NOW)
    assert vc._read_cache(path) == vc.VersionCache(checked_at=NOW, latest="5.0.0")


def test_unknown_latest_cache_does_not_nag(clean_env):
    # A "checked, unknown" stamp is fresh but has no version to compare.
    _write_cache(clean_env, checked_at=NOW, latest=None)
    assert vc._evaluate_nag("stats", now=NOW, is_tty=True) is None


# --- deferred emission via maybe_notify_update ------------------------------


def test_maybe_notify_defers_emit_when_behind(clean_env, monkeypatch, capsys):
    _write_cache(clean_env, checked_at=NOW, latest="1.2.0")
    monkeypatch.setattr(vc, "stderr_is_tty", lambda: True)
    monkeypatch.setattr(vc.time, "time", lambda: NOW)
    deferred = []
    vc.maybe_notify_update("stats", deferred.append)
    assert len(deferred) == 1
    # Nothing printed until the deferred callback is invoked (at command close).
    assert capsys.readouterr().err == ""
    deferred[0]()
    captured = capsys.readouterr()
    assert "1.2.0 is available" in captured.err
    assert captured.out == ""


def test_maybe_notify_no_defer_when_up_to_date(clean_env, monkeypatch):
    _write_cache(clean_env, checked_at=NOW, latest="1.0.0")
    monkeypatch.setattr(vc, "stderr_is_tty", lambda: True)
    monkeypatch.setattr(vc.time, "time", lambda: NOW)
    deferred = []
    vc.maybe_notify_update("stats", deferred.append)
    assert deferred == []
