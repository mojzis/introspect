"""Unobtrusive "a newer release exists" nag for the CLI.

Introspect is distributed via ``uvx introspy`` and ``uv tool install introspy``,
neither of which auto-updates, so users silently run stale versions. This module
adds an opt-out, privacy-respecting check that prints a single stderr line when a
newer PyPI release is available.

Design constraints (see the task spec):

- **Never blocks.** The current invocation only *reads* a local cache to decide
  whether to nag. Refreshing that cache happens in a short-lived background
  thread and benefits the *next* run.
- **stderr only.** ``mcp`` (stdio) uses stdout as its protocol channel and
  ``query`` output is meant to be pipeable, so the nag must never touch stdout.
- **Honest & minimal network.** The refresh does a single plain GET to the
  public PyPI JSON endpoint — no query params, no identifiers, no telemetry.
- **Leaf module.** Imports nothing from ``api`` or ``mcp``.
"""

import functools
import importlib.metadata
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

from introspect.db import DEFAULT_DB_PATH

# Distribution name on PyPI (see ``[project].name`` in pyproject.toml).
DIST_NAME = "introspy"
PYPI_URL = f"https://pypi.org/pypi/{DIST_NAME}/json"

# Commands where a stderr line is harmless. ``mcp`` is deliberately absent: its
# stdio transport must never see extra output. Anything not listed is suppressed.
ELIGIBLE_COMMANDS = frozenset(
    {
        "serve",
        "devserve",
        "sessions",
        "tools",
        "stats",
        "search",
        "query",
        "raw",
        "tables",
        "materialize",
        "refresh",
        "claude",
    }
)

# Opt-out and (undocumented) interval override, matching the INTROSPECT_* convention.
ENV_ENABLED = "INTROSPECT_VERSION_CHECK"
ENV_INTERVAL = "INTROSPECT_VERSION_CHECK_INTERVAL"

DEFAULT_INTERVAL_SECONDS = 24 * 60 * 60
FETCH_TIMEOUT_SECONDS = 2.0
CACHE_SCHEMA_VERSION = 1
_HTTP_OK = 200

# Environment values that read as "off" for the opt-out flag and as "not CI" for
# the CI markers.
_FALSEY = frozenset({"", "0", "false", "no", "off"})

# Common CI markers; presence of any (with a truthy-ish value) silences the nag
# so pipelines stay quiet.
_CI_ENV_VARS = (
    "CI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
)


@dataclass(frozen=True)
class VersionCache:
    """The tiny, versioned on-disk cache. ``latest`` is ``None`` when the last
    refresh failed (a "checked, unknown" stamp so we back off and don't hammer
    PyPI on every run)."""

    checked_at: float
    latest: str | None


# --- environment gating ------------------------------------------------------


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name)
    return val is not None and val.strip().lower() not in _FALSEY


def _version_check_enabled() -> bool:
    """``INTROSPECT_VERSION_CHECK`` defaults to on; ``off`` (and similar) disables."""
    val = os.environ.get(ENV_ENABLED)
    if val is None:
        return True
    return val.strip().lower() not in _FALSEY


def _is_ci() -> bool:
    return any(_env_truthy(name) for name in _CI_ENV_VARS)


def _passes_gate(command: str | None, *, is_tty: bool) -> bool:
    """Cheap short-circuit checks that must all pass before we touch the cache,
    the version metadata, or the network."""
    if command not in ELIGIBLE_COMMANDS:
        return False
    if not _version_check_enabled():
        return False
    if _is_ci():
        return False
    return is_tty


# --- current version ---------------------------------------------------------


def _current_version() -> str | None:
    try:
        return importlib.metadata.version(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_editable_install() -> bool:
    """True for editable/`pip install -e` working copies (per PEP 610), so we
    don't nag developers about their own checkout."""
    try:
        raw = importlib.metadata.distribution(DIST_NAME).read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, OSError):
        return False
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False
    return bool(data.get("dir_info", {}).get("editable"))


def _is_dev_build(current: str) -> bool:
    """Suppress the nag for dev/local/editable builds and unparseable versions."""
    if _is_editable_install():
        return True
    try:
        parsed = Version(current)
    except InvalidVersion:
        return True
    return parsed.is_devrelease or parsed.local is not None


def _is_newer(latest: str, current: str) -> bool:
    """True only when ``latest`` is a strictly greater *stable* release than
    ``current``. Stable users are never nagged onto a pre-release."""
    try:
        latest_v = Version(latest)
        current_v = Version(current)
    except InvalidVersion:
        return False
    if latest_v.is_prerelease and not current_v.is_prerelease:
        return False
    return latest_v > current_v


# --- cache -------------------------------------------------------------------


def _introspect_home() -> Path:
    """Directory that holds the cache — the parent of the DuckDB path, honouring
    the same ``INTROSPECT_DB_PATH`` override so we never hardcode ``~``."""
    override = os.environ.get("INTROSPECT_DB_PATH")
    if override:
        return Path(override).expanduser().parent
    return DEFAULT_DB_PATH.parent


def _cache_path() -> Path:
    return _introspect_home() / "version_check.json"


def _read_cache(path: Path) -> VersionCache | None:
    """Load the cache, returning ``None`` for missing/corrupt/wrong-schema files.
    A format change is a clean reset, never a crash."""
    try:
        raw = path.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA_VERSION:
        return None
    checked_at = data.get("checked_at")
    latest = data.get("latest")
    if not isinstance(checked_at, int | float) or isinstance(checked_at, bool):
        return None
    if latest is not None and not isinstance(latest, str):
        return None
    return VersionCache(checked_at=float(checked_at), latest=latest)


def _write_cache(path: Path, cache: VersionCache) -> None:
    """Atomically write the cache; swallow any filesystem error."""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "checked_at": cache.checked_at,
        "latest": cache.latest,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except OSError:
        pass


# --- refresh (out-of-band) ---------------------------------------------------


def _fetch_latest_version(timeout: float = FETCH_TIMEOUT_SECONDS) -> str | None:
    """Single GET to the PyPI JSON endpoint. Any failure — network, timeout,
    non-200, unparseable body — returns ``None`` rather than raising.

    Sends no query params and no identifying data; a static User-Agent only.
    """
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(  # noqa: S310 - constant https URL
        PYPI_URL,
        headers={"Accept": "application/json", "User-Agent": "introspy-update-check"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if resp.status != _HTTP_OK:
                return None
            data = json.loads(resp.read())
    except Exception:  # any failure — network, timeout, parse — is swallowed
        return None
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    if not isinstance(info, dict):
        return None
    version = info.get("version")
    return version if isinstance(version, str) else None


def _refresh_cache_now(path: Path, now: float) -> None:
    """Synchronous refresh body: fetch, then stamp the cache with the result
    (``latest=None`` on failure). Always stamps ``checked_at`` so a failed fetch
    still backs us off for the interval."""
    latest = _fetch_latest_version()
    _write_cache(path, VersionCache(checked_at=now, latest=latest))


def _trigger_refresh(path: Path, now: float) -> None:
    """Kick off the refresh in a daemon thread and return immediately. The main
    process never joins it, so a slow/hung PyPI can't delay exit; if the
    interpreter tears down first the cache simply refreshes next time."""
    thread = threading.Thread(
        target=_refresh_cache_now,
        args=(path, now),
        name="introspy-version-check",
        daemon=True,
    )
    thread.start()


# --- nag ---------------------------------------------------------------------


def _format_nag(current: str, latest: str) -> str:
    """One line, plain text (no color-dependent meaning), both upgrade paths."""
    return (
        f"introspy {latest} is available (you have {current}) — "
        f"run: uvx introspy@latest  |  uv tool upgrade introspy"
    )


def _stderr_is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _emit_nag(line: str) -> None:
    print(line, file=sys.stderr)  # noqa: T201 - user-facing nag, stderr only


def _evaluate_nag(command: str | None, *, now: float, is_tty: bool) -> str | None:
    """Pure decision core: return the nag line if this run should nag, else
    ``None``. Triggers an out-of-band refresh when the cache is stale/missing.

    Injecting ``now`` and ``is_tty`` keeps this deterministic and offline-testable.
    """
    if not _passes_gate(command, is_tty=is_tty):
        return None
    current = _current_version()
    if current is None or _is_dev_build(current):
        return None
    path = _cache_path()
    cache = _read_cache(path)
    if cache is None or (now - cache.checked_at) >= _check_interval_seconds():
        # Stale or missing: refresh for next time, stay silent this run. On fast
        # commands the interpreter often exits before the daemon fetch lands, so
        # the cache is typically populated by a longer-running command (e.g.
        # ``serve``) or an eventual run that outlives the ~2s GET — by design, we
        # never block this invocation on the network to make it land sooner.
        _trigger_refresh(path, now)
        return None
    if cache.latest and _is_newer(cache.latest, current):
        return _format_nag(current, cache.latest)
    return None


def _check_interval_seconds() -> float:
    raw = os.environ.get(ENV_INTERVAL)
    if raw is None:
        return DEFAULT_INTERVAL_SECONDS
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_INTERVAL_SECONDS
    return val if val > 0 else DEFAULT_INTERVAL_SECONDS


def maybe_notify_update(command: str | None, defer) -> None:
    """CLI entry point. Decide whether to nag; if so, register the one-line
    stderr print via ``defer`` (e.g. Typer ``Context.call_on_close``) so it
    lands *after* the command's real output instead of mid-render.

    Safe to call unconditionally from the shared Typer callback: the ``mcp``,
    non-TTY, CI, and opt-out paths all short-circuit before anything prints or
    hits the network.
    """
    line = _evaluate_nag(command, now=time.time(), is_tty=_stderr_is_tty())
    if line is not None:
        defer(functools.partial(_emit_nag, line))
