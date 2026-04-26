"""Background refresh loop for keeping the materialized DB in sync with JSONL files."""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import duckdb

from introspect.db import materialize_views
from introspect.search import build_search_corpus

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)


# Allowed tokens for the refresh-window picker. Owned by this module because
# both the background loop and the HTTP handler need them; placing them here
# keeps the dependency direction (handler -> refresh) one-way.
VALID_WINDOWS = frozenset({"1", "7", "30", "month"})
DEFAULT_WINDOW = "30"
# Days returned for fixed-length tokens. ``"month"`` is computed at call time.
_FIXED_WINDOW_DAYS: dict[str, int] = {"1": 1, "7": 7, "30": 30}


def window_to_days(window: str) -> int:
    """Convert a window token to a positive ``days`` value for ``materialize_views``.

    * ``"1"`` / ``"7"`` / ``"30"`` -> the literal int.
    * ``"month"`` -> days since the first of the current UTC calendar month
      (inclusive of today). On the 1st returns ``1``. UTC matches the
      timezone used by ``materialize_views``' day filter.
    * Anything else -> the ``DEFAULT_WINDOW`` days value (defensive; the
      handler and lifespan both pre-validate input).
    """
    if window in _FIXED_WINDOW_DAYS:
        return _FIXED_WINDOW_DAYS[window]
    if window == "month":
        today = datetime.now(UTC).date()
        return (today - today.replace(day=1)).days + 1
    return _FIXED_WINDOW_DAYS[DEFAULT_WINDOW]


class RefreshState(Protocol):
    """Contract between :func:`refresh_loop` (writer) and :func:`wait_for_refresh`
    (reader). FastAPI's ``app.state`` satisfies this after :mod:`api.main` sets
    the attributes during startup.
    """

    refresh_trigger: asyncio.Event | None
    refresh_in_progress: bool
    refresh_started_at: datetime | None
    last_refreshed_at: datetime | None
    refresh_window: str
    last_built_days: int


class RefreshOutcome(Enum):
    """Result classes for :func:`wait_for_refresh`."""

    DISABLED = "disabled"  # No trigger configured (auto-refresh off).
    UNCHANGED = "unchanged"  # Loop woke but JSONL files were unchanged.
    COMPLETED = "completed"  # Rebuild finished within the wait budget.
    STILL_RUNNING = "still_running"  # Started but did not finish in time.


@dataclass(frozen=True)
class RefreshResult:
    outcome: RefreshOutcome
    last_refreshed_at: datetime | None


# Internal poll cadence — kept private because callers tune *budgets*, not
# step granularity. Two phases: a brief one to detect that the loop picked up
# the trigger, then a longer one to wait for completion.
_START_TIMEOUT = 0.5
_START_STEP = 0.05
_FINISH_STEP = 0.1


async def wait_for_refresh(
    state: RefreshState,
    *,
    finish_timeout: float = 3.0,
) -> RefreshResult:
    """Set the refresh trigger and wait until the background loop finishes.

    Returns one of four outcomes based on what the loop did:

    * ``DISABLED`` — no trigger on ``state`` (auto-refresh is off).
    * ``UNCHANGED`` — loop woke but JSONL mtimes were unchanged; no rebuild ran.
    * ``COMPLETED`` — rebuild finished within ``finish_timeout``.
    * ``STILL_RUNNING`` — rebuild started but did not finish in time.
    """
    if state.refresh_trigger is None:
        return RefreshResult(RefreshOutcome.DISABLED, state.last_refreshed_at)

    last_before = state.last_refreshed_at
    state.refresh_trigger.set()

    waited = 0.0
    while waited < _START_TIMEOUT:
        if state.refresh_in_progress:
            break
        await asyncio.sleep(_START_STEP)
        waited += _START_STEP

    if state.refresh_in_progress:
        waited = 0.0
        while waited < finish_timeout:
            if not state.refresh_in_progress:
                break
            await asyncio.sleep(_FINISH_STEP)
            waited += _FINISH_STEP

    last_after = state.last_refreshed_at
    if last_after != last_before and last_after is not None:
        return RefreshResult(RefreshOutcome.COMPLETED, last_after)
    if state.refresh_in_progress:
        return RefreshResult(RefreshOutcome.STILL_RUNNING, last_after)
    return RefreshResult(RefreshOutcome.UNCHANGED, last_after)


def newest_mtime(jsonl_glob: str) -> float:
    """Return the newest mtime among files matching ``jsonl_glob``.

    Returns ``0.0`` if nothing matches. Defensively skips files that disappear
    between ``glob`` and ``os.path.getmtime``.
    """
    paths = glob.glob(jsonl_glob, recursive=True)  # noqa: PTH207
    latest = 0.0
    for p in paths:
        try:
            mtime = os.path.getmtime(p)  # noqa: PTH204
        except FileNotFoundError:
            continue
        latest = max(latest, mtime)
    return latest


def _rebuild_sidecar(
    sidecar: Path,
    jsonl_glob: str,
    days: int,
    resolve_projects: bool,
) -> None:
    """Rebuild the materialized DB into a fresh sidecar file."""
    with contextlib.suppress(FileNotFoundError):
        sidecar.unlink()
    conn = duckdb.connect(str(sidecar))
    try:
        materialize_views(conn, jsonl_glob, days, resolve_projects=resolve_projects)
        build_search_corpus(conn)
    finally:
        conn.close()


def _swap_in(db_path: Path, sidecar: Path) -> None:
    """Atomically rename ``sidecar`` over ``db_path``.

    Per-request connections are opened directly from ``db_path`` in the
    middleware, so this is now just an atomic file swap. In-flight cursors
    keep reading from the old inode (which lingers until they close), and
    new connections after the swap see the fresh data.
    """
    os.replace(str(sidecar), str(db_path))  # noqa: PTH105


def _compute_days(state: RefreshState, default: int) -> int:
    """Resolve the days-window value from ``state.refresh_window``.

    Falls back to ``default`` only when the state attribute is missing
    entirely. An invalid token is delegated to :func:`window_to_days`, which
    returns the ``DEFAULT_WINDOW`` days value — matching the lifespan's
    invalid-env fallback so the two code paths agree.
    """
    window = getattr(state, "refresh_window", None)
    if not isinstance(window, str):
        return default
    return window_to_days(window)


def _window_changed(state: RefreshState, current_days: int) -> bool:
    """Has the window changed since the last successful rebuild?

    Used by :func:`refresh_loop` to force a rebuild when the user picks a new
    window even though JSONL mtimes are unchanged.
    """
    last = getattr(state, "last_built_days", None)
    return last != current_days


async def refresh_loop(  # noqa: PLR0913
    app: FastAPI,
    db_path: Path,
    jsonl_glob: str,
    days: int,
    resolve_projects: bool,
    interval_seconds: float,
    trigger: asyncio.Event,
) -> None:
    """Poll JSONL mtime and rebuild the materialized DB when files change.

    The loop sleeps up to ``interval_seconds`` between ticks, but wakes early
    when ``trigger`` is set (e.g. a manual "Refresh now" click). The mtime
    short-circuit still gates the rebuild — a manual wake on an unchanged
    filesystem is a fast no-op unless the user picked a new window.

    The ``days`` parameter is the initial default; each rebuild re-reads
    ``app.state.refresh_window`` so the picker's choice is honoured by both
    manual refreshes and idle ticks.
    """
    sidecar = db_path.with_name(db_path.name + ".next")
    last_mtime = newest_mtime(jsonl_glob)
    while True:
        try:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(trigger.wait(), timeout=interval_seconds)
            trigger.clear()
            current = newest_mtime(jsonl_glob)
            current_days = _compute_days(app.state, days)
            # Skip when nothing changed AND the window matches the last build.
            # A manual wake with a new window forces a rebuild even on an
            # idle filesystem. The handler may have optimistically flipped
            # ``refresh_in_progress`` true on POST so the polling fragment
            # always starts; clear it here so a no-op tick doesn't leave the
            # indicator polling forever.
            if current <= last_mtime and not _window_changed(app.state, current_days):
                # The handler may have optimistically flipped these true on
                # POST so the polling fragment always starts. Clear them on
                # a no-op tick so the indicator doesn't poll forever.
                # last_refreshed_at intentionally stays put — nothing was
                # refreshed, so the UI honestly reverts to the prior value.
                app.state.refresh_in_progress = False
                app.state.refresh_started_at = None
                continue
            log.info("JSONL changed; rebuilding materialized DB")
            app.state.refresh_started_at = datetime.now(UTC)
            app.state.refresh_in_progress = True
            try:
                await asyncio.to_thread(
                    _rebuild_sidecar,
                    sidecar,
                    jsonl_glob,
                    current_days,
                    resolve_projects,
                )
                await asyncio.to_thread(_swap_in, db_path, sidecar)
                # Record the post-swap window first so a freak exception on
                # the timestamp assignment can't leave state thinking the DB
                # still holds the previous window's data and force a needless
                # rebuild on the next tick.
                app.state.last_built_days = current_days
                app.state.last_refreshed_at = datetime.now(UTC)
            finally:
                app.state.refresh_in_progress = False
                app.state.refresh_started_at = None
            last_mtime = current
            log.info("refresh complete")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning(
                "refresh failed; will retry next tick",
                exc_info=True,
            )
            continue
