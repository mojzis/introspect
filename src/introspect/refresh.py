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


class RefreshState(Protocol):
    """Contract between :func:`refresh_loop` (writer) and :func:`wait_for_refresh`
    (reader). FastAPI's ``app.state`` satisfies this after :mod:`api.main` sets
    the three attributes during startup.
    """

    refresh_trigger: asyncio.Event | None
    refresh_in_progress: bool
    last_refreshed_at: datetime | None


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


def _swap_in(
    app: FastAPI,
    db_path: Path,
    sidecar: Path,
) -> None:
    """Atomically rename ``sidecar`` over ``db_path`` and swap the read conn.

    DuckDB caches instances by path, so we must close the old connection
    before opening a new one to see the replaced inode.
    """
    os.replace(str(sidecar), str(db_path))  # noqa: PTH105
    old_conn = app.state.read_conn
    with contextlib.suppress(duckdb.ConnectionException, RuntimeError):
        if old_conn is not None:
            old_conn.close()
    app.state.read_conn = duckdb.connect(str(db_path), read_only=True)


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
    filesystem is a fast no-op.
    """
    sidecar = db_path.with_name(db_path.name + ".next")
    last_mtime = newest_mtime(jsonl_glob)
    while True:
        try:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(trigger.wait(), timeout=interval_seconds)
            trigger.clear()
            current = newest_mtime(jsonl_glob)
            if current <= last_mtime:
                continue
            log.info("JSONL changed; rebuilding materialized DB")
            app.state.refresh_in_progress = True
            try:
                await asyncio.to_thread(
                    _rebuild_sidecar, sidecar, jsonl_glob, days, resolve_projects
                )
                await asyncio.to_thread(_swap_in, app, db_path, sidecar)
                app.state.last_refreshed_at = datetime.now(UTC)
            finally:
                app.state.refresh_in_progress = False
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
