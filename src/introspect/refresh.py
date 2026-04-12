"""Background refresh loop for keeping the materialized DB in sync with JSONL files."""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from introspect.db import materialize_views
from introspect.search import build_search_corpus

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)


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
) -> None:
    """Poll JSONL mtime and rebuild the materialized DB when files change."""
    sidecar = db_path.with_name(db_path.name + ".next")
    last_mtime = newest_mtime(jsonl_glob)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            current = newest_mtime(jsonl_glob)
            if current <= last_mtime:
                continue
            log.info("JSONL changed; rebuilding materialized DB")
            await asyncio.to_thread(
                _rebuild_sidecar, sidecar, jsonl_glob, days, resolve_projects
            )
            await asyncio.to_thread(_swap_in, app, db_path, sidecar)
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
