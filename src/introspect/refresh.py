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
    grace_seconds: float = 2.0,
) -> None:
    """Atomically rename ``sidecar`` over ``db_path`` and swap the read conn.

    Note on ordering: DuckDB maintains a process-wide cache of database
    instances keyed by path. Calling ``duckdb.connect(path)`` while another
    connection on the same path is still open returns a reference to the
    *existing* in-memory instance — the new inode produced by ``os.replace``
    is not observed. This defeats the "soft-swap with grace period" shape
    described in the plan: we can't keep the old connection alive while
    opening a new one that sees the refreshed file. So the order is:

    1. ``os.replace`` the sidecar over ``db_path`` (atomic on one filesystem).
    2. Close the old connection so DuckDB releases its cached instance.
    3. Open a fresh read-only connection on ``db_path`` — this now sees the
       new inode.
    4. Publish it via ``app.state.read_conn``.

    The tradeoff is that any in-flight cursor holding the old connection will
    observe a closed connection. In practice this is extremely rare for this
    app (interactive log browsing, low concurrency) and any such request will
    just fail the usual way — the next click retries against the fresh conn.
    ``grace_seconds`` is kept in the signature for compatibility but is
    unused.
    """
    _ = grace_seconds  # signature parity with plan; see docstring
    os.replace(str(sidecar), str(db_path))  # noqa: PTH105
    old_conn = app.state.read_conn
    with contextlib.suppress(Exception):
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
            log.info("introspect: JSONL changed; rebuilding materialized DB")
            await asyncio.to_thread(
                _rebuild_sidecar, sidecar, jsonl_glob, days, resolve_projects
            )
            _swap_in(app, db_path, sidecar)
            last_mtime = current
            log.info("introspect: refresh complete")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning(
                "introspect: refresh failed; will retry next tick",
                exc_info=True,
            )
            continue
