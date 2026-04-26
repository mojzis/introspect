"""Tests for the background refresh loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import types
from pathlib import Path

import duckdb
import pytest

from introspect import refresh
from introspect.db import materialize_views
from introspect.refresh import newest_mtime, refresh_loop
from introspect.search import build_search_corpus
from tests.conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)


def _write_session(
    tmp_path: Path, session_id: str, subdir: str = "test-project"
) -> Path:
    user = make_user_message(
        session_id=session_id,
        uuid=f"{session_id}-u1",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00.000Z",
        content="hello",
        tool_use_result={"stdout": "", "stderr": ""},
    )
    assistant = make_assistant_message(
        session_id=session_id,
        uuid=f"{session_id}-a1",
        parent_uuid=f"{session_id}-u1",
        timestamp="2026-01-01T00:00:01.000Z",
        content=[{"type": "text", "text": "hi"}],
    )
    jsonl_path = tmp_path / "projects" / subdir / f"{session_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        f.write(json.dumps(user) + "\n")
        f.write(json.dumps(assistant) + "\n")
    return jsonl_path


def _build_initial_db(db_path: Path, jsonl_glob: str) -> None:
    conn = duckdb.connect(str(db_path))
    try:
        materialize_views(conn, jsonl_glob, days=0, resolve_projects=False)
        build_search_corpus(conn)
    finally:
        conn.close()


def _fake_app() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        state=types.SimpleNamespace(
            refresh_in_progress=False,
            last_refreshed_at=None,
            last_built_days=0,
        )
    )


def test_newest_mtime_empty_glob(tmp_path: Path) -> None:
    pattern = str(tmp_path / "nope" / "**" / "*.jsonl")
    assert newest_mtime(pattern) == 0.0


def test_newest_mtime_tracks_updates(tmp_path: Path) -> None:
    write_jsonl(tmp_path, "sess-1", [])
    pattern = glob_pattern(tmp_path)
    m1 = newest_mtime(pattern)
    assert m1 > 0.0
    time.sleep(0.01)
    jsonl = tmp_path / "projects" / "test-project" / "sess-1.jsonl"
    # Bump mtime explicitly so this works on filesystems with coarse resolution.
    new_ts = m1 + 1.0
    os.utime(jsonl, (new_ts, new_ts))
    m2 = newest_mtime(pattern)
    assert m2 > m1


def test_refresh_short_circuits_when_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session(tmp_path, "sess-stable")
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    counter = {"n": 0}

    def fake_rebuild(*args, **kwargs):
        counter["n"] += 1

    monkeypatch.setattr(refresh, "_rebuild_sidecar", fake_rebuild)

    app = _fake_app()

    async def run() -> None:
        task = asyncio.create_task(
            refresh_loop(
                app,  # ty: ignore[invalid-argument-type]
                db_path,
                jsonl_glob,
                0,
                False,
                interval_seconds=0.05,
                trigger=asyncio.Event(),
            )
        )
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert counter["n"] == 0


def test_refresh_rebuilds_and_swaps_on_change(tmp_path: Path) -> None:
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    _write_session(tmp_path, "sess-first", subdir="proj-a")
    _build_initial_db(db_path, jsonl_glob)

    app = _fake_app()

    async def run() -> None:
        task = asyncio.create_task(
            refresh_loop(
                app,  # ty: ignore[invalid-argument-type]
                db_path,
                jsonl_glob,
                0,
                False,
                interval_seconds=0.05,
                trigger=asyncio.Event(),
            )
        )
        try:
            # No change yet - loop should not rebuild.
            await asyncio.sleep(0.2)
            assert app.state.last_refreshed_at is None

            # Ensure a strictly greater mtime for the new file.
            current_latest = newest_mtime(jsonl_glob)
            time.sleep(0.05)
            _write_session(tmp_path, "sess-second", subdir="proj-b")
            new_file = tmp_path / "projects" / "proj-b" / "sess-second.jsonl"
            bumped = current_latest + 1.0
            os.utime(new_file, (bumped, bumped))

            # Wait long enough for at least one refresh cycle.
            for _ in range(40):
                await asyncio.sleep(0.1)
                if app.state.last_refreshed_at is not None:
                    break
            assert app.state.last_refreshed_at is not None
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())

    # Open a fresh read-only connection against the post-swap inode.
    fresh = duckdb.connect(str(db_path), read_only=True)
    try:
        session_ids = {
            row[0]
            for row in fresh.execute(
                "SELECT DISTINCT session_id FROM raw_messages"
            ).fetchall()
        }
    finally:
        fresh.close()
    assert "sess-first" in session_ids
    assert "sess-second" in session_ids


def test_refresh_survives_rebuild_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_session(tmp_path, "sess-err")
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    # Make newest_mtime strictly increasing so the loop sees "changed" on each tick.
    counter = {"n": 0}

    def fake_newest_mtime(_glob: str) -> float:
        counter["n"] += 1
        return float(counter["n"])

    monkeypatch.setattr(refresh, "newest_mtime", fake_newest_mtime)

    rebuild_calls = {"n": 0}

    def fake_rebuild(*args, **kwargs):
        rebuild_calls["n"] += 1
        if rebuild_calls["n"] == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(refresh, "_rebuild_sidecar", fake_rebuild)

    # Prevent _swap_in from touching the filesystem or read_conn.
    monkeypatch.setattr(refresh, "_swap_in", lambda *a, **kw: None)

    app = _fake_app()

    async def run() -> None:
        with caplog.at_level(logging.WARNING, logger="introspect.refresh"):
            task = asyncio.create_task(
                refresh_loop(
                    app,  # ty: ignore[invalid-argument-type]
                    db_path,
                    jsonl_glob,
                    0,
                    False,
                    interval_seconds=0.05,
                    trigger=asyncio.Event(),
                )
            )
            try:
                # Allow several ticks.
                for _ in range(30):
                    await asyncio.sleep(0.05)
                    if rebuild_calls["n"] >= 2:
                        break
                assert not task.done(), "refresh task should still be running"
                assert rebuild_calls["n"] >= 2
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(run())

    assert any(
        "refresh failed" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_refresh_wakes_on_trigger(tmp_path: Path) -> None:
    """Setting the trigger event should cause the loop to rebuild promptly.

    Interval is 10 s so a naive ``asyncio.sleep(interval)`` would never swap
    in time; if we see ``last_refreshed_at`` move inside ~5 s, the trigger
    wiring works.
    """
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    _write_session(tmp_path, "sess-trig-a", subdir="proj-a")
    _build_initial_db(db_path, jsonl_glob)

    app = _fake_app()
    trigger = asyncio.Event()

    async def run() -> None:
        task = asyncio.create_task(
            refresh_loop(
                app,  # ty: ignore[invalid-argument-type]
                db_path,
                jsonl_glob,
                0,
                False,
                interval_seconds=10.0,
                trigger=trigger,
            )
        )
        try:
            # Let the task actually start and capture ``last_mtime`` *before*
            # we bump the filesystem, otherwise the mtime short-circuit
            # swallows the wake.
            await asyncio.sleep(0.1)

            current_latest = newest_mtime(jsonl_glob)
            time.sleep(0.05)
            _write_session(tmp_path, "sess-trig-b", subdir="proj-b")
            new_file = tmp_path / "projects" / "proj-b" / "sess-trig-b.jsonl"
            bumped = current_latest + 1.0
            os.utime(new_file, (bumped, bumped))

            trigger.set()

            # Give the loop a chance to wake, rebuild, and swap. The rebuild
            # runs ``materialize_views`` in a thread so total time varies;
            # poll generously and rely on interval_seconds=10 to prove the
            # swap came from the trigger, not the timeout.
            for _ in range(100):
                await asyncio.sleep(0.05)
                if app.state.last_refreshed_at is not None:
                    break
            assert app.state.last_refreshed_at is not None, (
                "trigger did not cause a refresh"
            )
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_last_refreshed_at_updates_after_swap(tmp_path: Path) -> None:
    """After a successful swap, ``last_refreshed_at`` should advance."""
    from datetime import UTC, datetime  # noqa: PLC0415

    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    _write_session(tmp_path, "sess-lr-a", subdir="proj-a")
    _build_initial_db(db_path, jsonl_glob)

    app = _fake_app()
    before = datetime.now(UTC)
    app.state.last_refreshed_at = before

    async def run() -> None:
        task = asyncio.create_task(
            refresh_loop(
                app,  # ty: ignore[invalid-argument-type]
                db_path,
                jsonl_glob,
                0,
                False,
                interval_seconds=0.05,
                trigger=asyncio.Event(),
            )
        )
        try:
            # Let the task capture ``last_mtime`` before we bump the fs.
            await asyncio.sleep(0.1)

            current_latest = newest_mtime(jsonl_glob)
            time.sleep(0.05)
            _write_session(tmp_path, "sess-lr-b", subdir="proj-b")
            new_file = tmp_path / "projects" / "proj-b" / "sess-lr-b.jsonl"
            bumped = current_latest + 1.0
            os.utime(new_file, (bumped, bumped))

            for _ in range(40):
                await asyncio.sleep(0.1)
                if app.state.last_refreshed_at != before:
                    break
            assert app.state.last_refreshed_at > before
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_refresh_clears_in_progress_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising ``_rebuild_sidecar`` must leave ``refresh_in_progress=False``."""
    _write_session(tmp_path, "sess-err-flag")
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    # Force every tick to look like a change.
    counter = {"n": 0}

    def fake_newest_mtime(_glob: str) -> float:
        counter["n"] += 1
        return float(counter["n"])

    monkeypatch.setattr(refresh, "newest_mtime", fake_newest_mtime)

    def fake_rebuild(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(refresh, "_rebuild_sidecar", fake_rebuild)
    monkeypatch.setattr(refresh, "_swap_in", lambda *a, **kw: None)

    app = _fake_app()

    async def run() -> None:
        task = asyncio.create_task(
            refresh_loop(
                app,  # ty: ignore[invalid-argument-type]
                db_path,
                jsonl_glob,
                0,
                False,
                interval_seconds=0.05,
                trigger=asyncio.Event(),
            )
        )
        try:
            # Poll a few ticks; the loop should error-and-retry each time,
            # and refresh_in_progress should end up False after each attempt.
            for _ in range(30):
                await asyncio.sleep(0.05)
                if counter["n"] >= 3:
                    break
            # Give the finally block one more tick to settle.
            await asyncio.sleep(0.1)
            assert app.state.refresh_in_progress is False
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
