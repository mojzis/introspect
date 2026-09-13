"""Tests for the background refresh loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import types
from pathlib import Path

import duckdb
import pytest

from introspect import refresh
from introspect.db import materialize_views
from introspect.refresh import (
    LoadingPhase,
    LoadingStage,
    RefreshState,
    RefreshTarget,
    discover_cold_start_candidates,
    newest_mtime,
    refresh_loop,
    set_refresh_target,
    target_for_window,
)
from introspect.search import build_search_corpus
from tests.conftest import (
    codex_glob_pattern,
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_codex_rollout,
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


def _instrument_refresh_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event]:
    mtime_checked = threading.Event()
    refresh_ready = threading.Event()
    original_mtime = refresh.newest_mtime
    original_set_loading = refresh._set_loading

    def observe_mtime(jsonl_glob: str, codex_glob: str | None = None) -> float:
        result = original_mtime(jsonl_glob, codex_glob)
        mtime_checked.set()
        return result

    def observe_set_loading(
        state: RefreshState,
        phase: LoadingPhase,
        target: RefreshTarget,
        *,
        stage: LoadingStage | None = None,
        candidate_count: int = 0,
        completed_candidates: int = 0,
        error: str | None = None,
    ) -> None:
        original_set_loading(
            state,
            phase,
            target,
            stage=stage,
            candidate_count=candidate_count,
            completed_candidates=completed_candidates,
            error=error,
        )
        if state.loading_state.phase is LoadingPhase.READY:
            refresh_ready.set()

    monkeypatch.setattr(refresh, "newest_mtime", observe_mtime)
    monkeypatch.setattr(refresh, "_set_loading", observe_set_loading)
    return mtime_checked, refresh_ready


def test_newest_mtime_empty_glob(tmp_path: Path) -> None:
    pattern = str(tmp_path / "nope" / "**" / "*.jsonl")
    assert newest_mtime(pattern) == 0.0


def test_newest_mtime_tracks_updates(tmp_path: Path) -> None:
    write_jsonl(tmp_path, "sess-1", [])
    pattern = glob_pattern(tmp_path)
    m1 = newest_mtime(pattern)
    assert m1 > 0.0
    jsonl = tmp_path / "projects" / "test-project" / "sess-1.jsonl"
    # Bump mtime explicitly so this works on filesystems with coarse resolution.
    new_ts = m1 + 1.0
    os.utime(jsonl, (new_ts, new_ts))
    m2 = newest_mtime(pattern)
    assert m2 > m1


def test_newest_mtime_watches_codex_glob_too(tmp_path: Path) -> None:
    """A Codex-only mtime bump must be picked up when ``codex_glob`` is given."""
    write_jsonl(tmp_path, "sess-1", [])
    jsonl_glob = glob_pattern(tmp_path)
    codex_glob = codex_glob_pattern(tmp_path)

    # No Codex files yet — watching both globs matches the Claude-only mtime.
    baseline = newest_mtime(jsonl_glob, codex_glob)
    assert baseline == newest_mtime(jsonl_glob)

    codex_path = write_codex_rollout(tmp_path, "codex-1", [])
    new_ts = baseline + 1.0
    os.utime(codex_path, (new_ts, new_ts))

    assert newest_mtime(jsonl_glob, codex_glob) > baseline
    # Without codex_glob, the new Codex file is invisible.
    assert newest_mtime(jsonl_glob) == baseline


def test_cold_start_candidates_use_guard_band_and_codex_partitions(
    tmp_path: Path,
) -> None:
    now = refresh.datetime(2026, 8, 28, 12, tzinfo=refresh.UTC)
    claude = tmp_path / "claude" / "session.jsonl"
    stale_claude = tmp_path / "claude" / "stale.jsonl"
    codex = tmp_path / "codex" / "2026" / "08" / "28" / "rollout.jsonl"
    old_codex = tmp_path / "codex" / "2026" / "08" / "01" / "rollout.jsonl"
    for path in (  # zorilla: ignore[ZR001] -- provider partition fixture
        claude,
        stale_claude,
        codex,
        old_codex,
    ):  # zorilla: ignore[ZR001] -- provider partition fixture
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    os.utime(claude, (now.timestamp(), now.timestamp()))
    stale_mtime = (now - refresh.timedelta(days=3)).timestamp()
    os.utime(stale_claude, (stale_mtime, stale_mtime))
    os.utime(codex, (stale_mtime, stale_mtime))
    os.utime(old_codex, (now.timestamp(), now.timestamp()))

    candidates = discover_cold_start_candidates(
        str(tmp_path / "claude" / "*.jsonl"),
        str(tmp_path / "codex" / "**" / "*.jsonl"),
        days=1,
        now=now,
    )

    assert candidates.claude == (str(claude),)
    assert candidates.codex == (str(codex),)


def test_target_override_advances_generation_and_loading_is_terminal() -> None:
    state = types.SimpleNamespace(
        refresh_target=target_for_window("30"),
        refresh_window="30",
        refresh_pending=False,
    )
    target = set_refresh_target(state, "7")

    assert (  # zorilla: ignore[ZR004] -- refresh target contract
        target.days == 7
    )  # zorilla: ignore[ZR004] -- refresh target contract
    assert target.generation == 1
    assert state.refresh_target == target
    assert state.refresh_pending is True
    assert LoadingPhase.PREVIEW_READY.value == "preview_ready"

    state.refresh_pending = False
    same_target = set_refresh_target(state, "7")
    assert same_target == target
    assert same_target.generation == target.generation
    assert state.refresh_pending is False


def test_numeric_targets_include_custom_days_and_all_data() -> None:
    custom = target_for_window("14")
    all_data = target_for_window("30", days=0)

    assert custom.window == "14"
    assert custom.days == 14
    assert all_data.window == "0"
    assert all_data.days == 0


def test_initial_refresh_runs_without_waiting_for_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An initial full-target refresh starts immediately, including one-shot mode."""
    _write_session(tmp_path, "sess-initial")
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"
    app = _fake_app()
    app.state.refresh_target = target_for_window("7")
    app.state.refresh_window = "7"
    app.state.refresh_pending = True

    rebuild_calls = {"n": 0}

    def fake_rebuild(*args, **kwargs):
        rebuild_calls["n"] += 1

    monkeypatch.setattr(refresh, "_rebuild_sidecar", fake_rebuild)
    monkeypatch.setattr(refresh, "_swap_in", lambda *args, **kwargs: None)

    async def run() -> None:
        task = asyncio.create_task(
            refresh_loop(
                app,  # ty: ignore[invalid-argument-type]
                db_path,
                jsonl_glob,
                7,
                False,
                interval_seconds=600,
                trigger=asyncio.Event(),
                initial=True,
                one_shot=True,
            )
        )
        await asyncio.wait_for(task, timeout=0.5)

    asyncio.run(run())
    assert rebuild_calls["n"] == 1


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


def test_refresh_rebuilds_and_swaps_on_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    _write_session(tmp_path, "sess-first", subdir="proj-a")
    _build_initial_db(db_path, jsonl_glob)

    app = _fake_app()
    mtime_checked, refresh_ready = _instrument_refresh_lifecycle(monkeypatch)

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
            # The loop must capture the old mtime before the new file appears.
            assert await asyncio.to_thread(mtime_checked.wait, 1.0), (
                "refresh loop did not inspect the initial mtime"
            )
            assert app.state.last_refreshed_at is None, (
                "unchanged data must not rebuild"
            )

            # Ensure a strictly greater mtime for the new file.
            current_latest = newest_mtime(jsonl_glob)
            _write_session(tmp_path, "sess-second", subdir="proj-b")
            new_file = tmp_path / "projects" / "proj-b" / "sess-second.jsonl"
            bumped = current_latest + 1.0
            os.utime(new_file, (bumped, bumped))

            assert await asyncio.to_thread(refresh_ready.wait, 5.0), (
                "changed data was not swapped into the live database"
            )
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

    def fake_newest_mtime(_glob: str, _codex_glob: str | None = None) -> float:
        counter["n"] += 1
        return float(counter["n"])

    monkeypatch.setattr(refresh, "newest_mtime", fake_newest_mtime)

    rebuild_calls = {"n": 0}
    rebuild_retried = threading.Event()

    def fake_rebuild(*args, **kwargs):
        rebuild_calls["n"] += 1
        if rebuild_calls["n"] == 1:
            raise RuntimeError("boom")
        rebuild_retried.set()

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
                assert await asyncio.to_thread(rebuild_retried.wait, 1.0), (
                    "refresh loop did not retry after the rebuild error"
                )
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


def test_refresh_wakes_on_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    mtime_checked, refresh_ready = _instrument_refresh_lifecycle(monkeypatch)

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
            assert await asyncio.to_thread(mtime_checked.wait, 1.0), (
                "refresh loop did not inspect the initial mtime"
            )

            current_latest = newest_mtime(jsonl_glob)
            _write_session(tmp_path, "sess-trig-b", subdir="proj-b")
            new_file = tmp_path / "projects" / "proj-b" / "sess-trig-b.jsonl"
            bumped = current_latest + 1.0
            os.utime(new_file, (bumped, bumped))

            trigger.set()

            assert await asyncio.to_thread(refresh_ready.wait, 5.0), (
                "trigger did not complete a refresh swap"
            )
            assert app.state.last_refreshed_at is not None, (
                "trigger did not cause a refresh"
            )
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_last_refreshed_at_updates_after_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a successful swap, ``last_refreshed_at`` should advance."""
    from datetime import UTC, datetime  # noqa: PLC0415

    jsonl_glob = glob_pattern(tmp_path)
    db_path = tmp_path / "db.duckdb"

    _write_session(tmp_path, "sess-lr-a", subdir="proj-a")
    _build_initial_db(db_path, jsonl_glob)

    app = _fake_app()
    before = datetime.now(UTC)
    app.state.last_refreshed_at = before
    mtime_checked, refresh_ready = _instrument_refresh_lifecycle(monkeypatch)

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
            assert await asyncio.to_thread(mtime_checked.wait, 1.0), (
                "refresh loop did not inspect the initial mtime"
            )

            current_latest = newest_mtime(jsonl_glob)
            _write_session(tmp_path, "sess-lr-b", subdir="proj-b")
            new_file = tmp_path / "projects" / "proj-b" / "sess-lr-b.jsonl"
            bumped = current_latest + 1.0
            os.utime(new_file, (bumped, bumped))

            assert await asyncio.to_thread(refresh_ready.wait, 5.0), (
                "refresh did not complete its swap"
            )
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

    def fake_newest_mtime(_glob: str, _codex_glob: str | None = None) -> float:
        counter["n"] += 1
        return float(counter["n"])

    monkeypatch.setattr(refresh, "newest_mtime", fake_newest_mtime)

    rebuild_finished = threading.Event()

    def fake_rebuild(*args, **kwargs):
        try:
            raise RuntimeError("boom")
        finally:
            rebuild_finished.set()

    monkeypatch.setattr(refresh, "_rebuild_sidecar", fake_rebuild)
    monkeypatch.setattr(refresh, "_swap_in", lambda *a, **kw: None)
    cleanup_done = threading.Event()
    original_finish = refresh._finish_refresh

    def observe_finish(*args, **kwargs):
        result = original_finish(*args, **kwargs)
        if not args[0].refresh_in_progress:
            cleanup_done.set()
        return result

    monkeypatch.setattr(refresh, "_finish_refresh", observe_finish)

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
            assert await asyncio.to_thread(rebuild_finished.wait, 1.0), (
                "raising rebuild did not execute its cleanup"
            )
            assert await asyncio.to_thread(cleanup_done.wait, 1.0), (
                "refresh loop did not finish error cleanup"
            )
            assert app.state.refresh_in_progress is False, (
                "refresh cleanup must clear the in-progress flag"
            )
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
