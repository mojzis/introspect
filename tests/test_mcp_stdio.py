"""Standalone stdio MCP lifecycle coverage."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from introspect import refresh
from introspect.mcp import refresh_bridge
from introspect.mcp.server import create_mcp_server
from introspect.mcp.tools import recent_sessions, refresh_data
from introspect.refresh import LoadingPhase, run_stdio_refresh


def test_stdio_server_discovers_tools_without_starting_data_load():
    """Registration stays synchronous and does not touch the configured DB."""
    names = {tool.name for tool in asyncio.run(create_mcp_server().list_tools())}

    assert "recent_sessions" in names
    assert "refresh_data" in names


def test_data_tool_reports_loading_before_preview():
    state = types.SimpleNamespace(
        database_ready=False,
        loading_state=types.SimpleNamespace(
            phase=LoadingPhase.PREVIEWING,
            target=types.SimpleNamespace(window="30", days=30),
            candidate_count=2,
            completed_candidates=1,
            stage=None,
        ),
        refresh_target=types.SimpleNamespace(window="30", days=30),
        database_label="preview",
    )
    refresh_bridge.set_state(state)
    try:
        result = recent_sessions()
    finally:
        refresh_bridge.set_state(None)

    assert result.startswith("Data loading:")
    assert "candidates=1/2" in result


@pytest.mark.parametrize(("days", "expected_builds"), [(0, [0]), (30, [1, 30])])
def test_stdio_refresh_publishes_preview_then_authority(
    monkeypatch, tmp_path: Path, days: int, expected_builds: list[int]
):
    """The standalone task uses the existing preview and sidecar loop phases."""
    state = refresh.StdioRefreshState(
        db_path=tmp_path / "introspect.duckdb",
        jsonl_glob=str(tmp_path / "**" / "*.jsonl"),
        codex_glob=str(tmp_path / "codex" / "**" / "*.jsonl"),
        days=days,
        resolve_projects=False,
        interval_seconds=0,
        refresh_target=refresh.target_for_window(str(days)),
        refresh_window=str(days),
        refresh_trigger=asyncio.Event(),
        loading_state=refresh.LoadingState(
            refresh.LoadingPhase.DISCOVERING, refresh.target_for_window(str(days))
        ),
    )
    calls: list[int] = []

    monkeypatch.setattr(refresh, "has_compatible_materialized_db", lambda _: False)
    monkeypatch.setattr(
        refresh,
        "discover_cold_start_candidates",
        lambda *args, **kwargs: refresh.CandidateFiles(("preview.jsonl",), ()),
    )

    def build(*args, **kwargs):
        calls.append(args[2])

    monkeypatch.setattr(refresh, "_rebuild_sidecar", build)
    monkeypatch.setattr(refresh, "_swap_in", lambda *args: None)

    asyncio.run(run_stdio_refresh(state))

    assert calls == expected_builds
    assert state.database_ready is True
    assert state.loading_state.phase is LoadingPhase.READY

    assert state.database_label == "authoritative", "startup published full history"
    assert state.refresh_trigger is None, "one-shot startup has no manual consumer"
    refresh_bridge.set_state(state)
    try:
        response = asyncio.run(refresh_data(window="7"))
    finally:
        refresh_bridge.set_state(None)
    assert "manual refresh unavailable" in response
    assert state.refresh_target.days == days, "disabled refresh cannot change target"


def test_stdio_preview_failure_is_terminal_and_preserves_error(
    monkeypatch, tmp_path: Path
):
    state = refresh.StdioRefreshState(
        db_path=tmp_path / "introspect.duckdb",
        jsonl_glob=str(tmp_path / "**" / "*.jsonl"),
        codex_glob=str(tmp_path / "codex" / "**" / "*.jsonl"),
        days=30,
        resolve_projects=False,
        interval_seconds=0,
        refresh_target=refresh.target_for_window("30"),
        refresh_window="30",
        refresh_trigger=asyncio.Event(),
    )
    monkeypatch.setattr(refresh, "has_compatible_materialized_db", lambda _: False)
    monkeypatch.setattr(
        refresh,
        "discover_cold_start_candidates",
        lambda *args, **kwargs: refresh.CandidateFiles(("preview.jsonl",), ()),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic preview failure")  # noqa: TRY003

    monkeypatch.setattr(refresh, "_rebuild_sidecar", fail)
    asyncio.run(refresh.run_stdio_refresh(state))

    refresh_bridge.set_state(state)
    try:
        result = recent_sessions()
        refresh_result = asyncio.run(refresh_data())
    finally:
        refresh_bridge.set_state(None)

    assert "Data unavailable" in result
    assert "synthetic preview failure" in result
    assert "restart" in result, "failed startup must explain recovery"
    assert "Data unavailable" in refresh_result
    assert "no database snapshot" in refresh_result.lower()
    assert "synthetic preview failure" in refresh_result, "refresh retains error"


@pytest.mark.parametrize("days", [0, 30])
def test_stdio_refresh_keeps_periodic_consumer_alive(monkeypatch, tmp_path, days):
    target = refresh.target_for_window(str(days))
    state = refresh.StdioRefreshState(
        db_path=tmp_path / "introspect.duckdb",
        jsonl_glob=str(tmp_path / "*.jsonl"),
        codex_glob=str(tmp_path / "codex" / "*.jsonl"),
        days=days,
        resolve_projects=False,
        interval_seconds=600,
        refresh_target=target,
        refresh_window=target.window,
        refresh_trigger=None,
    )
    monkeypatch.setattr(refresh, "_rebuild_sidecar", lambda *args, **kwargs: None)
    published = asyncio.Event()

    def publish(*args):
        loop.call_soon_threadsafe(published.set)

    monkeypatch.setattr(refresh, "_swap_in", publish)

    async def exercise():
        nonlocal loop
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(run_stdio_refresh(state))
        try:
            await asyncio.wait_for(published.wait(), timeout=5)
            assert not task.done()
            refresh_bridge.set_state(state)
            try:
                response = await refresh_data(window="7")
            finally:
                refresh_bridge.set_state(None)
            assert "Refresh complete" in response
            assert state.last_built_days == 7
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert state.refresh_trigger is None

    loop: asyncio.AbstractEventLoop
    asyncio.run(exercise())
