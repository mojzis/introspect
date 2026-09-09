"""Standalone stdio MCP lifecycle coverage."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

from introspect import refresh
from introspect.mcp import refresh_bridge
from introspect.mcp.server import create_mcp_server
from introspect.mcp.tools import recent_sessions
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


def test_stdio_refresh_publishes_preview_then_authority(monkeypatch, tmp_path: Path):
    """The standalone task uses the existing preview and sidecar loop phases."""
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
        loading_state=refresh.LoadingState(
            refresh.LoadingPhase.DISCOVERING, refresh.target_for_window("30")
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

    assert calls == [1, 30]
    assert state.database_ready is True
    assert state.loading_state.phase is LoadingPhase.READY
