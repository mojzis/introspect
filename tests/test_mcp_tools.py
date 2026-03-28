"""Tests for MCP tool functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb

from introspect.db import materialize_views
from introspect.mcp.tools import (
    get_session,
    recent_sessions,
    search_conversations,
    tool_failures,
)
from introspect.search import build_search_corpus


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing MCP tools."""
    session_id = "test-session-mcp"
    jsonl_path = tmp_dir / "projects" / "test-project" / f"{session_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        {
            "type": "user",
            "timestamp": "2026-03-27T10:00:00.000Z",
            "sessionId": session_id,
            "uuid": "u1",
            "parentUuid": None,
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "message": {
                "role": "user",
                "content": "Help me refactor the database module",
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-27T10:00:01.000Z",
            "sessionId": session_id,
            "uuid": "a1",
            "parentUuid": "u1",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "requestId": "req1",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "id": "msg1",
                "content": [
                    {"type": "text", "text": "Sure, I can help with refactoring!"}
                ],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-27T10:00:02.000Z",
            "sessionId": session_id,
            "uuid": "a2",
            "parentUuid": "a1",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "requestId": "req2",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "id": "msg2",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_fail1",
                        "name": "Bash",
                        "input": {"command": "rm -rf /oops"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-03-27T10:00:03.000Z",
            "sessionId": session_id,
            "uuid": "u2",
            "parentUuid": "a2",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "sourceToolAssistantUUID": "a2",
            "toolUseResult": {
                "stdout": "",
                "stderr": "Permission denied",
            },
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fail1",
                        "content": "Permission denied",
                        "is_error": True,
                    }
                ],
            },
        },
    ]

    with jsonl_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


def _materialize_test_data(tmp_path: Path) -> Path:
    """Write sample data and materialize into DuckDB."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"
    glob_pattern = str(tmp_path / "projects" / "**" / "*.jsonl")

    conn = duckdb.connect(str(db_path))
    materialize_views(conn, glob_pattern)
    build_search_corpus(conn)
    conn.close()
    return db_path


def test_recent_sessions():
    """recent_sessions returns session metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = recent_sessions(n=10)

        assert "test-session-mcp" in result
        assert "main" in result


def test_get_session():
    """get_session returns full session content."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = get_session("test-session-mcp")

        assert "Session: test-session-mcp" in result
        assert "Messages" in result


def test_get_session_not_found():
    """get_session returns not-found message for missing session."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = get_session("nonexistent-session")

        assert "not found" in result


def test_tool_failures():
    """tool_failures returns failed tool calls."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = tool_failures()

        assert "Bash" in result
        assert "test-session-mcp" in result


def test_tool_failures_with_prefix():
    """tool_failures filters by command prefix."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = tool_failures(command_prefix="Bash")

        assert "Bash" in result

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = tool_failures(command_prefix="NonExistent")

        assert "No failed tool calls found" in result


def test_search_conversations():
    """search_conversations returns matching results."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = search_conversations("refactor database")

        assert "test-session-mcp" in result


def test_search_conversations_no_results():
    """search_conversations returns message when nothing matches."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            result = search_conversations("xyznonexistentterm123")

        assert "No results found" in result
