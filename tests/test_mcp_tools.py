"""Tests for MCP tool functions."""

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

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "test-session-mcp"


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing MCP tools."""
    lines = [
        make_user_message(
            SID,
            "u1",
            None,
            "2026-03-27T10:00:00.000Z",
            "Help me refactor the database module",
        ),
        make_assistant_message(
            SID,
            "a1",
            "u1",
            "2026-03-27T10:00:01.000Z",
            [{"type": "text", "text": "Sure, I can help with refactoring!"}],
        ),
        make_assistant_message(
            SID,
            "a2",
            "a1",
            "2026-03-27T10:00:02.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_fail1",
                    "name": "Bash",
                    "input": {"command": "rm -rf /oops"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u2",
            "a2",
            "2026-03-27T10:00:03.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_fail1",
                    "content": "Permission denied",
                    "is_error": True,
                }
            ],
            tool_use_result={"stdout": "", "stderr": "Permission denied"},
            source_tool_uuid="a2",
        ),
    ]
    return write_jsonl(tmp_dir, SID, lines)


def _materialize_test_data(tmp_path: Path) -> Path:
    """Write sample data and materialize into DuckDB."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"

    conn = duckdb.connect(str(db_path))
    materialize_views(conn, glob_pattern(tmp_path))
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
