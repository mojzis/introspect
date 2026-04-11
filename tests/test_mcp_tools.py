"""Tests for MCP tool functions."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from introspect.db import materialize_views
from introspect.mcp.tools import (
    _SQL_CELL_MAX,
    _SQL_ROW_CAP,
    describe_schema,
    get_session,
    recent_sessions,
    run_sql,
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


@pytest.fixture
def patched_mcp_db() -> Iterator[None]:
    """Materialize sample data and patch the MCP tool DB handles to use it.

    Patches both ``get_read_connection`` (used by the parameterized tools)
    and ``DEFAULT_DB_PATH`` (which ``run_sql`` opens directly as a strict
    read-only connection).
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _materialize_test_data(Path(tmp))
        with (
            patch("introspect.mcp.tools.get_read_connection") as mock_conn,
            patch("introspect.mcp.tools.DEFAULT_DB_PATH", db_path),
        ):
            mock_conn.return_value = duckdb.connect(str(db_path), read_only=True)
            yield


def test_describe_schema_lists_core_views(patched_mcp_db: None):
    """describe_schema surfaces the main views with their columns."""
    result = describe_schema()

    assert "logical_sessions:" in result
    assert "tool_calls:" in result
    assert "conversation_turns:" in result
    # Priority views should appear before alphabetically-later ones.
    assert result.index("logical_sessions:") < result.index("tool_calls:")
    assert "session_id" in result


def test_run_sql_happy_path(patched_mcp_db: None):
    """run_sql executes a SELECT and returns a formatted table."""
    result = run_sql("SELECT session_id, user_messages FROM logical_sessions")

    assert "session_id" in result
    assert "test-session-mcp" in result
    assert "1 rows" in result


def test_run_sql_with_cte(patched_mcp_db: None):
    """run_sql accepts WITH (CTE) queries in addition to plain SELECT."""
    result = run_sql("WITH s AS (SELECT session_id FROM logical_sessions) SELECT * FROM s")

    assert "test-session-mcp" in result


def test_run_sql_rejects_write_statement(patched_mcp_db: None):
    """run_sql blocks non-SELECT statements at the tool layer."""
    result = run_sql("DELETE FROM logical_sessions")

    assert "Error" in result
    assert "SELECT" in result


def test_run_sql_rejects_attach(patched_mcp_db: None):
    """run_sql rejects a single ATTACH statement by the first-keyword check."""
    result = run_sql("ATTACH 'evil.db' AS evil")

    assert "Error" in result
    assert "SELECT" in result


def test_run_sql_rejects_multiple_statements(patched_mcp_db: None):
    """run_sql rejects scripts with more than one statement."""
    result = run_sql("SELECT 1; SELECT 2")

    assert "Error" in result
    assert "Multiple" in result


def test_run_sql_allows_keywords_inside_string_literals(patched_mcp_db: None):
    """Literals like 'please delete' must not trigger false-positive rejection."""
    result = run_sql("SELECT 'please delete; drop insert' AS note")

    # Should execute successfully and return the literal.
    assert "please delete" in result
    assert "1 rows" in result


def test_run_sql_enforces_limit(patched_mcp_db: None):
    """run_sql caps the number of returned rows at the caller's limit."""
    result = run_sql("SELECT * FROM range(0, 50) AS t(n)", limit=5)

    assert "(5 rows)" in result


def test_run_sql_row_cap_clamps_oversized_limit(patched_mcp_db: None):
    """Caller limits above _SQL_ROW_CAP are clamped to the cap."""
    result = run_sql(
        f"SELECT * FROM range(0, {_SQL_ROW_CAP * 2}) AS t(n)",
        limit=_SQL_ROW_CAP * 2,
    )

    assert f"({_SQL_ROW_CAP} rows)" in result


def test_run_sql_truncates_long_cells(patched_mcp_db: None):
    """Cell values longer than _SQL_CELL_MAX are truncated with an ellipsis."""
    long_value_length = _SQL_CELL_MAX + 50
    result = run_sql(f"SELECT repeat('x', {long_value_length}) AS big")

    assert "…" in result
    # The header + separator + the truncated cell row + "(1 rows)" footer.
    assert "1 rows" in result


def test_run_sql_surfaces_duckdb_errors(patched_mcp_db: None):
    """run_sql reports DuckDB errors (e.g. unknown table) as text with type."""
    result = run_sql("SELECT * FROM no_such_table")

    assert "SQL error" in result
    # Exception type name is included so the caller can tell error classes apart.
    assert "CatalogException" in result


def test_run_sql_allows_double_quoted_identifier_with_forbidden_word(
    patched_mcp_db: None,
):
    """Double-quoted identifiers (not string literals) must not be rewritten."""
    # A column aliased with a word that used to be in the blocklist should work.
    result = run_sql('SELECT 1 AS "delete_me"')

    assert "delete_me" in result
    assert "1 rows" in result


def test_run_sql_missing_db_returns_friendly_error():
    """run_sql fails closed when the materialized DB file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does-not-exist.duckdb"
        with patch("introspect.mcp.tools.DEFAULT_DB_PATH", missing):
            result = run_sql("SELECT 1")

        assert "Error" in result
        assert "materialized DB not found" in result


def test_run_sql_outer_limit_caps_unbounded_queries(patched_mcp_db: None):
    """Caller gets capped rows even without LIMIT in their own SQL."""
    # 50 rows of input, caller requests limit=5 — the outer wrap must cap.
    result = run_sql("SELECT * FROM range(0, 50) AS t(n)", limit=5)

    assert "(5 rows)" in result


def test_search_conversations_rejects_invalid_role(patched_mcp_db: None):
    """Invalid role is caught in the MCP wrapper with a friendly error."""
    result = search_conversations("refactor", role="usr")

    assert "Error" in result
    assert "role" in result


def test_search_conversations_rejects_invalid_since(patched_mcp_db: None):
    """Garbage since values are rejected before hitting DuckDB."""
    result = search_conversations("refactor", since="last week")

    assert "Error" in result
    assert "since" in result
