"""Tests for introspect database views."""

import tempfile
from pathlib import Path

import duckdb

from introspect.db import get_connection, get_read_connection, materialize_views

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "test-session-001"


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing."""
    lines = [
        make_user_message(
            SID,
            "u1",
            None,
            "2026-03-27T10:00:00.000Z",
            "Hello, help me with tests",
        ),
        make_assistant_message(
            SID,
            "a1",
            "u1",
            "2026-03-27T10:00:01.000Z",
            [{"type": "text", "text": "Sure, I can help!"}],
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
        make_assistant_message(
            SID,
            "a2",
            "a1",
            "2026-03-27T10:00:02.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_test1",
                    "name": "Bash",
                    "input": {"command": "echo hello", "description": "test"},
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
                    "tool_use_id": "toolu_test1",
                    "content": "hello\n",
                    "is_error": False,
                }
            ],
            tool_use_result={
                "stdout": "hello\n",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
                "noOutputExpected": False,
            },
            source_tool_uuid="a2",
        ),
    ]
    return write_jsonl(tmp_dir, SID, lines)


def test_views_created():
    """Test that all views are created successfully."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        # Check views exist
        views = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_type = 'VIEW'
        """).fetchall()
        view_names = {v[0] for v in views}
        assert "raw_messages" in view_names
        assert "logical_sessions" in view_names
        assert "tool_calls" in view_names
        assert "conversation_turns" in view_names
        assert "session_titles" in view_names
        conn.close()


def test_raw_messages():
    """Test raw_messages view returns correct data."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        rows = conn.execute("SELECT * FROM raw_messages").fetchall()
        assert len(rows) == 4

        # Check session_id is consistent
        session_ids = {r[3] for r in rows}
        assert session_ids == {"test-session-001"}
        conn.close()


def test_logical_sessions():
    """Test logical_sessions view aggregation."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        rows = conn.execute("SELECT * FROM logical_sessions").fetchall()
        assert len(rows) == 1

        session = rows[0]
        # Fields: session_id, started_at, ended_at, duration,
        #   user_msgs, asst_msgs, model, cwd, git_branch, entrypoint
        assert session[0] == "test-session-001"
        assert session[4] == 1  # user_messages (not tool result)
        assert session[5] == 2
        conn.close()


def test_tool_calls():
    """Test tool_calls view joins use and result."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        rows = conn.execute("SELECT * FROM tool_calls").fetchall()
        assert len(rows) == 1

        tool_call = rows[0]
        # Fields: session_id, called_at, tool_name, tool_use_id,
        #   tool_input, is_error, tool_use_result, result_at, exec_time
        assert tool_call[2] == "Bash"
        assert tool_call[3] == "toolu_test1"
        conn.close()


def test_get_read_connection_uses_materialized():
    """get_read_connection returns read-only conn when materialized tables exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)

        # First materialize the data
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pat)
        conn.close()

        # Now get_read_connection should return a read-only connection
        conn = get_read_connection(db_path, glob_pat)
        try:
            # Should be able to query materialized tables
            rows = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
            assert rows is not None
            assert rows[0] == 4

            # Should have materialized tables (BASE TABLE, not VIEW)
            tables = conn.execute(
                "SELECT table_type FROM information_schema.tables "
                "WHERE table_name = 'raw_messages'"
            ).fetchone()
            assert tables is not None
            assert tables[0] == "BASE TABLE"
        finally:
            conn.close()


def test_get_read_connection_falls_back_to_lazy():
    """get_read_connection falls back to lazy views when no materialized tables."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "fresh.duckdb"
        glob_pat = glob_pattern(tmp_path)

        # No materialization — should fall back to lazy views
        conn = get_read_connection(db_path, glob_pat)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
            assert rows is not None
            assert rows[0] == 4

            # Should be a VIEW, not a BASE TABLE
            tables = conn.execute(
                "SELECT table_type FROM information_schema.tables "
                "WHERE table_name = 'raw_messages'"
            ).fetchone()
            assert tables is not None
            assert tables[0] == "VIEW"
        finally:
            conn.close()


def test_get_read_connection_nonexistent_db():
    """get_read_connection falls back when DB file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "nonexistent" / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)

        # DB path doesn't exist — should fall back to lazy views
        conn = get_read_connection(db_path, glob_pat)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
            assert rows is not None
            assert rows[0] == 4
        finally:
            conn.close()


def test_materialize_views_drops_existing_views():
    """Regression: materialize_views must drop views before tables.

    If a name (e.g. sessions) exists as a VIEW from a previous lazy-view
    connection, DROP TABLE IF EXISTS raises CatalogException. This is the
    exact error seen in production startup.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        glob_pat = glob_pattern(tmp_path)
        db_path = tmp_path / "test.duckdb"

        conn = duckdb.connect(str(db_path))

        # Simulate a previous lazy-view session leaving views behind
        for name in ("session_titles", "raw_messages", "raw_data"):
            conn.execute(f"CREATE VIEW {name} AS SELECT 1 AS x")

        # This must not raise CatalogException
        materialize_views(conn, glob_pat, days=0)

        # Verify materialized tables exist
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_name IN ('raw_data', 'raw_messages')
        """).fetchall()
        table_names = {t[0] for t in tables}
        assert "raw_data" in table_names
        assert "raw_messages" in table_names
        conn.close()


def test_materialize_views_drops_existing_tables():
    """Regression: materialize_views must drop tables before views.

    If a name (e.g. search_corpus) exists as a TABLE, DROP VIEW IF EXISTS
    raises CatalogException. Ensure materialize_views handles pre-existing
    tables gracefully.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        glob_pat = glob_pattern(tmp_path)
        db_path = tmp_path / "test.duckdb"

        conn = duckdb.connect(str(db_path))

        # Pre-create search_corpus as a TABLE (simulates build_search_corpus)
        conn.execute("CREATE TABLE search_corpus (id INTEGER)")

        # This must not raise CatalogException
        materialize_views(conn, glob_pat, days=0)

        # Verify materialized tables exist
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_name IN ('raw_data', 'raw_messages')
        """).fetchall()
        table_names = {t[0] for t in tables}
        assert "raw_data" in table_names
        assert "raw_messages" in table_names
        conn.close()
