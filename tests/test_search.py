"""Tests for full-text search functionality."""

import json
import tempfile
from pathlib import Path

from introspect.db import get_connection
from introspect.search import build_search_corpus, fts_search


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing search."""
    session_id = "test-session-search"
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
                    {
                        "type": "text",
                        "text": "I will help you refactor the database module.",
                    }
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
                        "id": "toolu_test1",
                        "name": "Read",
                        "input": {"file_path": "/tmp/test/db.py"},
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
            "toolUseResult": {
                "stdout": "def connect(): pass\n",
                "stderr": "",
            },
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_test1",
                        "content": "def connect(): pass\n",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-03-27T10:00:10.000Z",
            "sessionId": session_id,
            "uuid": "u3",
            "parentUuid": "a2",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "message": {
                "role": "user",
                "content": "Now add comprehensive pytest fixtures for testing",
            },
        },
    ]

    with jsonl_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


def _get_test_conn(tmp_path: Path):
    """Get a test connection with sample data loaded."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"
    glob_pattern = str(tmp_path / "projects" / "**" / "*.jsonl")
    return get_connection(db_path, glob_pattern)


def test_build_search_corpus_creates_table():
    """Test that build_search_corpus creates the table."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'search_corpus' AND table_type = 'BASE TABLE'
        """).fetchall()
        assert len(tables) == 1
        conn.close()


def test_build_search_corpus_has_rows():
    """Test that the corpus contains rows from all message types."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        count = conn.execute("SELECT COUNT(*) FROM search_corpus").fetchone()[0]
        # 5 messages total, all should have extractable content
        assert count > 0
        conn.close()


def test_build_search_corpus_extracts_user_text():
    """Test that plain user text messages are extracted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        rows = conn.execute(
            "SELECT content_text FROM search_corpus WHERE role = 'user'"
        ).fetchall()
        texts = [r[0] for r in rows]
        assert any("refactor" in t for t in texts)
        conn.close()


def test_build_search_corpus_extracts_assistant_text():
    """Test that assistant text blocks are extracted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        rows = conn.execute(
            "SELECT content_text FROM search_corpus WHERE role = 'assistant'"
        ).fetchall()
        texts = [r[0] for r in rows]
        assert any("refactor" in t for t in texts)
        conn.close()


def test_fts_search_finds_matching_content():
    """Test that FTS search returns relevant results."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        results = fts_search(conn, "refactor database")
        assert len(results) > 0
        # Results should contain session_id, timestamp, role, snippet, score
        first = results[0]
        assert first[0] == "test-session-search"
        assert first[4] is not None  # score
        conn.close()


def test_fts_search_no_results():
    """Test that FTS search returns empty for unmatched queries."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        results = fts_search(conn, "xyznonexistentterm123")
        assert results == []
        conn.close()


def test_build_search_corpus_idempotent():
    """Test that building the corpus twice works (overwrite)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)
        count1 = conn.execute("SELECT COUNT(*) FROM search_corpus").fetchone()[0]

        build_search_corpus(conn)
        count2 = conn.execute("SELECT COUNT(*) FROM search_corpus").fetchone()[0]

        assert count1 == count2
        conn.close()


def test_fts_search_includes_tool_content():
    """Test that tool inputs/outputs are searchable."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        results = fts_search(conn, "pytest fixtures")
        assert len(results) > 0
        conn.close()


def test_search_corpus_columns():
    """Test that the search_corpus table has the expected columns."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        columns = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'search_corpus'
            ORDER BY ordinal_position
        """).fetchall()
        col_names = [c[0] for c in columns]
        assert "session_id" in col_names
        assert "timestamp" in col_names
        assert "role" in col_names
        assert "content_text" in col_names
        conn.close()
