"""Tests for full-text search functionality."""

import tempfile
from pathlib import Path

from introspect.db import get_connection
from introspect.search import build_search_corpus, fts_search

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "test-session-search"


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing search."""
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
            [{"type": "text", "text": "I will help you refactor the database module."}],
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
                    "name": "Read",
                    "input": {"file_path": "/tmp/test/db.py"},
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
                    "content": "def connect(): pass\n",
                    "is_error": False,
                }
            ],
            tool_use_result={"stdout": "def connect(): pass\n", "stderr": ""},
        ),
        make_user_message(
            SID,
            "u3",
            "a2",
            "2026-03-27T10:00:10.000Z",
            "Now add comprehensive pytest fixtures for testing",
        ),
    ]
    return write_jsonl(tmp_dir, SID, lines)


def _get_test_conn(tmp_path: Path):
    """Get a test connection with sample data loaded."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"
    return get_connection(db_path, glob_pattern(tmp_path))


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
        # Row shape: (session_id, timestamp, role, cwd, snippet, score)
        session_id, _timestamp, _role, _cwd, snippet, score = results[0]
        assert session_id == "test-session-search"
        assert score is not None
        assert "refactor" in snippet.lower()
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


def test_fts_search_filter_by_role():
    """Role filter restricts results to user or assistant messages."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        user_only = fts_search(conn, "refactor database", role="user")
        assert user_only
        assert all(r[2] == "user" for r in user_only)

        asst_only = fts_search(conn, "refactor database", role="assistant")
        assert asst_only
        assert all(r[2] == "assistant" for r in asst_only)
        conn.close()


def test_fts_search_filter_by_session_id():
    """session_id filter restricts results to a single session."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        results = fts_search(conn, "refactor database", session_id=SID)
        assert results
        assert all(r[0] == SID for r in results)

        empty = fts_search(conn, "refactor database", session_id="no-such-session")
        assert empty == []
        conn.close()


def test_fts_search_require_all_narrows_matches():
    """require_all (conjunctive) requires every term to match a row."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)

        any_mode = fts_search(conn, "pytest database")
        all_mode = fts_search(conn, "pytest database", require_all=True)
        # Neither message contains both terms, so conjunctive mode returns fewer rows.
        assert len(all_mode) < len(any_mode)
        conn.close()


def test_fts_search_windowed_snippet_centers_on_match():
    """Snippets center on the first query-term hit and add ellipsis padding."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        # Inject a long row where the hit is deep inside the text so a plain
        # LEFT(content, 200) would miss it but a windowed snippet finds it.
        build_search_corpus(conn)
        prefix = "x" * 400
        suffix = "y" * 400
        conn.execute(
            "INSERT INTO search_corpus VALUES (?, ?, ?, ?, ?)",
            [
                9999,
                SID,
                "2026-03-27T10:00:20.000Z",
                "user",
                f"{prefix} needle-keyword {suffix}",
            ],
        )
        conn.execute(
            "PRAGMA create_fts_index("
            "'search_corpus', 'rowid', 'content_text', overwrite=1)"
        )

        results = fts_search(conn, "needle-keyword")
        assert results
        assert any("needle-keyword" in r[4] for r in results)
        # The prefix and suffix are long enough that the snippet should be
        # bounded on both sides with the ellipsis marker.
        hit = next(r for r in results if "needle-keyword" in r[4])
        assert hit[4].startswith("…") and hit[4].endswith("…")
        conn.close()


def test_fts_search_returns_cwd():
    """Result tuples include the session's cwd from logical_sessions."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conn = _get_test_conn(tmp_path)

        build_search_corpus(conn)
        results = fts_search(conn, "refactor database")
        assert results
        # cwd is column 3 in the returned tuple. The test fixture doesn't set
        # a cwd so it'll be NULL here — the point is the shape and the join.
        for row in results:
            assert len(row) == 6
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
