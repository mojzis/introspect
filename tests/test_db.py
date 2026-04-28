"""Tests for introspect database views."""

import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import pytest

from introspect.db import (
    _MAX_TOOL_RESULT_SIZE_BYTES,
    DatabaseLockedError,
    _filter_parseable_files,
    connect_writable,
    ensure_materialized,
    get_connection,
    get_read_connection,
    materialize_views,
    read_last_materialized,
)

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


_DERIVED_RELATION_NAMES = (
    "logical_sessions",
    "tool_calls",
    "conversation_turns",
    "session_titles",
    "session_stats",
)


def test_lazy_creates_views():
    """Lazy ``get_connection`` path backs derived names with VIEWs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        try:
            rows = conn.execute("""
                SELECT table_name, table_type FROM information_schema.tables
            """).fetchall()
            type_by_name = dict(rows)
            # raw_messages is also a VIEW in lazy mode.
            assert type_by_name.get("raw_messages") == "VIEW"
            for name in _DERIVED_RELATION_NAMES:
                assert type_by_name.get(name) == "VIEW", (
                    f"expected lazy path to back {name} as VIEW, got "
                    f"{type_by_name.get(name)!r}"
                )
        finally:
            conn.close()


def test_materialize_creates_tables():
    """``materialize_views`` backs the same derived names with BASE TABLEs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = duckdb.connect(str(db_path))
        try:
            materialize_views(conn, glob_pat)

            rows = conn.execute("""
                SELECT table_name, table_type FROM information_schema.tables
            """).fetchall()
            type_by_name = dict(rows)
            # raw_messages becomes a BASE TABLE under materialize_views.
            assert type_by_name.get("raw_messages") == "BASE TABLE"
            for name in _DERIVED_RELATION_NAMES:
                assert type_by_name.get(name) == "BASE TABLE", (
                    f"expected materialized path to back {name} as BASE "
                    f"TABLE, got {type_by_name.get(name)!r}"
                )
        finally:
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


_DISK_FULL_MSG = "IO Error: Disk is full"


def _raise_disk_full(*args, **kwargs):
    raise duckdb.IOException(_DISK_FULL_MSG)


def test_connect_writable_raises_when_locked(mock_locked_db):
    """connect_writable raises DatabaseLockedError when the DB is locked elsewhere."""
    db_path = Path("/tmp/fake.duckdb")

    with pytest.raises(DatabaseLockedError) as exc_info:
        connect_writable(db_path)
    assert exc_info.value.db_path == db_path
    assert str(db_path) in str(exc_info.value)
    # DatabaseLockedError subclasses duckdb.IOException for natural handling
    assert isinstance(exc_info.value, duckdb.IOException)


def test_connect_writable_passes_through_other_io_errors(monkeypatch):
    """connect_writable re-raises IOExceptions unrelated to lock conflicts."""
    monkeypatch.setattr("introspect.db.duckdb.connect", _raise_disk_full)

    with pytest.raises(duckdb.IOException) as exc_info:
        connect_writable(Path("/tmp/fake.duckdb"))
    assert not isinstance(exc_info.value, DatabaseLockedError)


def test_connect_writable_succeeds_when_unlocked():
    """connect_writable returns a live connection when no writer holds the lock."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"

        conn = connect_writable(db_path)
        try:
            row = conn.execute("SELECT 1").fetchone()
            assert row == (1,)
        finally:
            conn.close()


def test_connect_writable_detects_real_cross_process_lock():
    """Integration test: spawn a subprocess holding the DB, verify we detect it.

    DuckDB enforces its write lock across processes (not within a single
    process). This test catches the real failure mode that triggered the bug
    without relying on string-matched mocks.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        # Spawn a subprocess that opens and holds the DB
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import duckdb, sys, time;"
                    f"c = duckdb.connect({str(db_path)!r});"
                    "sys.stdout.write('ready\\n'); sys.stdout.flush();"
                    "time.sleep(30)"
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for the subprocess to acquire the lock
            assert holder.stdout is not None
            ready = holder.stdout.readline()
            assert ready.strip() == "ready"

            with pytest.raises(DatabaseLockedError) as exc_info:
                connect_writable(db_path)
            assert exc_info.value.db_path == db_path
        finally:
            holder.terminate()
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait()


def test_get_connection_raises_when_locked(mock_locked_db):
    """get_connection propagates DatabaseLockedError when the DB is locked."""
    with pytest.raises(DatabaseLockedError):
        get_connection(Path("/tmp/fake.duckdb"), "/tmp/*.jsonl")


def test_maximum_object_size_raised_above_default():
    """Default DuckDB limit is 16MB; some Claude tool results exceed it.

    Regression: a 31MB tool result aborted startup with InvalidInputException.
    """
    assert _MAX_TOOL_RESULT_SIZE_BYTES >= 64 * 1024 * 1024


def test_materialize_recovers_when_bulk_read_raises(monkeypatch, caplog):
    """A buffer-level read error should fall back to per-file load, not crash.

    Reproduces the production failure (``maximum_object_size`` exceeded) by
    making the first bulk-read raise ``InvalidInputException``. The fallback
    path probes each file individually, drops the bad one, and retries with
    the survivors so users still get the rest of their history.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        glob_pat = glob_pattern(tmp_path)

        # Force only the first ``_create_raw_tables`` call (the bulk read in
        # ``_load_raw_tables``) to raise. The retry call after filtering is
        # left to run normally so we can assert it produced real rows.
        import introspect.db as db_module  # noqa: PLC0415

        original = db_module._create_raw_tables
        calls = {"n": 0}

        boom_msg = "maximum_object_size exceeded"

        def fail_once(conn, source, day_filter, and_day_filter):
            calls["n"] += 1
            if calls["n"] == 1:
                raise duckdb.InvalidInputException(boom_msg)
            return original(conn, source, day_filter, and_day_filter)

        monkeypatch.setattr(db_module, "_create_raw_tables", fail_once)

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            with caplog.at_level("WARNING", logger="introspect.db"):
                materialize_views(conn, glob_pat, days=0, resolve_projects=False)

            session_ids = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT session_id FROM raw_messages"
                ).fetchall()
            }
            assert SID in session_ids
        finally:
            conn.close()

        # Operator-facing warning so the failure is visible in logs.
        assert any("Bulk JSONL load failed" in r.message for r in caplog.records)


def test_filter_parseable_files_keeps_good_skips_bad(caplog):
    """``_filter_parseable_files`` returns only files that probe cleanly.

    Includes a binary-garbage file alongside good and unopenable inputs.
    With ``ignore_errors=true`` DuckDB may treat the garbage file as
    "parseable" (returning NULLs); the contract is that the function never
    crashes and that hard errors (e.g. the missing file) are surfaced as
    warnings and excluded from the result.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        good_path = _write_sample_jsonl(tmp_path)

        binary_path = tmp_path / "not-json.jsonl"
        binary_path.write_bytes(b"\x00\x01\x02 not valid json at all \xff\xfe")

        missing_path = tmp_path / "missing.jsonl"

        with caplog.at_level("WARNING", logger="introspect.db"):
            result = _filter_parseable_files(
                [str(good_path), str(binary_path), str(missing_path)]
            )

        assert str(good_path) in result
        assert str(missing_path) not in result
        assert any(str(missing_path) in r.message for r in caplog.records)


def test_ensure_materialized_builds_when_db_missing():
    """ensure_materialized creates a materialized DB on first call and stamps it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "introspect.duckdb"
        glob_pat = glob_pattern(tmp_path)

        ts = ensure_materialized(db_path, glob_pat)

        assert ts is not None
        assert db_path.exists()
        with duckdb.connect(str(db_path), read_only=True) as conn:
            stamp = read_last_materialized(conn)
            assert stamp == ts
            row = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
            assert row is not None
            assert row[0] == 4


def test_ensure_materialized_reuses_existing_db():
    """ensure_materialized does not rebuild when the DB is already materialized."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "introspect.duckdb"
        glob_pat = glob_pattern(tmp_path)

        first = ensure_materialized(db_path, glob_pat)
        second = ensure_materialized(db_path, glob_pat)

        assert first is not None
        assert second == first


def test_ensure_materialized_handles_empty_glob():
    """An empty Claude home (no JSONL files) materializes empty stub tables."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "introspect.duckdb"
        glob_pat = str(tmp_path / "missing" / "**" / "*.jsonl")

        ts = ensure_materialized(db_path, glob_pat)

        assert ts is not None
        with duckdb.connect(str(db_path), read_only=True) as conn:
            for view in (
                "raw_messages",
                "raw_data",
                "logical_sessions",
                "tool_calls",
                "session_stats",
                "search_corpus",
            ):
                row = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
                assert row is not None
                assert row[0] == 0, f"{view} should be empty on no-JSONL build"


def test_empty_stub_raw_messages_columns_match_real_materialization():
    """``raw_messages`` schema must match in real and empty-stub paths.

    All derived views read from ``raw_messages``, so a missing column would
    silently break a consumer in the empty-stub case. ``raw_data`` is
    intentionally excluded — it's a ``SELECT *`` over the JSONL and its column
    set varies with whatever fields Claude Code happens to emit.
    """
    with (
        tempfile.TemporaryDirectory() as real_tmp,
        tempfile.TemporaryDirectory() as empty_tmp,
    ):
        real_path = Path(real_tmp)
        _write_sample_jsonl(real_path)
        real_db = real_path / "real.duckdb"
        real_conn = duckdb.connect(str(real_db))
        try:
            materialize_views(real_conn, glob_pattern(real_path))
        finally:
            real_conn.close()

        empty_path = Path(empty_tmp)
        empty_db = empty_path / "empty.duckdb"
        empty_conn = duckdb.connect(str(empty_db))
        try:
            materialize_views(
                empty_conn, str(empty_path / "missing" / "**" / "*.jsonl")
            )
        finally:
            empty_conn.close()

        def _columns(db: Path, table: str) -> list[str]:
            with duckdb.connect(str(db), read_only=True) as conn:
                rows = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'main' AND table_name = ? "
                    "ORDER BY column_name",
                    [table],
                ).fetchall()
            return [r[0] for r in rows]

        real_cols = set(_columns(real_db, "raw_messages"))
        empty_cols = set(_columns(empty_db, "raw_messages"))
        assert real_cols == empty_cols, (
            "raw_messages columns differ between real and empty-stub paths: "
            f"only-in-real={sorted(real_cols - empty_cols)}, "
            f"only-in-empty={sorted(empty_cols - real_cols)}"
        )
