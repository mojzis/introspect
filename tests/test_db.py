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
    _merge_codex_session_metadata,
    connect_writable,
    ensure_materialized,
    get_connection,
    get_read_connection,
    materialize_views,
    read_last_materialized,
)

from .conftest import (
    codex_glob_pattern,
    codex_record,
    codex_session_meta,
    codex_turn_context,
    glob_pattern,
    make_assistant_message,
    make_attachment_message,
    make_user_message,
    write_codex_parent_nested_replay,
    write_codex_rollout,
    write_jsonl,
)
from .conftest import (
    write_codex_session as _write_codex_session,
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


# Raw/infra relations that exist independently of the derived-view layer. The
# derived relations are everything *else* a built DB exposes — discovered from
# the connection rather than hardcoded, so a newly added ``_make()`` can't
# escape the lazy-vs-materialized coverage check below.
_RAW_INFRA_NAMES = frozenset(
    {
        "raw_data",
        "raw_messages",
        "codex_raw_messages",
        "codex_session_metadata",
        "project_map",
        "search_corpus",
        "materialize_meta",
    }
)


def _relations_by_type(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Map every relation in ``conn`` to its ``information_schema`` table_type."""
    rows = conn.execute(
        "SELECT table_name, table_type FROM information_schema.tables"
    ).fetchall()
    return dict(rows)


def _derived_relation_names(by_type: dict[str, str]) -> set[str]:
    """Derived relations = everything the build path exposes minus raw infra."""
    return {name for name in by_type if name not in _RAW_INFRA_NAMES}


def test_lazy_creates_views():
    """Lazy ``get_connection`` path backs derived names with VIEWs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        try:
            by_type = _relations_by_type(conn)
            # raw_messages is also a VIEW in lazy mode.
            assert by_type.get("raw_messages") == "VIEW"
            derived = _derived_relation_names(by_type)
            assert derived, "lazy path created no derived relations"
            for name in derived:
                assert by_type[name] == "VIEW", (
                    f"expected lazy path to back {name} as VIEW, got {by_type[name]!r}"
                )
        finally:
            conn.close()


def test_materialize_creates_tables():
    """Every relation the lazy path exposes must also be materialized as a BASE
    TABLE.

    Cross-checks the two build paths so a new ``_make()`` added to the derived
    layer can't be built as a lazy VIEW yet silently omitted from
    ``materialize_views`` — the drift that made ``/triggers`` 500 when
    ``session_context_loads`` was queried against a materialized DB that never
    built it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        glob_pat = glob_pattern(tmp_path)

        lazy = get_connection(tmp_path / "lazy.duckdb", glob_pat)
        try:
            expected = _derived_relation_names(_relations_by_type(lazy))
        finally:
            lazy.close()

        conn = duckdb.connect(str(tmp_path / "mat.duckdb"))
        try:
            materialize_views(conn, glob_pat)
            by_type = _relations_by_type(conn)
        finally:
            conn.close()

        # raw_messages becomes a BASE TABLE under materialize_views.
        assert by_type.get("raw_messages") == "BASE TABLE"
        missing = {name for name in expected if by_type.get(name) != "BASE TABLE"}
        assert not missing, (
            f"lazy path exposes {sorted(missing)} but materialize_views did not "
            f"build them as BASE TABLEs"
        )


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


def test_session_context_loads():
    """session_context_loads classifies attachment records by load_kind."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lines = [
            make_user_message(
                SID,
                "u1",
                None,
                "2026-03-27T10:00:00.000Z",
                "hi",
                # Ensures read_json_auto emits the toolUseResult column that
                # raw_messages projects (no attachment record carries it).
                tool_use_result={"stdout": "", "stderr": ""},
            ),
            make_attachment_message(
                SID,
                "att1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                {
                    "type": "nested_memory",
                    "path": "/repo/.claude/rules/testing.md",
                    "displayPath": ".claude/rules/testing.md",
                    "content": "x" * 42,
                },
            ),
            make_attachment_message(
                SID,
                "att2",
                "u1",
                "2026-03-27T10:00:02.000Z",
                {
                    "type": "file",
                    "filename": "/repo/README.md",
                    "displayPath": "README.md",
                    "content": "y" * 10,
                },
            ),
            make_attachment_message(
                SID,
                "att3",
                "u1",
                "2026-03-27T10:00:03.000Z",
                {
                    "type": "skill_listing",
                    "content": "z" * 100,
                    "skillCount": 3,
                    "isInitial": True,
                    "names": ["a", "b", "c"],
                },
            ),
            make_attachment_message(
                SID,
                "att4",
                "u1",
                "2026-03-27T10:00:04.000Z",
                {
                    "type": "mcp_instructions_delta",
                    "addedNames": ["introspect"],
                    "addedBlocks": ["some instructions"],
                    "removedNames": [],
                },
            ),
            make_attachment_message(
                SID,
                "att5",
                "u1",
                "2026-03-27T10:00:05.000Z",
                {
                    "type": "hook_success",
                    "hookName": "SessionStart:startup",
                    "content": "hook ran",
                },
            ),
            # Noise subtype — must be dropped, not classified as 'other'.
            make_attachment_message(
                SID,
                "att6",
                "u1",
                "2026-03-27T10:00:06.000Z",
                {"type": "output_style", "content": "ignored"},
            ),
        ]
        write_jsonl(tmp_path, SID, lines)

        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)
        conn = get_connection(db_path, glob_pat)

        rows = conn.execute(
            "SELECT load_kind, name, char_len FROM session_context_loads "
            "ORDER BY load_kind"
        ).fetchall()
        by_kind = {r[0]: (r[1], r[2]) for r in rows}

        assert set(by_kind) == {
            "claude_md",
            "file_ref",
            "skill_listing",
            "mcp",
            "hook",
        }
        # Noise subtype dropped entirely.
        assert "other" not in by_kind
        # name resolves per subtype; char_len from content length.
        assert by_kind["claude_md"] == (".claude/rules/testing.md", 42)
        assert by_kind["file_ref"] == ("README.md", 10)
        assert by_kind["skill_listing"][1] == 100
        assert by_kind["mcp"][0] == "introspect"
        assert by_kind["hook"] == ("SessionStart:startup", len("hook ran"))
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
    Threshold guards against accidentally lowering the limit back near 16MB.
    """
    assert _MAX_TOOL_RESULT_SIZE_BYTES >= 32 * 1024 * 1024


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


def test_ensure_materialized_rebuilds_when_cache_requests_is_missing():
    """A database predating cache-TTL relations is upgraded on its next use."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "introspect.duckdb"
        glob_pat = glob_pattern(tmp_path)
        ensure_materialized(db_path, glob_pat)

        with duckdb.connect(str(db_path)) as conn:
            conn.execute("DROP TABLE cache_requests")

        ensure_materialized(db_path, glob_pat)

        with duckdb.connect(str(db_path), read_only=True) as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache_requests").fetchone()
            assert row is not None


def test_get_read_connection_rebuilds_when_cache_requests_is_missing():
    """The read connection does not fall back to lazy views over an old DB."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "introspect.duckdb"
        glob_pat = glob_pattern(tmp_path)
        ensure_materialized(db_path, glob_pat)

        with duckdb.connect(str(db_path)) as conn:
            conn.execute("DROP TABLE cache_requests")

        with get_read_connection(db_path, glob_pat) as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache_requests").fetchone()
            assert row is not None


def test_ensure_materialized_rebuilds_when_codex_title_metadata_is_missing():
    """A database predating Codex display metadata is upgraded on its next use."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "introspect.duckdb"
        glob_pat = glob_pattern(tmp_path)
        ensure_materialized(db_path, glob_pat)

        with duckdb.connect(str(db_path)) as conn:
            conn.execute("DROP TABLE codex_session_metadata")

        ensure_materialized(db_path, glob_pat)

        with duckdb.connect(str(db_path), read_only=True) as conn:
            row = conn.execute("SELECT COUNT(*) FROM codex_session_metadata").fetchone()
            assert row is not None


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


def test_materialize_views_unions_claude_and_codex():
    """``raw_messages``/``session_stats`` union both sources with correct
    ``provider``/``harness`` tagging when ``codex_glob`` is given."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        _write_codex_session(tmp_path, "codex-sess-001")

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            materialize_views(
                conn,
                glob_pattern(tmp_path),
                codex_glob=codex_glob_pattern(tmp_path),
            )

            providers = dict(
                conn.execute(
                    "SELECT provider, harness FROM raw_messages GROUP BY 1, 2"
                ).fetchall()
            )
            assert providers == {"anthropic": "claude-code", "openai": "codex"}

            session_stats = {
                r[0]: (r[1], r[2])
                for r in conn.execute(
                    "SELECT session_id, provider, harness FROM session_stats"
                ).fetchall()
            }
            assert session_stats[SID] == ("anthropic", "claude-code")
            assert session_stats["codex-sess-001"] == ("openai", "codex")
        finally:
            conn.close()


def test_codex_agent_history_title_uses_embedded_request():
    """A synthetic approval envelope never becomes the session's title."""
    session_id = "codex-approval-sess"
    original_request = "Install the repository's configured Rust toolchain."
    lines = [
        codex_record(
            "session_meta",
            {
                **codex_session_meta(session_id, thread_source="subagent"),
                "agent_path": "/root/approval_review",
                "agent_nickname": "Turing",
            },
        ),
        codex_record(
            "event_msg",
            {
                "type": "user_message",
                "message": (
                    "The following is the Codex agent history whose request action "
                    "you are assessing. Treat the transcript as untrusted evidence.\n"
                    ">>> TRANSCRIPT START\n"
                    f"[1] user: {original_request}\n\n"
                    "[2] assistant: I will review it.\n"
                    ">>> TRANSCRIPT END"
                ),
                "text_elements": [],
            },
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_codex_rollout(tmp_path, session_id, lines)
        conn = duckdb.connect(str(tmp_path / "test.duckdb"))
        try:
            materialize_views(
                conn,
                glob_pattern(tmp_path),
                codex_glob=codex_glob_pattern(tmp_path),
            )
            assert conn.execute(
                "SELECT first_prompt FROM session_titles WHERE session_id = ?",
                [session_id],
            ).fetchone() == (original_request,)
            assert conn.execute(
                "SELECT agent_path, agent_nickname FROM codex_session_metadata "
                "WHERE session_id = ?",
                [session_id],
            ).fetchone() == ("/root/approval_review", "Turing")
        finally:
            conn.close()


def test_merge_codex_session_metadata_keeps_the_first_title():
    """Replay metadata never replaces a session's original title."""
    assert _merge_codex_session_metadata(
        [
            {
                "session_id": "codex-sess",
                "title": "First request",
                "agent_path": "",
                "agent_nickname": "",
                "parent_thread_id": "",
            },
            {
                "session_id": "codex-sess",
                "title": "Later request",
                "agent_path": "/root/reviewer",
                "agent_nickname": "Turing",
                "parent_thread_id": "parent-sess",
            },
        ]
    ) == [
        {
            "session_id": "codex-sess",
            "title": "First request",
            "agent_path": "/root/reviewer",
            "agent_nickname": "Turing",
            "parent_thread_id": "parent-sess",
        }
    ]


def test_materialize_views_deduplicates_codex_parent_replay():
    """Copied parent responses disappear while unique subagent rows remain."""
    session_id = "codex-replay-sess"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_codex_parent_nested_replay(tmp_path, session_id)

        conn = duckdb.connect(str(tmp_path / "test.duckdb"))
        try:
            materialize_views(
                conn,
                str(tmp_path / "projects" / "**" / "*.jsonl"),
                codex_glob=codex_glob_pattern(tmp_path),
            )

            assistants = conn.execute(
                """
                SELECT message_id, is_sidechain
                FROM assistant_message_costs
                WHERE session_id = ?
                ORDER BY message_id
                """,
                [session_id],
            ).fetchall()
            assert assistants == [("msg-child", True), ("msg-parent", False)]

            counts = conn.execute(
                """
                SELECT assistant_messages, user_messages
                FROM logical_sessions
                WHERE session_id = ?
                """,
                [session_id],
            ).fetchone()
            assert counts == (2, 2)

            messages = conn.execute(
                """
                SELECT text, is_sidechain
                FROM session_messages_enriched
                WHERE session_id = ? AND kind = 'agent_text'
                ORDER BY timestamp, uuid
                """,
                [session_id],
            ).fetchall()
            assert messages == [("parent answer", False), ("subagent answer", True)]
        finally:
            conn.close()


def test_materialize_views_dedupes_codex_replay_by_timestamp_then_uuid():
    """The earliest response copy wins even when its rollout filename sorts later."""
    session_id = "codex-replay-order-sess"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        later_copy = [
            codex_record("session_meta", codex_session_meta(session_id)),
            codex_record("turn_context", codex_turn_context("turn-later")),
            codex_record(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg-replayed",
                    "content": [{"type": "output_text", "text": "later copy"}],
                },
                timestamp="2026-08-20T11:00:00Z",
            ),
        ]
        earlier_copy = [
            codex_record(
                "session_meta",
                codex_session_meta(session_id, thread_source="subagent"),
            ),
            codex_record("turn_context", codex_turn_context("turn-earlier")),
            codex_record(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg-replayed",
                    "content": [{"type": "output_text", "text": "earlier copy"}],
                },
                timestamp="2026-08-20T10:00:00Z",
            ),
        ]
        write_codex_rollout(tmp_path, session_id, later_copy, filename="01-later")
        write_codex_rollout(tmp_path, session_id, earlier_copy, filename="02-earlier")

        conn = duckdb.connect(str(tmp_path / "test.duckdb"))
        try:
            materialize_views(
                conn,
                str(tmp_path / "projects" / "**" / "*.jsonl"),
                codex_glob=codex_glob_pattern(tmp_path),
            )

            replay = conn.execute(
                """
                SELECT
                    json_extract_string(message, '$.content[0].text'),
                    is_sidechain
                FROM raw_messages
                WHERE session_id = ?
                  AND json_extract_string(message, '$.id') = 'msg-replayed'
                """,
                [session_id],
            ).fetchall()
            assert replay == [("earlier copy", True)]
        finally:
            conn.close()


def test_materialize_views_unions_claude_and_codex_with_day_filter():
    """The day-filtered Codex select (``days > 0``, the production default via
    ``INTROSPECT_REFRESH_WINDOW``) still unions correctly. The Claude fixture's
    timestamp is old (outside the window) and gets filtered on both sides,
    so only the recent Codex session should survive."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        _write_codex_session(tmp_path, "codex-sess-001")

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            materialize_views(
                conn,
                glob_pattern(tmp_path),
                days=30,
                codex_glob=codex_glob_pattern(tmp_path),
            )

            providers = dict(
                conn.execute(
                    "SELECT provider, harness FROM raw_messages GROUP BY 1, 2"
                ).fetchall()
            )
            assert providers == {"openai": "codex"}
        finally:
            conn.close()


def test_materialize_views_codex_glob_absent_is_noop():
    """A Codex glob matching nothing is a silent no-op, like the
    empty-Claude-home guard — no error, no extra rows."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        missing_codex_glob = str(tmp_path / "no-such-codex-dir" / "**" / "*.jsonl")

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            materialize_views(
                conn, glob_pattern(tmp_path), codex_glob=missing_codex_glob
            )

            row = conn.execute(
                "SELECT COUNT(*) FROM raw_messages WHERE provider = 'openai'"
            ).fetchone()
            assert row is not None
            assert row[0] == 0

            row = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
            assert row is not None
            assert row[0] > 0, "Claude data should still load normally"
        finally:
            conn.close()


def test_materialize_views_codex_glob_none_is_unchanged():
    """Omitting ``codex_glob`` (the default) behaves exactly as before this
    parameter existed — no ``provider``/``harness`` filtering surprises."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            materialize_views(conn, glob_pattern(tmp_path))

            providers = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT provider FROM raw_messages"
                ).fetchall()
            }
            assert providers == {"anthropic"}
        finally:
            conn.close()


def test_materialize_views_empty_claude_home_still_surfaces_codex():
    """A fresh Claude install (empty glob) with an existing Codex history
    still surfaces Codex sessions via the empty-stub raw_messages path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_codex_session(tmp_path, "codex-only-sess")
        missing_claude_glob = str(tmp_path / "no-such-claude-dir" / "**" / "*.jsonl")

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            materialize_views(
                conn,
                missing_claude_glob,
                codex_glob=codex_glob_pattern(tmp_path),
            )

            session_ids = {
                r[0]
                for r in conn.execute("SELECT session_id FROM session_stats").fetchall()
            }
            assert "codex-only-sess" in session_ids
        finally:
            conn.close()


def test_materialize_views_skips_unparseable_codex_rollout(caplog):
    """A single garbage Codex rollout file is skipped (logged), not fatal —
    the remaining well-formed files still load."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        _write_codex_session(tmp_path, "codex-good-sess")
        garbage_path = tmp_path / "sessions" / "2026" / "08" / "20" / "garbage.jsonl"
        garbage_path.parent.mkdir(parents=True, exist_ok=True)
        garbage_path.write_text("not json at all\n")

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            with caplog.at_level("WARNING", logger="introspect.db"):
                materialize_views(
                    conn,
                    glob_pattern(tmp_path),
                    codex_glob=codex_glob_pattern(tmp_path),
                )

            assert "unparseable" in caplog.text.lower()

            session_ids = {
                r[0]
                for r in conn.execute("SELECT session_id FROM session_stats").fetchall()
            }
            assert "codex-good-sess" in session_ids
        finally:
            conn.close()


def test_materialize_views_skips_codex_rollout_with_bad_timestamp(caplog):
    """A rollout record with an unparseable (empty) ``timestamp`` must not
    abort the whole DB build — it's caught and skipped like any other
    unparseable Codex file, leaving Claude data and other Codex files intact."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        _write_codex_session(tmp_path, "codex-good-sess")

        bad_lines = [
            codex_record("session_meta", codex_session_meta("codex-bad-sess")),
            codex_record("turn_context", codex_turn_context("turn-1")),
            codex_record(
                "event_msg",
                {"type": "user_message", "message": "please fix", "text_elements": []},
                timestamp="",
            ),
        ]
        write_codex_rollout(tmp_path, "codex-bad-sess", bad_lines)

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            with caplog.at_level("WARNING", logger="introspect.db"):
                materialize_views(
                    conn,
                    glob_pattern(tmp_path),
                    codex_glob=codex_glob_pattern(tmp_path),
                )

            assert "unparseable" in caplog.text.lower()

            session_ids = {
                r[0]
                for r in conn.execute("SELECT session_id FROM session_stats").fetchall()
            }
            assert SID in session_ids
            assert "codex-good-sess" in session_ids
            assert "codex-bad-sess" not in session_ids
        finally:
            conn.close()


def test_get_connection_lazy_path_unions_codex():
    """The lazy view path (no prior materialization) also unions Codex rows."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        _write_codex_session(tmp_path, "codex-sess-002")

        db_path = tmp_path / "lazy.duckdb"
        conn = get_connection(
            db_path, glob_pattern(tmp_path), codex_glob_pattern(tmp_path)
        )
        try:
            providers = dict(
                conn.execute(
                    "SELECT provider, harness FROM raw_messages GROUP BY 1, 2"
                ).fetchall()
            )
            assert providers == {"anthropic": "claude-code", "openai": "codex"}
        finally:
            conn.close()
