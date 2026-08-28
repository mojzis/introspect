"""Tests for MCP tool functions."""

import asyncio
import tempfile
import types
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from introspect.db import connect_read_hardened, materialize_views
from introspect.mcp import refresh_bridge
from introspect.mcp.server import create_mcp_server
from introspect.mcp.tools import (
    _SQL_CELL_MAX,
    _SQL_ROW_CAP,
    _ModelSpend,
    _render_cost_lines,
    cache_ttl_choice,
    describe_schema,
    expensive_sessions,
    get_session,
    recent_sessions,
    refresh_data,
    run_sql,
    search_conversations,
    tool_failures,
)
from introspect.refresh import LoadingPhase, LoadingState, RefreshTarget
from introspect.search import build_search_corpus
from introspect.sql_query import CELL_TRUNCATION_MARKER, MCP_SQL_CELL_CAP

from .conftest import (
    TTL_T0,
    glob_pattern,
    make_assistant_message,
    make_user_message,
    nested_type_sql,
    ttl_turn,
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
            mock_conn.return_value = connect_read_hardened(db_path)
            result = recent_sessions(n=10)

        assert "test-session-mcp" in result
        assert "main" in result


def test_get_session():
    """get_session returns full session content."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = get_session("test-session-mcp")

        assert "Session: test-session-mcp" in result
        assert "Messages" in result


def test_get_session_not_found():
    """get_session returns not-found message for missing session."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = get_session("nonexistent-session")

        assert "not found" in result


COST_SID = "test-session-cost"

# The JSONL loader infers its column set from the file, and ``raw_messages``
# selects ``toolUseResult`` — so every fixture file needs at least one record
# carrying it, even when the test doesn't care about tool results.
_TOOL_RESULT_STUB = {"stdout": "", "stderr": ""}


def _cost_report(lines: list[dict], session_id: str = COST_SID) -> str:
    """Materialize ``lines`` into a throwaway DB and return get_session's output."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_jsonl(tmp_path, session_id, lines)
        db_path = tmp_path / "cost.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            return get_session(session_id)


def _opening_user_message(session_id: str = COST_SID) -> dict:
    return make_user_message(
        session_id,
        "u1",
        None,
        "2026-03-27T10:00:00.000Z",
        "hi",
        tool_use_result=_TOOL_RESULT_STUB,
    )


def test_get_session_token_and_cost_breakdown():
    """get_session reports tokens, $ cost split, requests and per-model spend."""
    result = _cost_report(
        [
            _opening_user_message(),
            make_assistant_message(
                COST_SID,
                "a1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                [{"type": "text", "text": "hello"}],
                model="claude-opus-4-6",
                msg_id="msg-cost-1",
                usage={
                    "input_tokens": 1_000,
                    "output_tokens": 2_000,
                    "cache_read_input_tokens": 100_000,
                    "cache_creation_input_tokens": 10_000,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 10_000,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            ),
            make_assistant_message(
                COST_SID,
                "a2",
                "a1",
                "2026-03-27T10:00:02.000Z",
                [{"type": "text", "text": "sub"}],
                model="claude-haiku-4-5",
                msg_id="msg-cost-2",
                usage={
                    "input_tokens": 500,
                    "output_tokens": 1_000,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 4_000,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 4_000,
                    },
                },
                is_sidechain=True,
            ),
        ]
    )

    assert "--- Tokens & cost ---" in result
    # opus-4-6: 1k in @ $5 + 2k out @ $25 + 100k read @ $0.50 + 10k 5m write
    # @ $6.25 = 0.005 + 0.05 + 0.05 + 0.0625 = $0.1675
    # haiku-4-5: 0.5k in @ $1 + 1k out @ $5 + 4k 1h write @ $2 = $0.0135
    assert "Cost: $0.18" in result
    assert "output $0.06" in result
    assert "cache read $0.05" in result
    assert "Tokens: 118,500 total" in result
    assert "input 1,500" in result
    assert "output 3,000" in result
    assert "cache read 100,000" in result
    assert "cache write 14,000" in result
    assert "[5m 10,000 / 1h 4,000]" in result
    assert "API requests: 2 (main 1 · subagent 1)" in result
    assert "claude-opus-4-6 $0.17/1 req" in result
    assert "claude-haiku-4-5 $0.01/1 req" in result


def test_get_session_bills_legacy_cache_creation_per_row():
    """A legacy usage record (no 5m/1h split) bills at the 5m write rate.

    The fallback is applied per row, so a model mixing a modern record with a
    legacy one still bills the legacy tokens — the failure mode of folding it
    into a per-model aggregate instead.
    """
    result = _cost_report(
        [
            _opening_user_message(),
            make_assistant_message(
                COST_SID,
                "a1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                [{"type": "text", "text": "modern"}],
                model="claude-opus-4-6",
                msg_id="msg-modern",
                usage={
                    "cache_creation_input_tokens": 100,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 100,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            ),
            make_assistant_message(
                COST_SID,
                "a2",
                "a1",
                "2026-03-27T10:00:02.000Z",
                [{"type": "text", "text": "legacy"}],
                model="claude-opus-4-6",
                msg_id="msg-legacy",
                # Pre-TTL-split schema: a total with no per-tier breakdown.
                usage={"cache_creation_input_tokens": 1_000_000},
            ),
        ]
    )

    # (100 + 1,000,000) tokens @ $6.25/M = $6.25 — not $0.000625.
    assert "Cost: $6.25" in result
    assert "cache write 1,000,100" in result
    assert "[5m 1,000,100 / 1h 0]" in result


def test_get_session_excludes_synthetic_messages():
    """``<synthetic>`` placeholders are local, not API calls — not counted."""
    result = _cost_report(
        [
            _opening_user_message(),
            make_assistant_message(
                COST_SID,
                "a1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                [{"type": "text", "text": "hello"}],
                model="claude-opus-4-6",
                msg_id="msg-real",
                usage={"input_tokens": 1_000, "output_tokens": 2_000},
            ),
            make_assistant_message(
                COST_SID,
                "a2",
                "a1",
                "2026-03-27T10:00:02.000Z",
                [{"type": "text", "text": "[Request interrupted]"}],
                model="<synthetic>",
                msg_id="msg-synthetic",
                usage={"input_tokens": 0, "output_tokens": 0},
            ),
        ]
    )

    assert "API requests: 1 (main 1 · subagent 0)" in result
    assert "synthetic" not in result.split("--- Messages ---")[0]
    # Single real model, nothing unpriced — no by-model line needed.
    assert "By model:" not in result


def test_get_session_flags_unpriced_model():
    """An unknown model is marked so its $0 doesn't read as a cheap session."""
    result = _cost_report(
        [
            _opening_user_message(),
            make_assistant_message(
                COST_SID,
                "a1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                [{"type": "text", "text": "hello"}],
                model="some-future-model-9",
                msg_id="msg-unpriced-1",
                usage={"input_tokens": 1_000, "output_tokens": 2_000},
            ),
        ]
    )

    assert "Cost: $0.00" in result
    assert "some-future-model-9 $0.00/1 req [unpriced]" in result


def test_get_session_omits_cost_block_without_usage():
    """Assistant messages without ``usage`` print no cost block, not zeros."""
    result = _cost_report(
        [
            _opening_user_message("no-usage"),
            make_assistant_message(
                "no-usage",
                "a1",
                "u1",
                "2026-03-27T10:00:01.000Z",
                [{"type": "text", "text": "hello"}],
                msg_id="msg-no-usage-1",
            ),
        ],
        session_id="no-usage",
    )

    assert "--- Tokens & cost ---" not in result
    assert "--- Messages ---" in result


def test_get_session_survives_a_missing_cost_view():
    """A DB without the derived view still returns metadata and messages."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)
        conn = duckdb.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS assistant_message_costs")
        conn.execute("DROP VIEW IF EXISTS assistant_message_costs")
        conn.close()

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = get_session(SID)

    assert f"Session: {SID}" in result
    assert "--- Tokens & cost ---" not in result
    assert "--- Messages ---" in result


def test_render_cost_lines_totals_across_models():
    """The rollup sums every model's counters into the header lines."""
    lines = _render_cost_lines(
        [
            _ModelSpend(
                model="claude-opus-5",
                requests=3,
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=30,
                cache_write_5m=40,
                output_usd=1.5,
            ),
            _ModelSpend(
                model="claude-haiku-4-5",
                requests=2,
                sidechain_requests=2,
                input_tokens=5,
                cache_write_1h=5,
                input_usd=0.25,
            ),
        ]
    )
    body = "\n".join(lines)

    assert "Cost: $1.75  (input $0.25 · output $1.50" in body
    assert "Tokens: 110 total  (input 15 · output 20 · cache read 30" in body
    assert "cache write 45 [5m 40 / 1h 5]" in body
    assert "API requests: 5 (main 3 · subagent 2)" in body
    assert "By model: claude-opus-5 $1.50/3 req · claude-haiku-4-5 $0.25/2 req" in body


def test_render_cost_lines_marks_sub_cent_costs():
    """A real but sub-cent cost is distinguishable from an unpriced $0.00."""
    lines = _render_cost_lines(
        [
            _ModelSpend(
                model="claude-opus-5", requests=1, output_tokens=100, output_usd=0.004
            )
        ]
    )

    assert "Cost: <$0.01" in "\n".join(lines)


def test_render_cost_lines_empty_without_spend():
    """No models, or models with no tokens, render nothing."""
    assert _render_cost_lines([]) == []
    assert _render_cost_lines([_ModelSpend(model="claude-opus-5", requests=1)]) == []


def test_tool_failures():
    """tool_failures returns failed tool calls."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = tool_failures()

        assert "Bash" in result
        assert "test-session-mcp" in result


def test_tool_failures_with_prefix():
    """tool_failures filters by command prefix."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = tool_failures(command_prefix="Bash")

        assert "Bash" in result

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = tool_failures(command_prefix="NonExistent")

        assert "No failed tool calls found" in result


def test_search_conversations():
    """search_conversations returns matching results."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = search_conversations("refactor database")

        assert "test-session-mcp" in result


def test_search_conversations_no_results():
    """search_conversations returns message when nothing matches."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_test_data(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = search_conversations("xyznonexistentterm123")

        assert "No results found" in result


@pytest.fixture
def patched_mcp_db() -> Iterator[None]:
    """Materialize sample data and patch the MCP tool DB handles to use it.

    Patches both ``get_read_connection`` (used by the parameterized tools)
    and ``DEFAULT_DB_PATH`` (which ``run_sql`` opens directly via
    ``connect_read_hardened``). Both handles must go through the same factory
    — DuckDB refuses a second connection to a file whose instance was opened
    with a different configuration.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _materialize_test_data(Path(tmp))
        with (
            patch("introspect.mcp.tools.get_read_connection") as mock_conn,
            patch("introspect.mcp.tools.DEFAULT_DB_PATH", db_path),
        ):
            mock_conn.return_value = connect_read_hardened(db_path)
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
    result = run_sql(
        "WITH s AS (SELECT session_id FROM logical_sessions) SELECT * FROM s"
    )

    assert "test-session-mcp" in result


def test_run_sql_rejects_write_statement(patched_mcp_db: None):
    """run_sql blocks non-SELECT statements at the tool layer."""
    result = run_sql("DELETE FROM logical_sessions")

    assert "Error" in result
    assert "DELETE" in result


def test_run_sql_rejects_attach(patched_mcp_db: None):
    """run_sql rejects a single ATTACH statement by statement type."""
    result = run_sql("ATTACH 'evil.db' AS evil")

    assert "Error" in result
    assert "ATTACH" in result


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


def test_run_sql_keeps_the_cell_truncation_marker(patched_mcp_db: None):
    """The clip marker must survive rendering, not be re-clipped away.

    ``execute_bounded`` appends CELL_TRUNCATION_MARKER at ``cell_cap``; if the
    formatter's own ceiling were also ``cell_cap`` it would chop every marked
    cell back down and replace the marker with a bare ellipsis.
    """
    result = run_sql(f"SELECT repeat('x', {MCP_SQL_CELL_CAP * 3}) AS big")

    assert CELL_TRUNCATION_MARKER in result


def test_run_sql_reports_truncation(patched_mcp_db: None):
    """A row-capped result says so; one that fits does not."""
    truncated = run_sql("SELECT * FROM range(0, 50) AS t(n)", limit=5)
    assert "truncated: hit the row cap (5)" in truncated

    complete = run_sql("SELECT * FROM range(0, 3) AS t(n)", limit=5)
    assert "truncated" not in complete


# LIST, STRUCT, MAP, nested LIST and BLOB each hide a payload far wider than
# the 200-character MCP cell cap inside an object with no visible width. The
# payload is smaller than the HTTP guard's because this cap is 20x tighter.
_NESTED_TYPE_PROBES = nested_type_sql(20_000)


@pytest.mark.parametrize(
    "sql",
    [sql for _, sql in _NESTED_TYPE_PROBES],
    ids=[label for label, _ in _NESTED_TYPE_PROBES],
)
def test_run_sql_clips_nested_and_binary_cells(patched_mcp_db: None, sql: str):
    """Regression guard: run_sql output stays small for LIST/STRUCT/MAP/BLOB.

    The formatter clipped these before ``execute_bounded`` did, so this passed
    already; it stays here so a future change to either clip cannot let a
    megabyte-wide cell through into a conversation.
    """
    result = run_sql(sql)

    assert len(result) < 4 * _SQL_CELL_MAX
    assert CELL_TRUNCATION_MARKER in result
    assert "truncated: hit the cell cap" in result


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


@pytest.fixture
def clear_refresh_bridge() -> Iterator[None]:
    """Ensure tests don't leak app state into the module-level bridge."""
    refresh_bridge.set_state(None)
    try:
        yield
    finally:
        refresh_bridge.set_state(None)


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_no_server():
    """refresh_data reports unavailable when no FastAPI lifespan registered state."""
    result = asyncio.run(refresh_data())

    assert "unavailable" in result.lower()


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_disabled():
    """refresh_data reports disabled when the trigger is None."""

    async def go() -> str:
        fake_state = types.SimpleNamespace(
            refresh_trigger=None,
            refresh_in_progress=False,
            last_refreshed_at=None,
        )
        refresh_bridge.set_state(fake_state)
        return await refresh_data()

    result = asyncio.run(go())

    assert "disabled" in result.lower()


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_no_change_needed():
    """When the loop wakes but mtimes are unchanged, returns no-op message."""

    async def go() -> tuple[str, asyncio.Event]:
        trigger = asyncio.Event()
        fake_state = types.SimpleNamespace(
            refresh_trigger=trigger,
            refresh_in_progress=False,
            last_refreshed_at=datetime.now(UTC),
        )
        refresh_bridge.set_state(fake_state)
        result = await refresh_data()
        return result, trigger

    result, trigger = asyncio.run(go())

    assert "No refresh needed" in result
    # The tool must have set the trigger so the background loop wakes early.
    assert trigger.is_set()


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_completes():
    """refresh_data waits for the background rebuild and reports completion."""

    async def go() -> str:
        trigger = asyncio.Event()
        fake_state = types.SimpleNamespace(
            refresh_trigger=trigger,
            refresh_in_progress=False,
            last_refreshed_at=datetime.now(UTC) - timedelta(hours=1),
            refresh_target=RefreshTarget("30", 30),
            database_label="warm snapshot",
            loading_state=LoadingState(
                LoadingPhase.LOADING,
                RefreshTarget("30", 30),
            ),
        )
        refresh_bridge.set_state(fake_state)

        async def simulated_loop() -> None:
            await trigger.wait()
            trigger.clear()
            fake_state.refresh_in_progress = True
            await asyncio.sleep(0.1)
            fake_state.last_refreshed_at = datetime.now(UTC)
            fake_state.refresh_in_progress = False
            fake_state.loading_state = LoadingState(
                LoadingPhase.READY,
                fake_state.refresh_target,
            )

        loop_task = asyncio.create_task(simulated_loop())
        try:
            return await refresh_data()
        finally:
            await loop_task

    result = asyncio.run(go())

    assert "Refresh complete" in result
    assert "phase=ready" in result
    assert "target=30 (30 days)" in result


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_accepts_custom_target_and_reports_contract():
    """MCP target selection uses the same contract as the web control."""

    async def go() -> str:
        trigger = asyncio.Event()
        target = RefreshTarget("14", 14)
        fake_state = types.SimpleNamespace(
            refresh_trigger=trigger,
            refresh_in_progress=False,
            last_refreshed_at=datetime.now(UTC) - timedelta(hours=1),
            refresh_target=target,
            database_label="authoritative",
            loading_state=LoadingState(LoadingPhase.LOADING, target),
        )
        refresh_bridge.set_state(fake_state)

        async def simulated_loop() -> None:
            await trigger.wait()
            trigger.clear()
            fake_state.refresh_in_progress = True
            await asyncio.sleep(0.01)
            fake_state.last_refreshed_at = datetime.now(UTC)
            fake_state.loading_state = LoadingState(
                LoadingPhase.READY,
                fake_state.refresh_target,
            )
            fake_state.refresh_in_progress = False

        loop_task = asyncio.create_task(simulated_loop())
        try:
            return await refresh_data("14")
        finally:
            await loop_task

    result = asyncio.run(go())

    assert "Refresh complete" in result
    assert "phase=ready" in result
    assert "target=14 (14 days)" in result


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_rejects_invalid_target():
    async def go() -> str:
        fake_state = types.SimpleNamespace(refresh_trigger=asyncio.Event())
        refresh_bridge.set_state(fake_state)
        return await refresh_data("two-weeks")

    result = asyncio.run(go())

    assert "Invalid refresh target" in result


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_data_still_running(monkeypatch: pytest.MonkeyPatch):
    """When the rebuild exceeds the wait budget, report ``still running``."""
    # Squeeze the finish budget so the test doesn't actually wait 30 seconds.
    monkeypatch.setattr("introspect.mcp.tools.REFRESH_TIMEOUT", 0.2)

    async def go() -> str:
        trigger = asyncio.Event()
        fake_state = types.SimpleNamespace(
            refresh_trigger=trigger,
            refresh_in_progress=False,
            last_refreshed_at=datetime.now(UTC) - timedelta(hours=1),
        )
        refresh_bridge.set_state(fake_state)

        async def simulated_slow_loop() -> None:
            await trigger.wait()
            trigger.clear()
            fake_state.refresh_in_progress = True
            # Stay "in progress" longer than the squeezed finish budget so
            # `refresh_data` gives up and reports STILL_RUNNING.
            await asyncio.sleep(1.0)
            fake_state.refresh_in_progress = False

        loop_task = asyncio.create_task(simulated_slow_loop())
        try:
            return await refresh_data()
        finally:
            await loop_task

    result = asyncio.run(go())

    assert "still running" in result.lower()


@pytest.mark.usefixtures("clear_refresh_bridge")
def test_refresh_bridge_rejects_double_registration():
    """Registering a second non-None state without an intervening clear raises."""
    fake_state = types.SimpleNamespace(
        refresh_trigger=None,
        refresh_in_progress=False,
        last_refreshed_at=None,
    )
    refresh_bridge.set_state(fake_state)

    other_state = types.SimpleNamespace(
        refresh_trigger=None,
        refresh_in_progress=False,
        last_refreshed_at=None,
    )
    with pytest.raises(RuntimeError, match="already has a registered state"):
        refresh_bridge.set_state(other_state)

    # Clearing first allows re-registration.
    refresh_bridge.set_state(None)
    refresh_bridge.set_state(other_state)


def test_server_instructions_mention_key_views():
    """The MCP server ships instructions orienting external clients."""
    instructions = create_mcp_server().instructions
    assert instructions is not None
    assert "session_stats" in instructions
    assert "describe_schema" in instructions


# ---------------------------------------------------------------------------
# cache_ttl_choice tests
# ---------------------------------------------------------------------------


def _ttl_db(tmp_path: Path, session_id: str, lines: list[dict]) -> Path:
    """Materialize one hand-built session for the TTL tool."""
    write_jsonl(tmp_path, session_id, lines)
    db_path = tmp_path / "ttl.duckdb"
    conn = duckdb.connect(str(db_path))
    materialize_views(conn, glob_pattern(tmp_path), resolve_projects=False)
    conn.close()
    return db_path


def _run_cache_ttl_choice(db_path: Path, **kwargs) -> str:
    with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
        mock_conn.return_value = connect_read_hardened(db_path)
        return cache_ttl_choice(**kwargs)


def test_cache_ttl_choice_recommends_5m_when_nothing_pauses():
    """No gaps → 1h's 2x write surcharge buys nothing, and it says so."""
    sid = "aaaaaaaa-0000-0000-0000-00000000ttl1".replace("ttl1", "0001")
    lines = ttl_turn(sid, 1, TTL_T0, read=0, create=40_000)
    for n in range(2, 5):
        lines += ttl_turn(
            sid,
            n,
            TTL_T0 + timedelta(seconds=15 * (n - 1)),
            read=40_000 * (n - 1),
            create=40_000,
        )
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _ttl_db(Path(tmp), sid, lines)
        result = _run_cache_ttl_choice(db_path)

    assert "5m saves" in result
    assert "0 recoverable gap(s)" in result
    assert "currently billed at 5m" in result


def test_cache_ttl_choice_recommends_1h_when_pauses_dominate():
    """20-minute pauses over a large prefix are what a 1h TTL is for."""
    sid = "bbbbbbbb-0000-0000-0000-000000000002"
    lines = ttl_turn(sid, 1, TTL_T0, read=0, create=200_000)
    for n in range(2, 6):
        lines += ttl_turn(
            sid,
            n,
            TTL_T0 + timedelta(minutes=20 * (n - 1)),
            read=0,
            create=200_000 * n,
        )
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _ttl_db(Path(tmp), sid, lines)
        result = _run_cache_ttl_choice(db_path)

    assert "1h saves" in result
    assert "recoverable gap(s)" in result


def test_cache_ttl_choice_reports_a_thin_margin_as_undecided():
    """An output-dominated session: the cache policy barely moves the bill.

    The delta still has a sign, but it is a rounding error next to the
    generation cost — reporting it as a recommendation would be dressing
    modelling noise up as a decision.
    """
    sid = "cccccccc-0000-0000-0000-000000000003"
    lines = ttl_turn(sid, 1, TTL_T0, read=0, create=1_000, inp=1, out=200_000)
    lines += ttl_turn(
        sid,
        2,
        TTL_T0 + timedelta(minutes=20),
        read=0,
        create=2_000,
        inp=1,
        out=200_000,
    )
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _ttl_db(Path(tmp), sid, lines)
        result = _run_cache_ttl_choice(db_path)

    assert "within noise" in result
    assert "saves" not in result


def test_cache_ttl_choice_reports_no_data():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _materialize_test_data(Path(tmp))
        result = _run_cache_ttl_choice(db_path, sidechain=True)

    assert result == "No cache data in range."


def test_cache_ttl_choice_rejects_a_malformed_since():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _materialize_test_data(Path(tmp))
        result = _run_cache_ttl_choice(db_path, since="not-a-date")

    assert result.startswith("Error: invalid 'since'")


def test_register_tools_adds_cache_ttl_choice_once():
    """The deterministic-template adapter wires it exactly once."""
    tools = asyncio.run(create_mcp_server().list_tools())
    names = [t.name for t in tools]

    assert names.count("cache_ttl_choice") == 1


def test_register_tools_adds_tool_failure_rate_without_shadowing_expensive_sessions():
    """register_tools wires `tool_failure_rate` as a deterministic-template
    adapter exactly once, and does not double-register or shadow the
    existing richer `expensive_sessions` tool with a generated passthrough."""
    tools = asyncio.run(create_mcp_server().list_tools())
    names = [t.name for t in tools]

    assert names.count("tool_failure_rate") == 1
    assert names.count("expensive_sessions") == 1

    expensive_tool = next(t for t in tools if t.name == "expensive_sessions")
    assert expensive_tool.description is not None
    assert "Pareto" in expensive_tool.description


# ---------------------------------------------------------------------------
# expensive_sessions tests
# ---------------------------------------------------------------------------

# Session IDs for expensive_sessions tests — distinct per test to avoid
# accidental state sharing.
_SID_CHEAP = "sess-expensive-cheap-aaa-aaaaaaaaa"
_SID_MID = "sess-expensive-mid---aaa-aaaaaaaaa"
_SID_PRICEY = "sess-expensive-pricey-aa-aaaaaaaaa"


def _make_cost_session(
    tmp_dir: Path,
    session_id: str,
    input_tokens: int,
    timestamp_day: str = "2026-05-01",
    timestamp_hour: str = "10",
    *,
    msg_id_suffix: str = "",
    is_sidechain: bool = False,
    cache_creation_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Write a JSONL for a session with known cost at claude-opus-4-7 pricing.

    claude-opus-4-7 input rate is $5/M, so ``input_tokens=1_000_000`` → $5.
    Each session uses a unique msg_id via ``msg_id_suffix`` to avoid dedup.
    The user message carries ``tool_use_result`` so union_by_name picks up
    the column (mirrors cost_helpers._session_at_cost pattern).
    """
    sid_short = session_id[:8]
    usage: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_creation": {
            "ephemeral_5m_input_tokens": cache_creation_tokens,
            "ephemeral_1h_input_tokens": 0,
        },
    }
    lines = [
        make_user_message(
            session_id,
            "u1",
            None,
            f"{timestamp_day}T{timestamp_hour}:00:00.000Z",
            "go",
            tool_use_result={"content": "seed"},
        ),
        make_assistant_message(
            session_id,
            "a1",
            "u1",
            f"{timestamp_day}T{timestamp_hour}:00:01.000Z",
            [{"type": "text", "text": "ok"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_short}{msg_id_suffix}-a1",
            usage=usage,
            is_sidechain=is_sidechain,
        ),
        # Second message for shape tests (distinct msg_id)
        make_assistant_message(
            session_id,
            "a2",
            "a1",
            f"{timestamp_day}T{timestamp_hour}:30:00.000Z",
            [{"type": "text", "text": "done"}],
            model="claude-opus-4-7",
            msg_id=f"msg-{sid_short}{msg_id_suffix}-a2",
            usage={"input_tokens": 0, "output_tokens": output_tokens},
        ),
    ]
    write_jsonl(tmp_dir, session_id, lines)


def _materialize_expensive_db(tmp_path: Path) -> Path:
    """Write three sessions with known costs and return the DB path.

    Session costs at $5/M claude-opus-4-7 input:
      _SID_PRICEY → 4M tokens → $20.00
      _SID_MID    → 2M tokens → $10.00
      _SID_CHEAP  →  1M tokens → $5.00
    Total = $35.00.  Pareto 80% = $28.00; both _SID_PRICEY and _SID_MID
    are needed (20+10=30 → first two cover 85.7%).
    """
    _make_cost_session(tmp_path, _SID_PRICEY, 4_000_000, msg_id_suffix="-p")
    _make_cost_session(tmp_path, _SID_MID, 2_000_000, msg_id_suffix="-m")
    _make_cost_session(tmp_path, _SID_CHEAP, 1_000_000, msg_id_suffix="-c")
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    materialize_views(conn, glob_pattern(tmp_path))
    conn.close()
    return db_path


def test_expensive_sessions_cost_ordering_and_pareto():
    """Sessions are ordered by cost desc; cumulative % and [pareto] marker correct."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_expensive_db(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = expensive_sessions(limit=15)

    # Pricey session should come first ($20), mid second ($10), cheap third ($5)
    # Use full session IDs to avoid prefix collisions
    pricey_pos = result.index(_SID_PRICEY)
    mid_pos = result.index(_SID_MID)
    cheap_pos = result.index(_SID_CHEAP)
    assert pricey_pos < mid_pos < cheap_pos

    # Total should be $35
    assert "$35.00" in result

    # Both pricey and mid should be in pareto (20+10=$30 > 80% of $35=$28)
    # Cheap is below the cutoff
    # The second session (mid) should cross 80% (cumulative 30/35 = 85.7%)
    assert "[pareto]" in result or "[pareto, crosses 80%]" in result

    # Cumulative % for pricey session: 20/35 ≈ 57%
    # Check that cum 57% appears near the pricey session
    assert "cum 57%" in result

    # The mid session tips over 80%: cumulative 30/35 ≈ 85%
    assert "[pareto, crosses 80%]" in result


def test_expensive_sessions_since_filters():
    """since= excludes older sessions and changes totals."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Pricey on 2026-05-01, mid on 2026-04-01
        _make_cost_session(
            tmp_path,
            _SID_PRICEY,
            4_000_000,
            timestamp_day="2026-05-01",
            msg_id_suffix="-p",
        )
        _make_cost_session(
            tmp_path,
            _SID_MID,
            2_000_000,
            timestamp_day="2026-04-01",
            msg_id_suffix="-m",
        )
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result_all = expensive_sessions(limit=15)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result_since = expensive_sessions(limit=15, since="2026-05-01")

    # All: both sessions visible, total $30
    assert "$30.00" in result_all

    # Since May 1: only pricey ($20)
    assert "$20.00" in result_since
    assert _SID_MID not in result_since
    assert "(since 2026-05-01)" in result_since


def test_expensive_sessions_limit_clamps_display_not_totals():
    """limit= truncates display rows but header totals reflect all sessions."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = _materialize_expensive_db(tmp_path)

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = expensive_sessions(limit=1)

    # Only first session displayed
    assert _SID_PRICEY in result
    assert _SID_MID not in result
    assert _SID_CHEAP not in result

    # But header total covers all 3 sessions and $35
    assert "$35.00" in result
    assert "3 sessions" in result


def test_expensive_sessions_invalid_since():
    """Invalid since returns Error: string."""
    result = expensive_sessions(since="last week")
    assert result.startswith("Error:")
    assert "since" in result


def test_expensive_sessions_no_cost_data():
    """When no sessions have cost, returns friendly message."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Write a session with no usage (no cost)
        lines = [
            make_user_message(
                "sess-no-cost-aaaa",
                "u1",
                None,
                "2026-05-01T10:00:00.000Z",
                "hello",
                tool_use_result={"content": "seed"},
            ),
            make_assistant_message(
                "sess-no-cost-aaaa",
                "a1",
                "u1",
                "2026-05-01T10:00:01.000Z",
                [{"type": "text", "text": "hi"}],
                msg_id="msg-nocost-a1",
            ),
        ]
        write_jsonl(tmp_path, "sess-no-cost-aaaa", lines)
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = expensive_sessions()

    assert "No sessions with cost found" in result


def test_expensive_sessions_split_and_shape_lines():
    """Split and shape lines appear when cache/output tokens exist."""
    sid = "sess-split-shape-aaaa-aaaaaaaaaaa"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Session with cache_creation tokens so split is non-zero
        _make_cost_session(
            tmp_path,
            sid,
            1_000_000,
            msg_id_suffix="-s",
            cache_creation_tokens=500_000,
            output_tokens=200_000,
        )
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = expensive_sessions(limit=5)

    # Split line should appear with cache write and output segments
    assert "split: cache read" in result
    assert "cache write" in result
    assert "output" in result

    # Shape line: two messages 30 min apart → not "single message"
    assert "shape:" in result
    assert "single message" not in result
    assert "min" in result


def test_expensive_sessions_subagent_flag():
    """Subagent flag appears for sessions with a sidechain message."""
    sid_sub = "sess-subagent-aaaaa-aaaa-aaaaaaaaaa"
    sid_normal = "sess-normal-aaaaa-aaaa-aaaaaaaaaa"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Session with a sidechain assistant message
        lines_sub = [
            make_user_message(
                sid_sub,
                "u1",
                None,
                "2026-05-01T10:00:00.000Z",
                "go",
                tool_use_result={"content": "seed"},
            ),
            make_assistant_message(
                sid_sub,
                "a1",
                "u1",
                "2026-05-01T10:00:01.000Z",
                [{"type": "text", "text": "ok"}],
                model="claude-opus-4-7",
                msg_id="msg-sub-a1",
                usage={"input_tokens": 2_000_000, "output_tokens": 0},
                is_sidechain=True,
            ),
        ]
        write_jsonl(tmp_path, sid_sub, lines_sub)

        # Normal session (cheaper, so it appears second)
        _make_cost_session(tmp_path, sid_normal, 1_000_000, msg_id_suffix="-n")

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()

        with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
            mock_conn.return_value = connect_read_hardened(db_path)
            result = expensive_sessions(limit=5)

    # The subagent session block should say subagents=yes
    sub_block_start = result.index(sid_sub[:8])
    # Find the next blank-separated block boundary or end
    next_block = result.find("\n\n", sub_block_start)
    sub_block = (
        result[sub_block_start:next_block]
        if next_block != -1
        else result[sub_block_start:]
    )
    assert "subagents=yes" in sub_block


def test_expensive_sessions_instructions_mention():
    """INSTRUCTIONS string mentions expensive_sessions."""
    instructions = create_mcp_server().instructions
    assert instructions is not None
    assert "expensive_sessions" in instructions
