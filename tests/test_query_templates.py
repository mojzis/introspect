"""Tests for the query-template registry and its cookbook/discoverability
adapters.

- Integrity: every registry entry's SQL binds and executes against the
  fixture DB without error.
- Cookbook render: ``list_query_templates`` renders every entry and filters
  by ``kind``.
- Discoverability: ``describe_schema`` mentions the template count.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from introspect.db import materialize_views
from introspect.mcp.tools import describe_schema, list_query_templates
from introspect.query_templates import QUERY_TEMPLATES
from introspect.search import build_search_corpus

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "qt-fixture-session"
SEARCH_WORD = "refactor"


def _write_fixture_jsonl(tmp_dir: Path) -> None:
    """Write a session with cost, a tool success + a tool failure, and
    searchable text — enough surface for all four templates to bind and
    execute.
    """
    usage_a1 = {
        "input_tokens": 1_000_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }
    usage_a2 = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }
    lines = [
        make_user_message(
            SID,
            "u1",
            None,
            "2026-04-01T10:00:00.000Z",
            "Help me refactor the database module",
        ),
        make_assistant_message(
            SID,
            "a1",
            "u1",
            "2026-04-01T10:00:01.000Z",
            [{"type": "text", "text": "Sure, looking into it now."}],
            model="claude-opus-4-7",
            msg_id="qt-msg-a1",
            usage=usage_a1,
        ),
        make_assistant_message(
            SID,
            "a2",
            "a1",
            "2026-04-01T10:00:02.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_ok",
                    "name": "Bash",
                    "input": {"command": "ls -la"},
                }
            ],
            model="claude-opus-4-7",
            msg_id="qt-msg-a2",
            usage=usage_a2,
        ),
        make_user_message(
            SID,
            "u2",
            "a2",
            "2026-04-01T10:00:03.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_ok",
                    "content": "total 0",
                    "is_error": False,
                }
            ],
            tool_use_result={"stdout": "total 0", "stderr": ""},
            source_tool_uuid="a2",
        ),
        make_assistant_message(
            SID,
            "a3",
            "u2",
            "2026-04-01T10:00:04.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_fail",
                    "name": "Bash",
                    "input": {"command": "rm -rf /oops"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3",
            "a3",
            "2026-04-01T10:00:05.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_fail",
                    "content": "Permission denied",
                    "is_error": True,
                }
            ],
            tool_use_result={"stdout": "", "stderr": "Permission denied"},
            source_tool_uuid="a3",
        ),
    ]
    write_jsonl(tmp_dir, SID, lines)


@pytest.fixture
def fixture_db_path() -> Iterator[Path]:
    """Materialize the fixture session into a DuckDB file and yield its path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fixture_jsonl(tmp_path)
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        build_search_corpus(conn)
        conn.close()
        yield db_path


# Sample $named params per template, keyed by template name, plus the
# minimum row count the fixture data guarantees for that query — proves the
# SQL doesn't just parse but actually returns matching data.
_SAMPLE_PARAMS: dict[str, dict[str, object]] = {
    "expensive_sessions": {"limit": 5, "since": None},
    "tool_failure_rate": {"limit": 5, "since": None, "min_calls": 1},
    "session_cost_tail": {"session_id": SID},
    "topic_to_cost": {"query": SEARCH_WORD, "limit": 5},
}
_MIN_EXPECTED_ROWS: dict[str, int] = {
    "expensive_sessions": 1,  # fixture session has cost
    "tool_failure_rate": 1,  # Bash has one failure out of two calls
    "session_cost_tail": 2,  # two cost-bearing assistant messages
    "topic_to_cost": 1,  # SEARCH_WORD appears in the fixture's first prompt
}


@pytest.mark.parametrize("template", QUERY_TEMPLATES, ids=lambda t: t.name)
def test_template_sql_executes(fixture_db_path: Path, template) -> None:
    """Every registry entry's SQL binds, executes, and returns the rows the
    fixture guarantees — proving the SQL matches real data, not just that it
    parses."""
    params = _SAMPLE_PARAMS[template.name]
    conn = duckdb.connect(str(fixture_db_path), read_only=True)
    try:
        rows = conn.execute(template.sql, params).fetchall()
    finally:
        conn.close()
    assert len(rows) >= _MIN_EXPECTED_ROWS[template.name]


def test_all_template_names_have_sample_params() -> None:
    """Guard against a registry entry silently missing test coverage."""
    registry_names = {t.name for t in QUERY_TEMPLATES}
    assert registry_names == set(_SAMPLE_PARAMS)


def test_list_query_templates_renders_all_entries() -> None:
    """list_query_templates() includes every entry's name and note."""
    result = list_query_templates()
    for template in QUERY_TEMPLATES:
        assert template.name in result
        assert template.note in result


def test_list_query_templates_frames_as_starting_points() -> None:
    """The cookbook explicitly tells the model these are adapt-and-run, not
    canned answers."""
    result = list_query_templates()
    assert "not canned answers" in result
    assert "run_sql" in result


def test_list_query_templates_filters_by_kind() -> None:
    """kind='deterministic' includes tool_failure_rate, excludes
    session_cost_tail (exploratory)."""
    result = list_query_templates(kind="deterministic")
    assert "tool_failure_rate" in result
    assert "session_cost_tail" not in result


def test_list_query_templates_invalid_kind_returns_friendly_error() -> None:
    """An unknown kind returns an Error: string, not a traceback."""
    result = list_query_templates(kind="bogus")
    assert result.startswith("Error:")
    assert "deterministic" in result
    assert "exploratory" in result


def test_describe_schema_mentions_query_templates(
    monkeypatch: pytest.MonkeyPatch, fixture_db_path: Path
) -> None:
    """describe_schema() points at list_query_templates() with the count."""
    monkeypatch.setattr(
        "introspect.mcp.tools.get_read_connection",
        lambda: duckdb.connect(str(fixture_db_path), read_only=True),
    )
    result = describe_schema()
    assert f"{len(QUERY_TEMPLATES)} query templates" in result
    assert "list_query_templates()" in result
