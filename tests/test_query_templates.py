"""Tests for the query-template registry and its cookbook/discoverability
adapters.

- Integrity: every registry entry's SQL binds and executes against the
  fixture DB without error.
- Cookbook render: ``list_query_templates`` renders every entry and filters
  by ``kind``.
- Discoverability: ``describe_schema`` mentions the template count.
- Parity: the ``tool_failure_rate`` MCP tool's output reflects exactly what
  the registry's SQL returns when run directly — tool and cookbook can't
  drift apart.
- Behavior: ``tool_failure_rate``'s failure-rate math and ``min_calls``
  noise filter.
- Exploratory prompts: ``session_cost_tail``/``topic_to_cost`` render seed
  text from the registry and are discoverable via ``list_prompts``.
"""

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest
from mcp.types import GetPromptResult, TextContent

from introspect.db import materialize_views
from introspect.mcp.server import create_mcp_server
from introspect.mcp.tools import (
    describe_schema,
    list_query_templates,
    tool_failure_rate,
)
from introspect.query_templates import QUERY_TEMPLATES, QueryTemplate, get_template
from introspect.search import build_search_corpus

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_attachment_message,
    make_user_message,
    write_jsonl,
)

SID = "qt-fixture-session"
SEARCH_WORD = "refactor"
RATE_SID = "qt-rate-session"


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
    "first_prompt_triggers": {"limit": 5, "project": None},
}
_MIN_EXPECTED_ROWS: dict[str, int] = {
    "expensive_sessions": 1,  # fixture session has cost
    "tool_failure_rate": 1,  # Bash has one failure out of two calls
    "session_cost_tail": 2,  # two cost-bearing assistant messages
    "topic_to_cost": 1,  # SEARCH_WORD appears in the fixture's first prompt
    "first_prompt_triggers": 1,  # fixture session has a first prompt
}


@pytest.mark.parametrize("template", QUERY_TEMPLATES, ids=lambda t: t.name)
def test_template_sql_executes(fixture_db_path: Path, template: QueryTemplate) -> None:
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


def test_get_template_lookup_hit_and_miss() -> None:
    """get_template() returns the matching entry by name, or None for an
    unknown name."""
    template = get_template("tool_failure_rate")
    assert template is not None
    assert template.name == "tool_failure_rate"

    assert get_template("not_a_real_template") is None


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


# --- tool_failure_rate: parity + behavior -----------------------------------

# (tool_name, tool_use_id, tool_input, is_error). Bash: 4 calls, 2 failures
# (50%). Read: 2 calls, 0 failures — below a min_calls=3 threshold, so it
# exercises the noise filter without ever showing a failure of its own.
_RATE_CALLS: tuple[tuple[str, str, dict, bool], ...] = (
    ("Bash", "toolu_b1", {"command": "ls"}, False),
    ("Bash", "toolu_b2", {"command": "false"}, True),
    ("Bash", "toolu_b3", {"command": "pwd"}, False),
    ("Bash", "toolu_b4", {"command": "exit 1"}, True),
    ("Read", "toolu_r1", {"file_path": "/tmp/a"}, False),
    ("Read", "toolu_r2", {"file_path": "/tmp/b"}, False),
)


def _write_failure_rate_jsonl(tmp_dir: Path) -> None:
    """Write a session with two tools at different call volumes — enough to
    verify failure-rate math (Bash) and the min_calls noise filter (Read).
    """
    lines: list[dict] = [
        make_user_message(
            RATE_SID, "u0", None, "2026-05-01T09:00:00.000Z", "Run some commands"
        )
    ]
    parent = "u0"
    second = 0
    for i, (tool_name, tool_use_id, tool_input, is_error) in enumerate(_RATE_CALLS):
        second += 1
        a_uuid = f"a{i}"
        lines.append(
            make_assistant_message(
                RATE_SID,
                a_uuid,
                parent,
                f"2026-05-01T09:00:{second:02d}.000Z",
                [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                ],
                msg_id=f"qt-rate-msg-{i}",
            )
        )
        second += 1
        u_uuid = f"u{i + 1}"
        lines.append(
            make_user_message(
                RATE_SID,
                u_uuid,
                a_uuid,
                f"2026-05-01T09:00:{second:02d}.000Z",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "error" if is_error else "ok",
                        "is_error": is_error,
                    }
                ],
                tool_use_result={
                    "stdout": "" if is_error else "ok",
                    "stderr": "boom" if is_error else "",
                },
                source_tool_uuid=a_uuid,
            )
        )
        parent = u_uuid
    write_jsonl(tmp_dir, RATE_SID, lines)


@pytest.fixture
def failure_rate_db_path() -> Iterator[Path]:
    """Materialize the two-tool failure-rate fixture into a DuckDB file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_failure_rate_jsonl(tmp_path)
        db_path = tmp_path / "rate.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()
        yield db_path


def test_tool_failure_rate_parity_with_registry_sql(
    failure_rate_db_path: Path,
) -> None:
    """The tool's reported failures/calls/failure_rate for every tool match
    exactly what running the registry's SQL directly (bound dict) returns —
    proving the tool can't drift from the cookbook entry."""
    template = get_template("tool_failure_rate")
    assert template is not None
    params = {"limit": 20, "since": None, "min_calls": 1}

    conn = duckdb.connect(str(failure_rate_db_path), read_only=True)
    try:
        direct_rows = conn.execute(template.sql, params).fetchall()
    finally:
        conn.close()

    assert direct_rows  # sanity: the fixture actually produced rows

    with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
        mock_conn.return_value = duckdb.connect(
            str(failure_rate_db_path), read_only=True
        )
        result = tool_failure_rate(limit=20, since="", min_calls=1)

    for tool_name, calls, failures, failure_rate in direct_rows:
        expected_line = (
            f"{tool_name}: {failures}/{calls} failed ({float(failure_rate):.1%})"
        )
        assert expected_line in result


def test_tool_failure_rate_counts_and_rate(failure_rate_db_path: Path) -> None:
    """N calls / K failures yields the right failures, calls, and rate."""
    with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
        mock_conn.return_value = duckdb.connect(
            str(failure_rate_db_path), read_only=True
        )
        result = tool_failure_rate(min_calls=1)

    assert "Bash: 2/4 failed (50.0%)" in result
    assert "Read: 0/2 failed (0.0%)" in result


def test_tool_failure_rate_min_calls_suppresses_low_n(
    failure_rate_db_path: Path,
) -> None:
    """min_calls filters out tools below the call-count threshold."""
    with patch("introspect.mcp.tools.get_read_connection") as mock_conn:
        mock_conn.return_value = duckdb.connect(
            str(failure_rate_db_path), read_only=True
        )
        result = tool_failure_rate(min_calls=3)

    assert "Bash" in result
    assert "Read" not in result


def test_tool_failure_rate_invalid_since() -> None:
    """An unparseable `since` returns a friendly Error: string, not a
    traceback — mirrors expensive_sessions' validation."""
    result = tool_failure_rate(since="not-a-date")
    assert result.startswith("Error:")


# --- exploratory prompts: render + discoverability --------------------------

# Sample args per exploratory prompt, mirroring _SAMPLE_PARAMS above.
_PROMPT_SAMPLE_ARGS: dict[str, dict[str, object]] = {
    "session_cost_tail": {"session_id": SID},
    "topic_to_cost": {"query": SEARCH_WORD},
    "first_prompt_triggers": {},
}


def _prompt_text(result: GetPromptResult) -> str:
    """Extract the rendered text from a single-message prompt result.

    Every introspect prompt fn returns a plain `str`, which FastMCP wraps as
    one `TextContent`-bearing message — narrows the `content` union so ty
    doesn't flag `.text` as missing on the other content variants.
    """
    content = result.messages[0].content
    assert isinstance(content, TextContent)
    return content.text


@pytest.mark.parametrize("name", sorted(_PROMPT_SAMPLE_ARGS), ids=lambda n: n)
def test_prompt_renders_registry_sql_and_note(name: str) -> None:
    """Each exploratory prompt's seed message contains its registry entry's
    canonical SQL and note — the seed is built FROM the registry, not
    hand-duplicated."""
    template = get_template(name)
    assert template is not None

    mcp = create_mcp_server()
    result = asyncio.run(mcp.get_prompt(name, _PROMPT_SAMPLE_ARGS[name]))

    text = _prompt_text(result)
    assert template.sql in text
    assert template.note in text


def test_prompt_frames_as_starting_point() -> None:
    """The seed message explicitly says it's a starting point to adapt, not
    a canned answer — the spec's anti-anchoring mitigation."""
    mcp = create_mcp_server()
    result = asyncio.run(mcp.get_prompt("session_cost_tail", {"session_id": SID}))
    text = _prompt_text(result)
    assert "starting point to adapt" in text
    assert "not a canned answer" in text


def test_list_prompts_includes_all_exploratory_prompts() -> None:
    """Every exploratory-template prompt is discoverable via list_prompts,
    matching the registry's exploratory entries."""
    mcp = create_mcp_server()
    prompts = asyncio.run(mcp.list_prompts())
    names = {p.name for p in prompts}
    assert names == {t.name for t in QUERY_TEMPLATES if t.kind == "exploratory"}


# --- first_prompt_triggers: path-extraction behavior ------------------------

# (session_id, first_prompt, expected referenced_paths). Covers @mentions,
# rooted + relative paths, and the prose false-positive ("and/or") the regex
# must NOT catch — the heuristic's whole value is separating real file refs
# from incidental slashes in prose.
_TRIGGER_CASES: tuple[tuple[str, str, list[str]], ...] = (
    (
        "fpt-refs",
        "please look at @src/db.py and ./scripts/run.sh now",
        ["@src/db.py", "./scripts/run.sh"],
    ),
    ("fpt-prose", "use the right skill / claude file, and/or the project", []),
    ("fpt-none", "just a plain question with no paths at all", []),
)


def _write_trigger_jsonl(tmp_dir: Path) -> None:
    """Write one session per _TRIGGER_CASES entry, each a single human prompt,
    so first_prompt_triggers' path regex can be exercised end to end. Each
    session lives in its own file (one sessionId per file).

    Also writes a schema-anchor session carrying a tool_use_result so the
    union-by-name JSONL load produces every column raw_messages references
    (``toolUseResult`` in particular); a corpus of prompt-only sessions would
    otherwise fail to bind. The anchor is not part of _TRIGGER_CASES, so it
    doesn't affect the assertions.
    """
    for sid, prompt, _expected in _TRIGGER_CASES:
        write_jsonl(
            tmp_dir,
            sid,
            [
                make_user_message(
                    sid, f"{sid}-u1", None, "2026-06-01T10:00:00.000Z", prompt
                )
            ],
        )
    anchor = "fpt-anchor"
    write_jsonl(
        tmp_dir,
        anchor,
        [
            make_user_message(
                anchor, "an-u0", None, "2026-06-09T10:00:00.000Z", "anchor session"
            ),
            make_assistant_message(
                anchor,
                "an-a1",
                "an-u0",
                "2026-06-09T10:00:01.000Z",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_an",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            ),
            make_user_message(
                anchor,
                "an-u1",
                "an-a1",
                "2026-06-09T10:00:02.000Z",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_an",
                        "content": "ok",
                        "is_error": False,
                    }
                ],
                tool_use_result={"stdout": "ok", "stderr": ""},
                source_tool_uuid="an-a1",
            ),
        ],
    )


@pytest.fixture
def trigger_db_path() -> Iterator[Path]:
    """Materialize the path-extraction fixture sessions into a DuckDB file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_trigger_jsonl(tmp_path)
        db_path = tmp_path / "triggers.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()
        yield db_path


def test_first_prompt_triggers_extracts_paths(trigger_db_path: Path) -> None:
    """The path regex captures real file/dir references and ignores prose
    slashes like 'and/or'."""
    template = get_template("first_prompt_triggers")
    assert template is not None

    conn = duckdb.connect(str(trigger_db_path), read_only=True)
    try:
        rows = conn.execute(template.sql, {"limit": 50, "project": None}).fetchall()
    finally:
        conn.close()

    # Map session_id -> (n_referenced_paths, referenced_paths).
    by_session = {r[0]: (r[3], r[4]) for r in rows}
    for sid, _prompt, expected in _TRIGGER_CASES:
        n_paths, paths = by_session[sid]
        assert list(paths) == expected, sid
        assert n_paths == len(expected), sid


def test_first_prompt_triggers_auto_load_rollup() -> None:
    """The auto_loaded_* columns roll up session_context_loads: a session that
    auto-loaded a nested CLAUDE.md, expanded an @-file, and showed the skill
    menu reports all three; a bare session reports the COALESCEd defaults."""
    loaded_sid, bare_sid = "fpt-loaded", "fpt-bare"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_jsonl(
            tmp_path,
            loaded_sid,
            [
                make_user_message(
                    loaded_sid,
                    f"{loaded_sid}-u1",
                    None,
                    "2026-06-01T10:00:00.000Z",
                    "look at @src/db.py",
                    # Anchor the toolUseResult column for the JSONL union load.
                    tool_use_result={"stdout": "", "stderr": ""},
                ),
                make_attachment_message(
                    loaded_sid,
                    f"{loaded_sid}-att1",
                    f"{loaded_sid}-u1",
                    "2026-06-01T10:00:01.000Z",
                    {
                        "type": "nested_memory",
                        "path": "/repo/.claude/rules/x.md",
                        "displayPath": ".claude/rules/x.md",
                        "content": "rule",
                    },
                ),
                make_attachment_message(
                    loaded_sid,
                    f"{loaded_sid}-att2",
                    f"{loaded_sid}-u1",
                    "2026-06-01T10:00:02.000Z",
                    {
                        "type": "file",
                        "filename": "/repo/src/db.py",
                        "displayPath": "src/db.py",
                        "content": "code",
                    },
                ),
                make_attachment_message(
                    loaded_sid,
                    f"{loaded_sid}-att3",
                    f"{loaded_sid}-u1",
                    "2026-06-01T10:00:03.000Z",
                    {"type": "skill_listing", "content": "skills", "skillCount": 2},
                ),
            ],
        )
        write_jsonl(
            tmp_path,
            bare_sid,
            [
                make_user_message(
                    bare_sid,
                    f"{bare_sid}-u1",
                    None,
                    "2026-06-01T11:00:00.000Z",
                    "just a question",
                )
            ],
        )
        db_path = tmp_path / "rollup.duckdb"
        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pattern(tmp_path))
        conn.close()

        template = get_template("first_prompt_triggers")
        assert template is not None
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = conn.execute(template.sql, {"limit": 50, "project": None}).fetchall()
        finally:
            conn.close()

    # Columns: ..., auto_loaded_claude_md(6), n_auto_loaded_files(7),
    # skill_menu_loaded(8).
    by_session = {r[0]: (r[6], r[7], r[8]) for r in rows}
    assert by_session[loaded_sid] == (True, 1, True)
    assert by_session[bare_sid] == (False, 0, False)
