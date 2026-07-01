"""Tests for the session Trajectory tab."""

import json
import tempfile
from pathlib import Path

import duckdb

from introspect.api.handlers.trajectory import (
    _classify,
    _detail_label,
    _prefix,
    _tooltip,
    build_trajectory_context,
)

from .conftest import SID, _patched_client


def _conn(calls: list[tuple[str, dict, str | None]]) -> duckdb.DuckDBPyConnection:
    """In-memory ``tool_calls`` table from ``(tool_name, input, is_error)`` rows.

    ``is_error`` mirrors DuckDB's JSON text: ``'true'`` / ``'false'`` / ``None``.
    Rows are ordered by insertion (called_at + tool_use_id both increment).
    """
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE tool_calls (
            session_id VARCHAR, called_at BIGINT, tool_use_id VARCHAR,
            tool_name VARCHAR, tool_input VARCHAR, is_error VARCHAR
        )
        """
    )
    for i, (tool_name, inp, is_error) in enumerate(calls):
        conn.execute(
            "INSERT INTO tool_calls VALUES ('s', ?, ?, ?, ?, ?)",
            [i, f"tu{i:03d}", tool_name, json.dumps(inp), is_error],
        )
    return conn


def test_classify_tool_names():
    assert _classify("Read", "") == "read"
    assert _classify("Edit", "") == "edit"
    assert _classify("MultiEdit", "") == "edit"
    assert _classify("Write", "") == "write"
    assert _classify("NotebookEdit", "") == "write"
    assert _classify("Task", "") == "task"
    assert _classify("WebSearch", "") == "web"
    assert _classify("mcp__github__get_me", "") == "mcp"
    assert _classify("TodoWrite", "") == "other"


def test_classify_bash_subcategories():
    assert _classify("Bash", "uv run pytest tests/") == "test"
    assert _classify("Bash", "poe check") == "test"
    assert _classify("Bash", "grep -rn foo src/") == "search"
    assert _classify("Bash", "rg foo") == "search"
    assert _classify("Bash", "git status") == "git"
    assert _classify("Bash", "uv sync") == "pkg"
    assert _classify("Bash", "echo hello") == "bash"


def test_classify_bash_first_match_wins():
    # command greps AND runs pytest -> primary intent is the test run
    assert _classify("Bash", "grep -q x && pytest") == "test"


def test_prefix():
    assert _prefix("git status --short") == "git status"
    assert _prefix("ls") == "ls"
    assert _prefix("") == "(empty)"


def test_detail_label():
    assert _detail_label("Bash", "git", {"command": "git status -s"}) == "git status"
    assert _detail_label("Read", "read", {"file_path": "/a/b/main.py"}) == "main.py"
    assert _detail_label("mcp__github__get_me", "mcp", {}) == "github__get_me"
    # non-file, non-bash, non-mcp tool falls through to the bare name
    assert _detail_label("Task", "task", {}) == "Task"


def test_tooltip():
    assert _tooltip("Bash", {"command": "  ls -la  "}) == "ls -la"
    assert _tooltip("Bash", {}) == "Bash"
    assert _tooltip("Read", {"file_path": "/a/b.py"}) == "Read: /a/b.py"
    assert _tooltip("mcp__github__get_me", {}) == "mcp__github__get_me"
    assert _tooltip("Task", {}) == "Task"


def test_build_empty_session():
    ctx = build_trajectory_context(_conn([]), "s")
    assert ctx == {"has_data": False, "view": "category"}


def test_build_normal_locate_implement_verify():
    # read (locate) -> edit (implement) -> test (verify)
    ctx = build_trajectory_context(
        _conn(
            [
                ("Read", {"file_path": "/a.py"}, None),
                ("Edit", {"file_path": "/a.py"}, None),
                ("Bash", {"command": "uv run pytest"}, None),
            ]
        ),
        "s",
    )
    assert ctx["phases"] == {"locate_end": 1, "implement_end": 2}
    assert ctx["metrics"]["locate_len"] == 1
    assert ctx["metrics"]["edits"] == 1


def test_build_edits_but_no_tests_leaves_verify_band_empty():
    ctx = build_trajectory_context(
        _conn(
            [
                ("Read", {"file_path": "/a.py"}, None),
                ("Edit", {"file_path": "/a.py"}, None),
                ("Read", {"file_path": "/a.py"}, None),
            ]
        ),
        "s",
    )
    # verify starts at end -> [3, 3) is empty
    assert ctx["phases"] == {"locate_end": 1, "implement_end": 3}


def test_build_no_edits_puts_everything_in_locate():
    ctx = build_trajectory_context(
        _conn(
            [
                ("Read", {"file_path": "/a.py"}, None),
                ("Bash", {"command": "grep foo"}, None),
                ("Read", {"file_path": "/b.py"}, None),
            ]
        ),
        "s",
    )
    assert ctx["phases"] == {"locate_end": 3, "implement_end": 3}
    assert ctx["metrics"]["locate_len"] == 3
    assert ctx["metrics"]["edits"] == 0


def test_build_test_before_edit_collapses_implement_band():
    # test precedes the first edit -> implement band [1, 1) is empty
    ctx = build_trajectory_context(
        _conn(
            [
                ("Bash", {"command": "uv run pytest"}, None),
                ("Edit", {"file_path": "/a.py"}, None),
            ]
        ),
        "s",
    )
    assert ctx["phases"] == {"locate_end": 1, "implement_end": 1}


def test_build_reread_metrics_and_is_error():
    ctx = build_trajectory_context(
        _conn(
            [
                ("Read", {"file_path": "/hot.py"}, None),
                ("Read", {"file_path": "/hot.py"}, None),
                ("Bash", {"command": "false"}, "true"),
            ]
        ),
        "s",
    )
    assert ctx["metrics"]["distinct_files"] == 1
    assert ctx["metrics"]["max_rereads"] == 2
    assert ctx["calls"][2]["is_error"] is True
    assert ctx["calls"][0]["is_error"] is False


def test_build_invalid_view_falls_back_to_default():
    ctx = build_trajectory_context(_conn([]), "s", view="bogus")
    assert ctx["view"] == "category"


def test_trajectory_tab_renders():
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=trajectory")
        assert response.status_code == 200
        text = response.text
        assert "Trajectory" in text
        # sample fixture has tool calls -> a phase band + metrics render
        assert "Locate len" in text
        assert "locate" in text


def test_trajectory_tab_in_strip():
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=messages")
        assert response.status_code == 200
        assert "tab=trajectory" in response.text


def test_trajectory_detail_view_toggle():
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get(f"/sessions/{SID}?tab=trajectory&view=detail")
        assert response.status_code == 200
        # detail view labels Bash calls by prefix; fixture runs "echo hello"
        assert "echo hello" in response.text


def test_trajectory_empty_session():
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        missing = "ffffffff-0000-0000-0000-000000000000"
        response = client.get(f"/sessions/{missing}?tab=trajectory")
        assert response.status_code == 200
        assert "not found" in response.text or "nothing to plot" in response.text
