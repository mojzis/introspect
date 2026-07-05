"""Tests for the session Trajectory tab."""

import json
import tempfile
from pathlib import Path

import duckdb

from introspect.api.handlers.trajectory import (
    _classify,
    _detail_label,
    _is_verify,
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


def test_is_verify_signal():
    # test runners land in the `test` category and count directly
    assert _is_verify("test", "Bash", "uv run pytest") is True
    # linters / type-checkers classify as pkg/bash but are still verify signals
    assert _is_verify("pkg", "Bash", "uv run ruff check .") is True
    assert _is_verify("pkg", "Bash", "uv run ty check") is True
    assert _is_verify("pkg", "Bash", "npx tsc --noEmit") is True
    assert _is_verify("bash", "Bash", "cargo clippy") is True
    assert _is_verify("test", "Bash", "go test ./...") is True
    # plain edits / unrelated commands are not
    assert _is_verify("edit", "Edit", "") is False
    assert _is_verify("bash", "Bash", "echo hi") is False
    assert _is_verify("pkg", "Bash", "uv sync") is False
    # a search that merely names a checker is navigation, not verification —
    # even when _classify mislabels `rg pytest` as the `test` category
    assert _is_verify("search", "Bash", "grep -rn ruff src/") is False
    assert _is_verify("test", "Bash", "rg pytest tests/") is False


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


def test_build_test_before_edit_stays_out_of_verify():
    # a test that precedes the (final) edit is not a verify signal: the edit
    # is last, so there's no trailing verify band and the edit stays in implement
    ctx = build_trajectory_context(
        _conn(
            [
                ("Bash", {"command": "uv run pytest"}, None),
                ("Edit", {"file_path": "/a.py"}, None),
            ]
        ),
        "s",
    )
    # locate is the pytest run, implement is the edit, verify band is empty
    assert ctx["phases"] == {"locate_end": 1, "implement_end": 2}


def test_build_early_test_does_not_open_verify_early():
    # TDD shape: write test -> run it (fails) -> implement -> run it (passes).
    # Verify must be only the final run, not everything after the first test.
    ctx = build_trajectory_context(
        _conn(
            [
                ("Write", {"file_path": "/test_a.py"}, None),  # 0 write test
                ("Bash", {"command": "uv run pytest"}, None),  # 1 red run
                ("Edit", {"file_path": "/a.py"}, None),  # 2 implement
                ("Bash", {"command": "uv run pytest"}, None),  # 3 green run
            ]
        ),
        "s",
    )
    # empty locate, implement spans write/test/edit, verify is only the green run
    assert ctx["phases"] == {"locate_end": 0, "implement_end": 3}


def test_build_lint_run_after_edits_starts_verify():
    # a linter / type-checker (not a test runner) is still a verify signal
    ctx = build_trajectory_context(
        _conn(
            [
                ("Edit", {"file_path": "/a.py"}, None),
                ("Bash", {"command": "uv run ruff check ."}, None),
            ]
        ),
        "s",
    )
    assert ctx["phases"] == {"locate_end": 0, "implement_end": 1}


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
