"""Tests for the session Trajectory tab."""

import tempfile
from pathlib import Path

from introspect.api.handlers.trajectory import (
    _classify,
    _detail_label,
    _prefix,
)

from .conftest import SID, _patched_client


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
