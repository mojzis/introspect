"""Tests for git worktree-aware project resolution."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb

from introspect.db import materialize_views
from introspect.projects import get_canonical_project, resolve_project_map

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "test-project-session"


def _make_completed_process(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_canonical_project_normal_git_dir():
    """A regular .git dir resolves to the repo root."""
    with patch("introspect.projects.subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process("/home/user/myrepo/.git\n")
        result = get_canonical_project("/home/user/myrepo")
        assert result == "/home/user/myrepo"


def test_canonical_project_worktree():
    """A worktree resolves to the main repo root."""
    with patch("introspect.projects.subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(
            "/home/user/myrepo/.git/worktrees/feature-branch\n"
        )
        result = get_canonical_project("/home/user/worktrees/feature-branch")
        assert result == "/home/user/myrepo"


def test_canonical_project_relative_git_dir():
    """A relative .git path is resolved against the target cwd, not process cwd."""
    with patch("introspect.projects.subprocess.run") as mock_run:
        # git rev-parse --git-common-dir typically returns ".git" (relative)
        mock_run.return_value = _make_completed_process(".git\n")
        result = get_canonical_project("/home/user/some-other-project")
        # Must resolve to the target cwd, not the introspect process cwd
        assert result == "/home/user/some-other-project"


def test_canonical_project_fallback_on_error():
    """Non-git directories fall back to the cwd itself."""
    with patch("introspect.projects.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(128, "git")
        result = get_canonical_project("/tmp/not-a-repo")
        assert result == "/tmp/not-a-repo"


def test_canonical_project_fallback_on_oserror():
    """Missing git binary falls back gracefully."""
    with patch("introspect.projects.subprocess.run") as mock_run:
        mock_run.side_effect = OSError("git not found")
        result = get_canonical_project("/tmp/no-git")
        assert result == "/tmp/no-git"


def test_resolve_project_map_empty():
    """Empty input returns empty mapping."""
    assert resolve_project_map([]) == {}


def test_resolve_project_map_parallel():
    """Multiple cwds are resolved in parallel."""
    with patch("introspect.projects.get_canonical_project") as mock_fn:
        mock_fn.side_effect = lambda cwd: f"/canonical{cwd}"
        result = resolve_project_map(["/a", "/b", "/c"])
        assert result == {
            "/a": "/canonical/a",
            "/b": "/canonical/b",
            "/c": "/canonical/c",
        }
        assert mock_fn.call_count == 3


def _write_sample_jsonl(tmp_dir: Path) -> None:
    lines = [
        make_user_message(SID, "u1", None, "2026-03-27T10:00:00.000Z", "Hello"),
        make_assistant_message(
            SID,
            "a1",
            "u1",
            "2026-03-27T10:00:01.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "echo hi"},
                }
            ],
        ),
        make_user_message(
            SID,
            "u2",
            "a1",
            "2026-03-27T10:00:02.000Z",
            [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi\n"}],
            tool_use_result={"stdout": "hi\n", "stderr": ""},
        ),
    ]
    write_jsonl(tmp_dir, SID, lines)


def test_materialize_creates_project_map():
    """materialize_views creates a project_map table."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)

        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pat, resolve_projects=False)

        rows = conn.execute("SELECT * FROM project_map").fetchall()
        assert len(rows) == 1
        cwd, canonical, name = rows[0]
        assert cwd == "/tmp/test"
        assert canonical == "/tmp/test"
        assert name == "test"
        conn.close()


def test_logical_sessions_has_project_column():
    """logical_sessions view includes the resolved project name."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)

        conn = duckdb.connect(str(db_path))
        materialize_views(conn, glob_pat, resolve_projects=False)

        cols = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'logical_sessions'
            ORDER BY ordinal_position
        """).fetchall()
        col_names = [c[0] for c in cols]
        assert "project" in col_names

        row = conn.execute(
            "SELECT project FROM logical_sessions WHERE session_id = ?", [SID]
        ).fetchone()
        assert row is not None
        assert row[0] == "test"
        conn.close()


def test_materialize_with_resolve_projects():
    """When resolve_projects=True, get_canonical_project is called for each cwd."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_sample_jsonl(tmp_path)
        db_path = tmp_path / "test.duckdb"
        glob_pat = glob_pattern(tmp_path)

        conn = duckdb.connect(str(db_path))
        with patch("introspect.db.resolve_project_map") as mock_resolve:
            mock_resolve.return_value = {"/tmp/test": "/home/user/my-project"}
            materialize_views(conn, glob_pat, resolve_projects=True)

        mock_resolve.assert_called_once_with(["/tmp/test"])

        row = conn.execute(
            "SELECT project FROM logical_sessions WHERE session_id = ?", [SID]
        ).fetchone()
        assert row is not None
        assert row[0] == "my-project"
        conn.close()
