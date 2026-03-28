"""Tests for introspect web UI routes."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from introspect.api.main import app

from .conftest import (
    glob_pattern,
    make_assistant_message,
    make_user_message,
    write_jsonl,
)

SID = "01234567-abcd-abcd-abcd-0123456789ab"


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
        make_assistant_message(
            SID,
            "a3",
            "u2",
            "2026-03-27T10:00:04.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp1",
                    "name": "mcp__github__get_me",
                    "input": {},
                }
            ],
        ),
        make_user_message(
            SID,
            "u3",
            "a3",
            "2026-03-27T10:00:05.000Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_mcp1",
                    "content": '{"login": "test"}',
                    "is_error": False,
                }
            ],
            tool_use_result={},
            source_tool_uuid="a3",
        ),
    ]
    return write_jsonl(tmp_dir, SID, lines)


@contextmanager
def _patched_client(tmp_path: Path):
    """Context manager that yields a TestClient with materialized test data."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"

    with (
        patch.dict(
            os.environ,
            {
                "INTROSPECT_DB_PATH": str(db_path),
                "INTROSPECT_JSONL_GLOB": glob_pattern(tmp_path),
                "INTROSPECT_DAYS": "0",
            },
        ),
        TestClient(app) as client,
    ):
        yield client


def test_dashboard_returns_200():
    """Dashboard page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Dashboard" in response.text


def test_sessions_returns_200():
    """Sessions page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200


def test_session_detail_returns_200():
    """Session detail page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions/01234567-abcd-abcd-abcd-0123456789ab")
        assert response.status_code == 200


def test_search_returns_200():
    """Search page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/search")
        assert response.status_code == 200


def test_tools_returns_200():
    """Tools page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/tools")
        assert response.status_code == 200


def test_mcps_returns_200():
    """MCPs page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/mcps")
        assert response.status_code == 200
        assert "MCP Servers" in response.text


def test_mcps_filter_by_server():
    """MCPs page filters by server name."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/mcps?server=github")
        assert response.status_code == 200
        assert "github" in response.text


def test_base_template_has_history_restore_spinner_fix():
    """Regression: htmx:historyRestore listener must clear the loading spinner."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions")
        assert response.status_code == 200
        assert "htmx:historyRestore" in response.text
        assert "loading-overlay" in response.text


def test_htmx_partial_response_excludes_spinner():
    """HTMX partial responses should not contain the loading overlay."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/sessions", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "loading-overlay" not in response.text


def test_stats_returns_200():
    """Stats page loads without error."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        response = client.get("/stats")
        assert response.status_code == 200


def test_mcp_endpoint_mounted():
    """MCP streamable-HTTP endpoint is reachable."""
    with tempfile.TemporaryDirectory() as tmp, _patched_client(Path(tmp)) as client:
        # The streamable-http app mounts its route at /mcp internally,
        # so full path is /mcp/mcp. We expect a non-404 response
        # (405 Method Not Allowed or 421 from MCP transport security).
        response = client.get("/mcp/mcp")
        assert response.status_code != 404
