"""Tests for introspect web UI routes."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from introspect.api.main import app
from introspect.db import get_connection


def _write_sample_jsonl(tmp_dir: Path) -> Path:
    """Write a minimal JSONL file for testing."""
    session_id = "01234567-abcd-abcd-abcd-0123456789ab"
    jsonl_path = tmp_dir / "projects" / "test-project" / f"{session_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        {
            "type": "user",
            "timestamp": "2026-03-27T10:00:00.000Z",
            "sessionId": session_id,
            "uuid": "u1",
            "parentUuid": None,
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "message": {"role": "user", "content": "Hello, help me with tests"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-27T10:00:01.000Z",
            "sessionId": session_id,
            "uuid": "a1",
            "parentUuid": "u1",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "requestId": "req1",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "id": "msg1",
                "content": [{"type": "text", "text": "Sure, I can help!"}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-27T10:00:02.000Z",
            "sessionId": session_id,
            "uuid": "a2",
            "parentUuid": "a1",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "requestId": "req2",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "id": "msg2",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_test1",
                        "name": "Bash",
                        "input": {"command": "echo hello", "description": "test"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-03-27T10:00:03.000Z",
            "sessionId": session_id,
            "uuid": "u2",
            "parentUuid": "a2",
            "isSidechain": False,
            "cwd": "/tmp/test",
            "version": "2.1.0",
            "entrypoint": "cli",
            "gitBranch": "main",
            "sourceToolAssistantUUID": "a2",
            "toolUseResult": {
                "stdout": "hello\n",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
                "noOutputExpected": False,
            },
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_test1",
                        "content": "hello\n",
                        "is_error": False,
                    }
                ],
            },
        },
    ]

    with jsonl_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


def _make_client(tmp_path: Path) -> TestClient:
    """Create a test client with a temporary database."""
    _write_sample_jsonl(tmp_path)
    db_path = tmp_path / "test.duckdb"
    glob_pattern = str(tmp_path / "projects" / "**" / "*.jsonl")

    def _patched_get_connection():
        return get_connection(db_path, glob_pattern)

    with patch("introspect.api.main.get_connection", _patched_get_connection):
        return TestClient(app)


def test_dashboard_returns_200():
    """Dashboard page loads without error."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        response = client.get("/")
        assert response.status_code == 200
        assert "Dashboard" in response.text


def test_sessions_returns_200():
    """Sessions page loads without error."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        response = client.get("/sessions")
        assert response.status_code == 200


def test_session_detail_returns_200():
    """Session detail page loads without error."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        response = client.get("/sessions/01234567-abcd-abcd-abcd-0123456789ab")
        assert response.status_code == 200


def test_search_returns_200():
    """Search page loads without error."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        response = client.get("/search")
        assert response.status_code == 200


def test_tools_returns_200():
    """Tools page loads without error."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        response = client.get("/tools")
        assert response.status_code == 200


def test_stats_returns_200():
    """Stats page loads without error."""
    with tempfile.TemporaryDirectory() as tmp:
        client = _make_client(Path(tmp))
        response = client.get("/stats")
        assert response.status_code == 200
